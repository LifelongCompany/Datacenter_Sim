import simpy
from typing import Dict, Any, List, Optional
from .energy_model import HardwareEnergyTracker

class Node:
    def __init__(self, node_id: int, total_cpus: int, total_gpus: int):
        self.node_id = node_id
        self.total_cpus = total_cpus
        self.total_gpus = total_gpus

        self.used_cpus = 0
        self.used_gpus = 0

        self.active_tasks = 0
        self.io_blocked_tasks = 0

    def can_allocate(self, cpu_req: int, gpu_req: int) -> bool:
        return (self.total_cpus - self.used_cpus) >= cpu_req and \
               (self.total_gpus - self.used_gpus) >= gpu_req

    def allocate(self, cpu_req: int, gpu_req: int):
        self.used_cpus += cpu_req
        self.used_gpus += gpu_req

    def release(self, cpu_req: int, gpu_req: int):
        self.used_cpus -= cpu_req
        self.used_gpus -= gpu_req

    @property
    def is_stranded(self) -> bool:
        # A node has stranded GPUs if CPU is exhausted (or near exhausted preventing typical tasks)
        # but there are GPUs left.
        # Let's define stranded: CPU utilization is 100% (or very high, leaving < 4 cores)
        # and there are unused GPUs.
        # Since minimum CPU request is 4, < 4 means no other GPU tasks can be scheduled.
        return (self.total_cpus - self.used_cpus) < 4 and (self.total_gpus - self.used_gpus) > 0

import simpy
from collections import deque  # 引入双端队列
from typing import Dict, Any, List, Optional
from .energy_model import HardwareEnergyTracker


class Node:
    def __init__(self, node_id: int, total_cpus: int, total_gpus: int):
        self.node_id = node_id
        self.total_cpus = total_cpus
        self.total_gpus = total_gpus
        self.used_cpus = 0
        self.used_gpus = 0
        self.active_tasks = 0
        self.io_blocked_tasks = 0

    def allocate(self, cpu_req: int, gpu_req: int):
        self.used_cpus += cpu_req
        self.used_gpus += gpu_req

    def release(self, cpu_req: int, gpu_req: int):
        self.used_cpus -= cpu_req
        self.used_gpus -= gpu_req

    @property
    def is_stranded(self) -> bool:
        return (self.total_cpus - self.used_cpus) < 4 and (self.total_gpus - self.used_gpus) > 0


class ClusterResourceManager:
    def __init__(self, env: simpy.Environment, num_nodes: int = 100,
                 cpus_per_node: int = 96, gpus_per_node: int = 8):
        self.env = env
        self.nodes = [Node(i, cpus_per_node, gpus_per_node) for i in range(num_nodes)]
        self.tracker = HardwareEnergyTracker()
        self.tracker.set_num_nodes(num_nodes)

        self.task_energies = {'LLM': [], 'Diffusion': [], 'DLRM': [], 'Training': []}
        self.task_counts = {'LLM': 0, 'Diffusion': 0, 'DLRM': 0, 'Training': 0}

        self.cluster_free_cpus = num_nodes * cpus_per_node
        self.cluster_free_gpus = num_nodes * gpus_per_node

        # 【修复 1】将普通列表改为高效的双端队列
        self.pending_queue = deque()
        self.trigger_schedule = self.env.event()
        self.env.process(self.scheduler_daemon())

    def scheduler_daemon(self):
        """Kube-Scheduler 级中央守护进程，结合 Backfill 限深扫描"""
        MAX_SCAN_DEPTH = 200  # 【修复 2】熔断上限：每次唤醒最多扫描前 200 个任务，防 O(N^2) 风暴

        while True:
            yield self.trigger_schedule
            self.trigger_schedule = self.env.event()

            # 全局 GPU 枯竭或队列为空，直接休眠
            if self.cluster_free_gpus == 0 or not self.pending_queue:
                continue

            unallocated = []
            scan_count = 0

            # 仅当有 GPU 且队列有任务且未达到扫描上限时执行
            while self.pending_queue and self.cluster_free_gpus > 0 and scan_count < MAX_SCAN_DEPTH:
                req = self.pending_queue.popleft()  # O(1) 弹出
                scan_count += 1
                task, allocated_event, req_c, req_g = req

                # 如果全局资源不足以满足该任务，暂时搁置
                if self.cluster_free_cpus < req_c or self.cluster_free_gpus < req_g:
                    unallocated.append(req)
                    continue

                temp_allocations = []
                rem_c, rem_g = req_c, req_g

                for node in self.nodes:
                    if rem_c <= 0 and rem_g <= 0: break
                    avail_c = node.total_cpus - node.used_cpus
                    avail_g = node.total_gpus - node.used_gpus

                    alloc_c = min(rem_c, avail_c)
                    alloc_g = min(rem_g, avail_g)

                    if alloc_c > 0 or alloc_g > 0:
                        temp_allocations.append((node, alloc_c, alloc_g))
                        rem_c -= alloc_c
                        rem_g -= alloc_g

                if rem_c == 0 and rem_g == 0:
                    for n, c, g in temp_allocations:
                        n.allocate(c, g)
                        self.cluster_free_cpus -= c
                        self.cluster_free_gpus -= g
                    allocated_event.succeed(temp_allocations)
                else:
                    unallocated.append(req)

            # 【修复 3】将未分配的任务原路塞回队首，维持优先级顺序 (O(k) 操作，极速)
            for req in reversed(unallocated):
                self.pending_queue.appendleft(req)

            # 如果队列还有剩余且由于 MAX_SCAN 退出，稍后再次自我唤醒以消化剩余任务
            if self.pending_queue and self.cluster_free_gpus > 0:
                if not self.trigger_schedule.triggered:
                    self.trigger_schedule.succeed()

    def process_task(self, task: Dict[str, Any]):
        import math
        raw_cpu = task.get('cpu_req', 0)
        raw_gpu = task.get('gpu_req', 0)

        if raw_cpu is None or (isinstance(raw_cpu, float) and math.isnan(raw_cpu)): raw_cpu = 0
        if raw_gpu is None or (isinstance(raw_gpu, float) and math.isnan(raw_gpu)): raw_gpu = 0

        max_c = sum(n.total_cpus for n in self.nodes)
        max_g = sum(n.total_gpus for n in self.nodes)

        req_cpus = max(0, min(int(raw_cpu), max_c))
        req_gpus = max(0, min(int(raw_gpu), max_g))

        if req_cpus == 0 and req_gpus == 0:
            io_time = task.get('io_time_ms', 0)
            compute_time = task.get('compute_time_ms', 0)
            if io_time > 0: yield self.env.timeout(io_time)
            if compute_time > 0: yield self.env.timeout(compute_time)
            t_type = task.get('type', 'LLM')
            if t_type in self.task_counts: self.task_counts[t_type] += 1
            return

        # 任务注册到中央调度器，然后挂起睡觉，绝不消耗任何 CPU 算力
        allocated_event = self.env.event()
        self.pending_queue.append((task, allocated_event, req_cpus, req_gpus))

        if not self.trigger_schedule.triggered:
            self.trigger_schedule.succeed()

        # 【核心】在这里乖乖睡觉，等中央调度器把节点塞到手里才醒来！
        nodes_needed = yield allocated_event

        # ----------- 任务执行与能耗统计阶段 -----------
        io_time = task.get('io_time_ms', 0)
        if io_time > 0:
            for n, _, _ in nodes_needed: n.io_blocked_tasks += 1
            yield self.env.timeout(io_time)
            for n, _, _ in nodes_needed: n.io_blocked_tasks -= 1

        compute_time = task.get('compute_time_ms', 0)
        if compute_time > 0:
            for n, _, _ in nodes_needed: n.active_tasks += 1
            yield self.env.timeout(compute_time)
            for n, _, _ in nodes_needed: n.active_tasks -= 1

            task_energy_j = 0.0
            for n, alloc_c, alloc_g in nodes_needed:
                node_fraction = max(alloc_g / n.total_gpus, alloc_c / n.total_cpus)
                eng = (alloc_g * self.tracker.p_dynamic_gpu + self.tracker.p_static * node_fraction) * (
                            compute_time / 1000.0)
                eng += (self.tracker.p_static * node_fraction) * (io_time / 1000.0)
                task_energy_j += eng

            t_type = task.get('type', 'LLM')
            if len(self.task_energies[t_type]) < 100000:
                self.task_energies[t_type].append(task_energy_j)
            else:
                import random
                idx = random.randint(0, self.task_counts[t_type])
                if idx < 100000:
                    self.task_energies[t_type][idx] = task_energy_j

        # ----------- 资源释放与唤醒调度器 -----------
        for n, alloc_c, alloc_g in nodes_needed:
            n.release(alloc_c, alloc_g)
            self.cluster_free_cpus += alloc_c
            self.cluster_free_gpus += alloc_g

        self.task_counts[task.get('type', 'LLM')] += 1

        # 资源释放完毕，踢一脚中央调度器去处理排队的兄弟
        if not self.trigger_schedule.triggered:
            self.trigger_schedule.succeed()

    def monitor_power_process(self):
        """
        [能耗监控探头]
        后台守护进程：周期性为集群耗电做状态快照，这是对标《Joule》顶刊图表的核心数据源！
        """
        while True:
            # 统计真正干活的 GPU 数量
            active_gpus = sum(n.used_gpus for n in self.nodes)

            # 统计因加载模型而陷入 I/O 阻塞的节点数（冷启动惩罚）
            io_blocked_nodes = sum(1 for n in self.nodes if n.io_blocked_tasks > 0)

            # 统计被“搁浅”的节点和 GPU 数量（硬件错配导致的隐性能耗）
            stranded_nodes = 0
            stranded_gpus = 0
            for n in self.nodes:
                if n.is_stranded:
                    stranded_nodes += 1
                    stranded_gpus += (n.total_gpus - n.used_gpus)

            # 将当前状态写入打点器
            self.tracker.record_power_state(self.env.now, active_gpus, io_blocked_nodes, stranded_nodes)

            # 每 10 秒（10000 毫秒）做一次切片采样。
            # 这里设置 10 秒既能保证宏观曲线平滑，又不会给 SimPy 引擎带来过高的事件开销。
            yield self.env.timeout(10000.0)



def process_events(env: simpy.Environment, cluster: ClusterResourceManager, event_stream):
    last_event_time = 0.0
    tasks_done = simpy.Store(env)
    total_dispatched = 0

    def task_wrapper(task):
        yield from cluster.process_task(task)
        tasks_done.put(1)

    print("  [Simulator] 开始读取日志并注入事件流，请稍候...")

    # 1. 注入事件阶段的进度监控
    for task in event_stream:
        wait_time = task['relative_time_ms'] - last_event_time
        if wait_time > 0:
            yield env.timeout(wait_time)
        last_event_time = task['relative_time_ms']

        # 派发任务到集群
        env.process(task_wrapper(task))
        total_dispatched += 1

        # 【新增】每处理 10万 条数据，打印一次当前进度与模拟时间
        if total_dispatched % 100000 == 0:
            sim_hours = env.now / (1000.0 * 60 * 60)
            print(f"  [Progress] 已注入 {total_dispatched} 个请求... 当前模拟器时间: 第 {sim_hours:.2f} 小时",
                  flush=True)

    print(f"  [Simulator] 所有 {total_dispatched} 个请求已全部注入！等待集群消化剩余任务...")

    # 2. 等待集群处理完毕阶段的进度监控
    completed = 0
    for _ in range(total_dispatched):
        yield tasks_done.get()
        completed += 1

        # 【新增】每完成 10万 个任务，打印一次消化进度
        if completed % 10000 == 0:
            print(f"  [Finishing] 集群已处理完毕 {completed} / {total_dispatched} 个任务...")

def run_simulation(env: simpy.Environment, event_stream, duration_ms: float = None) -> ClusterResourceManager:
    cluster = ClusterResourceManager(env)

    # Run the monitor process
    monitor_proc = env.process(cluster.monitor_power_process())

    # Run the event stream process
    env.process(process_events(env, cluster, event_stream))

    # Run the simulation
    if duration_ms:
        env.run(until=duration_ms)
    else:
        # Since monitor runs forever, we cannot just do env.run().
        # We need to run until process_events finishes.
        # So instead of returning early from process_events, we run it and wait
        pass # Handle running in main instead
    return cluster
