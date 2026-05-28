import simpy
from collections import deque
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

        self.pending_queue = deque()

        # -------------------------------------------------------
        # 【修复核心】使用 BoundedSemaphore 风格的 "脏标志 + 事件" 机制
        # 替代原来容易丢信号的单次 Event。
        # _schedule_dirty = True 表示"有工作要做"，调度器在空转时会立刻自我唤醒。
        # -------------------------------------------------------
        self._schedule_dirty = False
        self._schedule_event = self.env.event()

        self.env.process(self.scheduler_daemon())

    # ------------------------------------------------------------------
    # 内部工具：安全触发调度器
    # 无论被调用多少次，都不会因为 event 已经 triggered 而丢信号。
    # ------------------------------------------------------------------
    def _kick_scheduler(self):
        self._schedule_dirty = True
        if not self._schedule_event.triggered:
            self._schedule_event.succeed()

    def scheduler_daemon(self):
        """
        Kube-Scheduler 级中央守护进程，结合 Backfill 限深扫描。
        【关键修复】改用"脏标志"轮询方式，彻底消灭调度信号丢失导致的死锁。
        """
        MAX_SCAN_DEPTH = 200  # 每次唤醒最多扫描前 200 个任务，防 O(N^2) 风暴

        while True:
            # 等待唤醒信号
            yield self._schedule_event
            # 立即重置事件，为下次唤醒做准备（在处理完之前随时可能再次被踢）
            self._schedule_event = self.env.event()
            self._schedule_dirty = False

            # 反复处理，直到本轮没有进展为止
            while self.pending_queue and self.cluster_free_gpus > 0:
                made_progress = False
                unallocated = []
                scan_count = 0

                while self.pending_queue and self.cluster_free_gpus > 0 and scan_count < MAX_SCAN_DEPTH:
                    req = self.pending_queue.popleft()
                    scan_count += 1
                    task, allocated_event, req_c, req_g = req

                    # 全局资源不足，暂时搁置
                    if self.cluster_free_cpus < req_c or self.cluster_free_gpus < req_g:
                        unallocated.append(req)
                        continue

                    # 尝试跨节点分配
                    temp_allocations = []
                    rem_c, rem_g = req_c, req_g

                    for node in self.nodes:
                        if rem_c <= 0 and rem_g <= 0:
                            break
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
                        made_progress = True
                    else:
                        unallocated.append(req)

                # 将未分配的任务原路塞回队首（O(k)，极速）
                for req in reversed(unallocated):
                    self.pending_queue.appendleft(req)

                # 本轮没有任何任务被分配出去（资源不足），退出内循环等下一次踢醒
                if not made_progress:
                    break

                # 达到 MAX_SCAN_DEPTH 但队列还有任务，立即再踢自己一下继续消化
                if self.pending_queue and self.cluster_free_gpus > 0:
                    # 直接继续 while 循环，不需要额外事件
                    pass

            # 如果退出循环时脏标志又被置位了（资源释放发生在处理过程中），立即重新唤醒
            if self._schedule_dirty and not self._schedule_event.triggered:
                self._schedule_event.succeed()

    def process_task(self, task: Dict[str, Any]):
        import math
        raw_cpu = task.get('cpu_req', 0)
        raw_gpu = task.get('gpu_req', 0)

        if raw_cpu is None or (isinstance(raw_cpu, float) and math.isnan(raw_cpu)):
            raw_cpu = 0
        if raw_gpu is None or (isinstance(raw_gpu, float) and math.isnan(raw_gpu)):
            raw_gpu = 0

        max_c = sum(n.total_cpus for n in self.nodes)
        max_g = sum(n.total_gpus for n in self.nodes)

        req_cpus = max(0, min(int(raw_cpu), max_c))
        req_gpus = max(0, min(int(raw_gpu), max_g))

        # 无资源需求的任务直接执行，不进入调度队列
        if req_cpus == 0 and req_gpus == 0:
            io_time = task.get('io_time_ms', 0) or 0
            compute_time = task.get('compute_time_ms', 0) or 0
            if io_time > 0:
                yield self.env.timeout(io_time)
            if compute_time > 0:
                yield self.env.timeout(compute_time)
            t_type = task.get('type', 'LLM')
            if t_type in self.task_counts:
                self.task_counts[t_type] += 1
            return

        # 注册到中央调度器队列，然后挂起等待分配
        allocated_event = self.env.event()
        self.pending_queue.append((task, allocated_event, req_cpus, req_gpus))
        self._kick_scheduler()

        # 在这里乖乖睡觉，等中央调度器把节点塞到手里才醒来
        nodes_needed = yield allocated_event

        # ----------- 任务执行与能耗统计阶段 -----------
        io_time = task.get('io_time_ms', 0) or 0
        if io_time > 0:
            for n, _, _ in nodes_needed:
                n.io_blocked_tasks += 1
            yield self.env.timeout(io_time)
            for n, _, _ in nodes_needed:
                n.io_blocked_tasks -= 1

        compute_time = task.get('compute_time_ms', 0) or 0
        if compute_time > 0:
            for n, _, _ in nodes_needed:
                n.active_tasks += 1
            yield self.env.timeout(compute_time)
            for n, _, _ in nodes_needed:
                n.active_tasks -= 1

            task_energy_j = 0.0
            for n, alloc_c, alloc_g in nodes_needed:
                node_fraction = max(
                    alloc_g / n.total_gpus if n.total_gpus > 0 else 0,
                    alloc_c / n.total_cpus if n.total_cpus > 0 else 0
                )
                eng = (alloc_g * self.tracker.p_dynamic_gpu + self.tracker.p_static * node_fraction) * (
                        compute_time / 1000.0)
                eng += (self.tracker.p_static * node_fraction) * (io_time / 1000.0)
                task_energy_j += eng

            t_type = task.get('type', 'LLM')
            if t_type not in self.task_energies:
                self.task_energies[t_type] = []
            if len(self.task_energies[t_type]) < 100000:
                self.task_energies[t_type].append(task_energy_j)
            else:
                import random
                idx = random.randint(0, self.task_counts.get(t_type, 100000))
                if idx < 100000:
                    self.task_energies[t_type][idx] = task_energy_j

        # ----------- 资源释放与唤醒调度器 -----------
        for n, alloc_c, alloc_g in nodes_needed:
            n.release(alloc_c, alloc_g)
            self.cluster_free_cpus += alloc_c
            self.cluster_free_gpus += alloc_g

        self.task_counts[task.get('type', 'LLM')] = self.task_counts.get(task.get('type', 'LLM'), 0) + 1

        # 资源释放完毕，踢一脚中央调度器去处理排队的兄弟
        self._kick_scheduler()

    def monitor_power_process(self):
        """
        后台守护进程：周期性为集群耗电做状态快照。
        """
        while True:
            active_gpus = sum(n.used_gpus for n in self.nodes)
            io_blocked_nodes = sum(1 for n in self.nodes if n.io_blocked_tasks > 0)

            stranded_nodes = 0
            stranded_gpus = 0
            for n in self.nodes:
                if n.is_stranded:
                    stranded_nodes += 1
                    stranded_gpus += (n.total_gpus - n.used_gpus)

            self.tracker.record_power_state(self.env.now, active_gpus, io_blocked_nodes, stranded_nodes)

            yield self.env.timeout(10000.0)


def process_events(env: simpy.Environment, cluster: ClusterResourceManager, event_stream):
    last_event_time = 0.0
    tasks_done = simpy.Store(env)
    total_dispatched = 0

    def task_wrapper(task):
        yield from cluster.process_task(task)
        tasks_done.put(1)

    print("  [Simulator] 开始读取日志并注入事件流，请稍候...")

    # 1. 注入事件阶段
    for task in event_stream:
        wait_time = task['relative_time_ms'] - last_event_time
        if wait_time > 0:
            yield env.timeout(wait_time)
        last_event_time = task['relative_time_ms']

        env.process(task_wrapper(task))
        total_dispatched += 1

        if total_dispatched % 100000 == 0:
            sim_hours = env.now / (1000.0 * 60 * 60)
            print(f"  [Progress] 已注入 {total_dispatched} 个请求... 当前模拟器时间: 第 {sim_hours:.2f} 小时",
                  flush=True)

    print(f"  [Simulator] 所有 {total_dispatched} 个请求已全部注入！等待集群消化剩余任务...", flush=True)

    # 2. 等待集群处理完毕
    completed = 0
    report_interval = max(1, total_dispatched // 20)  # 每完成 5% 打印一次
    for _ in range(total_dispatched):
        yield tasks_done.get()
        completed += 1

        if completed % report_interval == 0 or completed == total_dispatched:
            pct = completed / total_dispatched * 100
            print(f"  [Finishing] 集群已处理 {completed:,} / {total_dispatched:,} ({pct:.1f}%) 个任务...",
                  flush=True)


def run_simulation(env: simpy.Environment, event_stream, duration_ms: float = None) -> ClusterResourceManager:
    cluster = ClusterResourceManager(env)
    env.process(cluster.monitor_power_process())
    env.process(process_events(env, cluster, event_stream))

    if duration_ms:
        env.run(until=duration_ms)
    return cluster
