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
from typing import Dict, Any, List, Optional
from .energy_model import HardwareEnergyTracker


# (保留原来的 Node 类不动...)
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

        # 【核心护盾1】条件变量锁：取代死循环 Polling，防止 CPU 卡死
        self.resource_freed_event = self.env.event()

    def allocate(self, cpu_req: int, gpu_req: int) -> Optional[Node]:
        for node in self.nodes:
            if node.can_allocate(cpu_req, gpu_req):
                node.allocate(cpu_req, gpu_req)
                return node
        return None

    def release(self, node: Node, cpu_req: int, gpu_req: int):
        node.release(cpu_req, gpu_req)
        # 只要有任何资源释放，立刻触发事件，全局广播唤醒所有处于饥饿排队中的任务
        if not self.resource_freed_event.triggered:
            self.resource_freed_event.succeed()
            self.resource_freed_event = self.env.event()

    def process_task(self, task: Dict[str, Any]):
        nodes_needed = []

        # 【安全截断】防止单个异常长尾任务（如需求 1000 卡）超过集群总和导致全盘死锁
        max_cluster_cpus = sum(n.total_cpus for n in self.nodes)
        max_cluster_gpus = sum(n.total_gpus for n in self.nodes)
        req_cpus = min(task['cpu_req'], max_cluster_cpus)
        req_gpus = min(task['gpu_req'], max_cluster_gpus)

        # 【核心护盾2】原子级事务分配（All-or-Nothing）彻底封杀分布式死锁
        while True:
            remaining_cpus = req_cpus
            remaining_gpus = req_gpus
            temp_allocations = []

            # 模拟 Gang Scheduling：遍历集群搜集一切可用碎片
            for node in self.nodes:
                if remaining_cpus <= 0 and remaining_gpus <= 0:
                    break

                avail_c = node.total_cpus - node.used_cpus
                avail_g = node.total_gpus - node.used_gpus

                alloc_c = min(remaining_cpus, avail_c)
                alloc_g = min(remaining_gpus, avail_g)

                if alloc_c > 0 or alloc_g > 0:
                    temp_allocations.append((node, alloc_c, alloc_g))
                    remaining_cpus -= alloc_c
                    remaining_gpus -= alloc_g

            # 检查是否满足了所有需求
            if remaining_cpus == 0 and remaining_gpus == 0:
                # 彻底凑齐！执行真实分配（提交事务）
                for n, c, g in temp_allocations:
                    n.allocate(c, g)
                    nodes_needed.append((n, c, g))
                break  # 跳出资源获取循环，进入计算阶段
            else:
                # 【破局点】如果没凑齐，决不能持有已预占的资源死等！
                # 放弃所有临时预占，通过 Yield 将自身彻底挂起（0 CPU消耗），直到集群有人释放资源才被唤醒重试
                yield self.resource_freed_event

        # 1. I/O Block Cold Start Penalty
        io_time = task['io_time_ms']
        if io_time > 0:
            for n, _, _ in nodes_needed:
                n.io_blocked_tasks += 1
            yield self.env.timeout(io_time)
            for n, _, _ in nodes_needed:
                n.io_blocked_tasks -= 1

        # 2. Active Compute
        compute_time = task['compute_time_ms']
        if compute_time > 0:
            for n, _, _ in nodes_needed:
                n.active_tasks += 1
            yield self.env.timeout(compute_time)
            for n, _, _ in nodes_needed:
                n.active_tasks -= 1

            # 能耗计算与统筹
            task_energy_j = 0.0
            for n, alloc_c, alloc_g in nodes_needed:
                node_fraction = max(alloc_g / n.total_gpus, alloc_c / n.total_cpus)
                eng = (alloc_g * self.tracker.p_dynamic_gpu + self.tracker.p_static * node_fraction) * (
                            compute_time / 1000.0)
                eng += (self.tracker.p_static * node_fraction) * (io_time / 1000.0)
                task_energy_j += eng

            # 蓄水池采样防绘图时 OOM 内存爆炸
            if len(self.task_energies[task['type']]) < 100000:
                self.task_energies[task['type']].append(task_energy_j)
            else:
                import random
                idx = random.randint(0, self.task_counts[task['type']])
                if idx < 100000:
                    self.task_energies[task['type']][idx] = task_energy_j

        # 3. 释放资源并触发全局广播唤醒（唤醒其他处于 yield self.resource_freed_event 的任务）
        for n, alloc_c, alloc_g in nodes_needed:
            self.release(n, alloc_c, alloc_g)
        self.task_counts[task['type']] += 1

    def monitor_power_process(self):
        # 周期性为集群耗电做状态快照
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
            # 【核心护盾3】从每秒 1 次（1000.0）放宽到每 10 秒次（10000.0）快照
            # 大幅减少数百万次无意义的协程切换，提升引擎长尾推进速度
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
