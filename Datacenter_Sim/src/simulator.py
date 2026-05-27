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

class ClusterResourceManager:
    def __init__(self, env: simpy.Environment, num_nodes: int = 100,
                 cpus_per_node: int = 96, gpus_per_node: int = 8):
        self.env = env
        self.nodes = [Node(i, cpus_per_node, gpus_per_node) for i in range(num_nodes)]
        self.tracker = HardwareEnergyTracker()
        self.tracker.set_num_nodes(num_nodes)

        # Track individual task energy per type for plotting
        self.task_energies = {
            'LLM': [],
            'Diffusion': [],
            'DLRM': [],
            'Training': []
        }
        self.task_counts = {
            'LLM': 0,
            'Diffusion': 0,
            'DLRM': 0,
            'Training': 0
        }

    def allocate(self, cpu_req: int, gpu_req: int) -> Optional[Node]:
        # First-Fit greedy scheduling
        for node in self.nodes:
            if node.can_allocate(cpu_req, gpu_req):
                node.allocate(cpu_req, gpu_req)
                return node
        return None

    def release(self, node: Node, cpu_req: int, gpu_req: int):
        node.release(cpu_req, gpu_req)

    def process_task(self, task: Dict[str, Any]):
        # Wait until resources are available
        # In a real cluster, tasks queue up. We will retry periodically if no resources.
        node = None
        while node is None:
            node = self.allocate(task['cpu_req'], task['gpu_req'])
            if node is None:
                # Wait a bit and retry
                yield self.env.timeout(100.0) # 100ms

        # 1. I/O Block Cold Start Penalty
        io_time = task['io_time_ms']
        if io_time > 0:
            node.io_blocked_tasks += 1
            yield self.env.timeout(io_time)
            node.io_blocked_tasks -= 1

        # 2. Active Compute
        compute_time = task['compute_time_ms']
        if compute_time > 0:
            node.active_tasks += 1
            yield self.env.timeout(compute_time)
            node.active_tasks -= 1

            # [León-Vega et al., 2024] - Energy slicing rule: Assign fraction of static power
            node_fraction = max(task['gpu_req'] / node.total_gpus, task['cpu_req'] / node.total_cpus)

            # Record single task total energy for violin plot
            # (Energy = P * T). Active + I/O wasted energy for this task. Time is ms so divide by 1000.0
            task_energy_j = (task['gpu_req'] * self.tracker.p_dynamic_gpu + self.tracker.p_static * node_fraction) * (compute_time / 1000.0)
            task_energy_j += (self.tracker.p_static * node_fraction) * (io_time / 1000.0)

            # Reservoir sample task energies to prevent OOM
            if len(self.task_energies[task['type']]) < 100000:
                self.task_energies[task['type']].append(task_energy_j)
            else:
                import random
                idx = random.randint(0, self.task_counts[task['type']])
                if idx < 100000:
                    self.task_energies[task['type']][idx] = task_energy_j

        # Release resources
        self.release(node, task['cpu_req'], task['gpu_req'])
        self.task_counts[task['type']] += 1

    def monitor_power_process(self):
        # Periodically snapshot the cluster state to log power profile over time
        while True:
            active_gpus = sum(n.used_gpus for n in self.nodes) # approximation of active
            io_blocked_nodes = sum(1 for n in self.nodes if n.io_blocked_tasks > 0)

            # Calculate stranded GPUs and add energy
            stranded_nodes = 0
            stranded_gpus = 0
            for n in self.nodes:
                if n.is_stranded:
                    stranded_nodes += 1
                    stranded_gpus += (n.total_gpus - n.used_gpus)

            self.tracker.record_power_state(self.env.now, active_gpus, io_blocked_nodes, stranded_nodes)
            yield self.env.timeout(1000.0) # snapshot every 1 second

def process_events(env: simpy.Environment, cluster: ClusterResourceManager, event_stream):
    last_event_time = 0.0
    tasks_done = simpy.Store(env)
    total_dispatched = 0

    def task_wrapper(task):
        yield from cluster.process_task(task)
        tasks_done.put(1)

    for task in event_stream:
        wait_time = task['relative_time_ms'] - last_event_time
        if wait_time > 0:
            yield env.timeout(wait_time)
        last_event_time = task['relative_time_ms']

        # Dispatch task to cluster
        env.process(task_wrapper(task))
        total_dispatched += 1

    # Wait for all dispatched tasks to finish
    for _ in range(total_dispatched):
        yield tasks_done.get()

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
