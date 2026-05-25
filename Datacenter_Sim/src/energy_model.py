class HardwareEnergyTracker:
    def __init__(self, p_static: float = 350.0, p_dynamic_gpu: float = 1000.0):
        """
        p_static: Static power of a node in Watts (W)
        p_dynamic_gpu: Dynamic power of a fully loaded GPU in Watts (W)
        """
        self.p_static = p_static
        self.p_dynamic_gpu = p_dynamic_gpu

        # Tracker states
        self.total_energy_joules = 0.0
        self.last_update_time_ms = 0.0
        self.num_nodes = 0

        # Power over time arrays
        self.time_history = []
        self.power_active_history = []
        self.power_io_waste_history = []
        self.power_stranded_waste_history = []
        self.power_idle_history = []

    def set_num_nodes(self, num_nodes: int):
        self.num_nodes = num_nodes

    def record_power_state(self, current_time_ms: float, active_gpus: int, io_blocked_nodes: int, stranded_nodes: int):
        """
        Calculates exact physics-based power.
        Updates total energy based on time elapsed since last record.
        Records instantaneous power profile for visualization.
        """
        time_elapsed_s = (current_time_ms - self.last_update_time_ms) / 1000.0
        self.last_update_time_ms = current_time_ms

        # P_static is ALWAYS consumed by all 100 nodes, regardless of what they do.
        # But for stacked area chart, we split it up to show where it's "wasted" vs "active".
        # 1. Total static power is num_nodes * p_static.
        # 2. Dynamic power is active_gpus * p_dynamic_gpu.

        # For visualization breakdown, we will apportion the static power:
        # - stranded nodes consume their p_static purely as stranded waste
        # - io blocked nodes consume their p_static as IO waste
        # - active nodes (nodes with active GPUs) consume p_static as active power + their dynamic power
        # - remaining nodes consume idle p_static.

        power_dynamic = active_gpus * self.p_dynamic_gpu

        # Approximate node count for each state (cap at total num_nodes)
        nodes_stranded = min(stranded_nodes, self.num_nodes)
        nodes_io = min(io_blocked_nodes, self.num_nodes - nodes_stranded)

        # Active nodes estimate: at least active_gpus / 8 (assuming 8 gpus per node)
        nodes_active = min(max(1, int(active_gpus / 8) + 1) if active_gpus > 0 else 0,
                           self.num_nodes - nodes_stranded - nodes_io)

        nodes_idle = self.num_nodes - nodes_stranded - nodes_io - nodes_active

        p_stranded = nodes_stranded * self.p_static
        p_io = nodes_io * self.p_static
        p_active_static = nodes_active * self.p_static
        p_idle = nodes_idle * self.p_static

        total_power = power_dynamic + (self.num_nodes * self.p_static)

        # Update energy
        self.total_energy_joules += total_power * time_elapsed_s

        # Append for plotting
        self.time_history.append(current_time_ms / (1000.0 * 60 * 60))
        self.power_active_history.append(power_dynamic + p_active_static + p_idle) # Group idle with active baseline
        self.power_io_waste_history.append(p_io)
        self.power_stranded_waste_history.append(p_stranded)

    def calc_total_energy(self) -> float:
        """
        Return the total integrated energy over time.
        """
        return self.total_energy_joules
