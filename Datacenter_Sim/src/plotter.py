import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as plt_sns
from .simulator import ClusterResourceManager
from typing import Dict

class Plotter:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Seaborn aesthetics
        plt_sns.set_theme(style="whitegrid")

    def plot_violin_energies(self, task_energies: Dict[str, list]):
        """
        图 1 (小提琴图): 对比 LLM, Diffusion, DLRM 的单次请求能耗分布
        """
        data = []
        for t_type, energies in task_energies.items():
            for e in energies:
                data.append({'Task Type': t_type, 'Energy (Joules)': e})

        df = pd.DataFrame(data)
        if df.empty:
            print("No task energies to plot.")
            return

        plt.figure(figsize=(10, 6))
        plt_sns.violinplot(x='Task Type', y='Energy (Joules)', data=df, inner="quartile")
        plt.title('Single Request Energy Distribution (Non-Token Tasks Tail Energy)')

        filepath = os.path.join(self.output_dir, 'fig1_violin_energy.pdf')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_stacked_power(self, tracker):
        """
        图 2 (堆叠面积图): 展示系统总功率
        """
        if not tracker.time_history:
            return

        df = pd.DataFrame({
            'Time (Hours)': tracker.time_history,
            'Active Compute Power (W)': tracker.power_active_history,
            'I/O Blocking Power (W)': tracker.power_io_waste_history,
            'GPU Stranded Power (W)': tracker.power_stranded_waste_history
        })

        plt.figure(figsize=(12, 6))
        plt.stackplot(df['Time (Hours)'],
                      df['Active Compute Power (W)'],
                      df['I/O Blocking Power (W)'],
                      df['GPU Stranded Power (W)'],
                      labels=['Active Compute', 'I/O Blocking Waste', 'GPU Stranded Waste'],
                      colors=['#2ecc71', '#f1c40f', '#e74c3c'])

        plt.xlabel('Simulation Time (Hours)')
        plt.ylabel('Power (Watts)')
        plt.title('Data Center Total Power Breakdown')
        plt.legend(loc='upper left')

        filepath = os.path.join(self.output_dir, 'fig2_stacked_power.pdf')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_carbon_footprint_comparison(self, ideal_energy: float, real_energy: float):
        """
        图 3 (柱状图): 对比理想集群与真实集群的总碳足迹(引入 Liu et al. 2025 碳排放因子)
        """
        # [Liu et al., 2025] - Carbon Footprint conversion
        CARBON_INTENSITY = 0.4 # kg CO2 / kWh

        # Convert Joules to kWh (1 kWh = 3.6e6 Joules)
        ideal_kwh = ideal_energy / 3.6e6
        real_kwh = real_energy / 3.6e6

        ideal_carbon = ideal_kwh * CARBON_INTENSITY
        real_carbon = real_kwh * CARBON_INTENSITY

        data = pd.DataFrame({
            'Cluster Scenario': ['Ideal Cluster', 'Real-World Cluster (Resource Mismatch)'],
            'Carbon Emission (kg CO2)': [ideal_carbon, real_carbon]
        })

        plt.figure(figsize=(8, 6))
        ax = plt_sns.barplot(x='Cluster Scenario', y='Carbon Emission (kg CO2)', hue='Cluster Scenario', data=data, palette='pastel', legend=False)

        # Add values on top of bars
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.2f'),
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha = 'center', va = 'center',
                        xytext = (0, 9),
                        textcoords = 'offset points')

        plt.title('Total Carbon Emission Comparison (Based on Liu et al. 2025)')

        filepath = os.path.join(self.output_dir, 'fig3_carbon_comparison.pdf')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
