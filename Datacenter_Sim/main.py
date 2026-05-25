import os
import argparse
import simpy
from src.data_loader import event_generator
from src.simulator import run_simulation
from src.plotter import Plotter

def main():
    parser = argparse.ArgumentParser(description="Data Center Energy Simulator")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing sample CSVs.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the resulting PDF plots.")
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plotter = Plotter(output_dir)

    from src.simulator import process_events, ClusterResourceManager

    print("Starting simulation for Ideal Cluster (LLM & Diffusion Only)...")
    env_ideal = simpy.Environment()
    cluster_ideal = ClusterResourceManager(env_ideal)
    env_ideal.process(cluster_ideal.monitor_power_process())
    stream_ideal = event_generator(data_dir, include_dlrm=False)

    # Start the event loop and wait for it
    proc_ideal = env_ideal.process(process_events(env_ideal, cluster_ideal, stream_ideal))
    env_ideal.run(until=proc_ideal)
    ideal_energy = cluster_ideal.tracker.calc_total_energy()
    print(f"Ideal Cluster Simulation Finished. Total Energy: {ideal_energy} Joules")

    print("\nStarting simulation for Real-World Cluster (LLM, Diffusion & DLRM)...")
    env_real = simpy.Environment()
    cluster_real = ClusterResourceManager(env_real)
    env_real.process(cluster_real.monitor_power_process())
    stream_real = event_generator(data_dir, include_dlrm=True)

    proc_real = env_real.process(process_events(env_real, cluster_real, stream_real))
    env_real.run(until=proc_real)
    real_energy = cluster_real.tracker.calc_total_energy()
    print(f"Real-World Cluster Simulation Finished. Total Energy: {real_energy} Joules")

    print("\nGenerating Plots...")
    # Plot violin using real cluster (since it contains DLRM as well)
    plotter.plot_violin_energies(cluster_real.task_energies)
    # Plot stacked power using real cluster
    plotter.plot_stacked_power(cluster_real.tracker)
    # Plot carbon comparison between ideal and real
    plotter.plot_carbon_footprint_comparison(ideal_energy, real_energy)
    print(f"Plots saved to {output_dir}/")

if __name__ == "__main__":
    main()
