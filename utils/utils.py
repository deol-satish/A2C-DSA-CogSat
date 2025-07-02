import numpy as np
import os
import matplotlib.pyplot as plt

def compare_sinr_per_element(saved_folder, num_episodes=1016):
    # Load baseline data
    SINR_baseline = np.load(f'{saved_folder}/Baseline_SINR.npy')  # (20, 61)
    total_values = SINR_baseline.size  # 20 * 61 = 1220

    # Initialize arrays to store statistics for each episode
    episode_numbers = []
    better_counts = []
    better_percentages = []
    avg_improvements = []

    for episode_idx in range(1, num_episodes + 1):
        sinr_path = f'{saved_folder}/Episode_{episode_idx}_SINR.npy'

        if os.path.exists(sinr_path):
            sinr = np.load(sinr_path)  # (20, 61)

            better_mask = sinr > SINR_baseline
            better_count = np.sum(better_mask)
            percentage_better = (better_count / total_values) * 100

            # Average improvement
            if better_count > 0:
                avg_improvement = np.mean((sinr - SINR_baseline))
            else:
                avg_improvement = 0

            # Store statistics
            episode_numbers.append(episode_idx)
            better_counts.append(better_count)
            better_percentages.append(percentage_better)
            avg_improvements.append(avg_improvement)

            # Optional print
            print(f"Episode {episode_idx}:")
            print(f"  Better than baseline: {better_count}/{total_values} ({percentage_better:.2f}%)")
            print(f"  Avg improvement: {avg_improvement:.4f} dB")
            print("----------------------------------")

    # Convert to numpy arrays
    episode_numbers = np.array(episode_numbers)
    better_counts = np.array(better_counts)
    better_percentages = np.array(better_percentages)
    avg_improvements = np.array(avg_improvements)

    # Find best episode
    best_idx = np.argmax(better_counts)
    max_count = better_counts[best_idx]
    candidates = np.where(better_counts == max_count)[0]
    best_candidate = candidates[np.argmax(avg_improvements[candidates])]

    print("\n\n✅ FINAL RESULTS:")
    print(f"Best Episode: {episode_numbers[best_candidate]}")
    print(f"➡️  Values better than baseline: {better_counts[best_candidate]}/{total_values} ({better_percentages[best_candidate]:.2f}%)")
    print(f"📈 Avg improvement: {avg_improvements[best_candidate]:.4f} dB")

    # Plot results
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 1, 1)
    plt.plot(episode_numbers, better_percentages, 'b-', label='Percentage better')
    plt.scatter(episode_numbers[best_candidate], better_percentages[best_candidate], color='red', label='Best Episode')
    plt.xlabel('Episode Number')
    plt.ylabel('Percentage Better than Baseline (%)')
    plt.title('Percentage of SINR Values Better Than Baseline')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(episode_numbers, avg_improvements, 'g-', label='Average improvement')
    plt.scatter(episode_numbers[best_candidate], avg_improvements[best_candidate], color='red', label='Best Episode')
    plt.xlabel('Episode Number')
    plt.ylabel('Average SINR Improvement (dB)')
    plt.title('Average SINR Improvement Where Better Than Baseline')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()
    return episode_numbers[best_candidate]


def compare_sinr_user_mean(saved_folder, num_episodes=1016):
    # Load baseline SINR data
    SINR_baseline = np.load(f'{saved_folder}/Baseline_SINR.npy')  # (20, 61)
    baseline_mean = np.mean(SINR_baseline, axis=1)  # Mean across timesteps per user

    episode_numbers = []
    mean_comparisons = []

    for episode_idx in range(1, num_episodes + 1):
        sinr_path = f'{saved_folder}/Episode_{episode_idx}_SINR.npy'

        if os.path.exists(sinr_path):
            sinr = np.load(sinr_path)  # (20, 61)
            episode_mean = np.mean(sinr, axis=1)

            better_mask = episode_mean > baseline_mean
            worse_mask = baseline_mean > episode_mean
            equal_mask = np.isclose(episode_mean, baseline_mean)

            better_count = np.sum(better_mask)
            worse_count = np.sum(worse_mask)
            equal_count = np.sum(equal_mask)

            mean_diff = episode_mean - baseline_mean
            avg_improvement = np.mean(mean_diff[better_mask]) if better_count > 0 else 0
            avg_degradation = np.mean(-mean_diff[worse_mask]) if worse_count > 0 else 0

            episode_numbers.append(episode_idx)
            mean_comparisons.append({
                'better_users': np.where(better_mask)[0],
                'worse_users': np.where(worse_mask)[0],
                'equal_users': np.where(equal_mask)[0],
                'better_count': better_count,
                'worse_count': worse_count,
                'equal_count': equal_count,
                'mean_diff': mean_diff,
                'avg_improvement': avg_improvement,
                'avg_degradation': avg_degradation,
                'overall_mean_diff': np.mean(mean_diff)
            })

            print(f"\nEpisode {episode_idx} Mean Comparison:")
            print(f"Users with higher mean: {better_count}/20")
            print(f"Users with lower mean: {worse_count}/20")
            print(f"Users with equal mean: {equal_count}/20")
            print(f"Overall mean difference: {np.mean(mean_diff):.4f} dB")

            if better_count > 0:
                print(f"Avg improvement for better users: {avg_improvement:.4f} dB")
                print("Better performing users:", np.where(better_mask)[0])

            if worse_count > 0:
                print(f"Avg degradation for worse users: {avg_degradation:.4f} dB")
                print("Worse performing users:", np.where(worse_mask)[0])

    episode_numbers = np.array(episode_numbers)
    overall_diffs = [comp['overall_mean_diff'] for comp in mean_comparisons]
    best_episode_idx = np.argmax(overall_diffs)
    best_episode = episode_numbers[best_episode_idx]

    print("\n\n✅ FINAL RESULTS - MEAN COMPARISON:")
    print(f"Best Episode: {best_episode}")
    print(f"best_episode_idx: {best_episode_idx}")
    print(f"Highest overall mean improvement: {overall_diffs[best_episode_idx]:.4f} dB")
    print(f"Users improved: {mean_comparisons[best_episode_idx]['better_count']}/20")
    print(f"Users degraded: {mean_comparisons[best_episode_idx]['worse_count']}/20")

    # Plotting
    plt.figure(figsize=(15, 8))

    plt.subplot(2, 1, 1)
    plt.plot(episode_numbers, overall_diffs, 'b-', label='Overall mean difference')
    plt.scatter(best_episode, overall_diffs[best_episode_idx], color='red', label='Best Episode')
    plt.xlabel('Episode Number')
    plt.ylabel('Mean Difference (dB)')
    plt.title('Overall Mean SINR Difference (Episode - Baseline)')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    better_counts = [comp['better_count'] for comp in mean_comparisons]
    worse_counts = [comp['worse_count'] for comp in mean_comparisons]
    plt.plot(episode_numbers, better_counts, 'g-', label='Improved users')
    plt.plot(episode_numbers, worse_counts, 'r-', label='Degraded users')
    plt.scatter(best_episode, better_counts[best_episode_idx], color='green')
    plt.scatter(best_episode, worse_counts[best_episode_idx], color='red')
    plt.xlabel('Episode Number')
    plt.ylabel('Number of Users')
    plt.title('Number of Users with Improved/Degraded Mean SINR')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()
    return best_episode



import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

def analyze_and_plot_best_episode(saved_folder, best_episode_index, graphs_folder, data_csv_folder):
    def save_data_to_csv(data_dict, filename):
        df = pd.DataFrame(data_dict)
        df.to_csv(os.path.join(data_csv_folder, filename), index=False)
        print(f"✅ CSV saved: {filename}")

    def plot_and_save_metric_vs_user(users_data, baseline_data, best_data, title, ylabel, filename_prefix):
        plt.figure(figsize=(10, 6))
        plt.plot(users_data, baseline_data, label='Random Allocation', color='r', linewidth=3, marker='o')
        plt.plot(users_data, best_data, label='Advanced DSA (A2C)', color='b', linewidth=3, marker='x')
        plt.xlabel('User Index')
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        for ext in ['png', 'eps', 'pdf']:
            plt.savefig(os.path.join(graphs_folder, f'{filename_prefix}.{ext}'), dpi=500, format=ext)
        plt.show()
        plt.close()

        data = {
            'User Index': users_data,
            'Random Allocation': baseline_data,
            'Advanced DSA (A2C)': best_data
        }
        save_data_to_csv(data, f'{filename_prefix}_data.csv')

    def plot_and_save_metric_vs_time(best_data, baseline_data, user_type, title, ylabel, filename_prefix):
        time_steps = np.arange(len(best_data)) * 30  # Assuming 30-second intervals
        plt.figure(figsize=(10, 6))
        plt.plot(time_steps, baseline_data, label='Random Allocation', color='red')
        plt.plot(time_steps, best_data, label='Advanced DSA (A2C)', color='blue')
        plt.title(f'{user_type} - {title} vs Time')
        plt.xlabel('Time (s)')
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        for ext in ['pdf', 'eps', 'png']:
            plt.savefig(os.path.join(graphs_folder, f'{filename_prefix}.{ext}'), dpi=500, format=ext)
        plt.show()
        plt.close()

        data = {
            'Time (s)': time_steps,
            'Random Allocation': baseline_data,
            'Advanced DSA (A2C)': best_data
        }
        save_data_to_csv(data, f'{filename_prefix}_data.csv')

    def plot_and_save_metric_vs_time_ma(best_data, baseline_data, user_type, title, ylabel, filename_prefix, window_size=5):
        ma_best = np.convolve(best_data, np.ones(window_size)/window_size, mode='valid')
        ma_base = np.convolve(baseline_data, np.ones(window_size)/window_size, mode='valid')
        ma_time = np.arange(len(ma_best)) * 30

        plt.figure(figsize=(10, 6))
        plt.plot(ma_time, ma_base, label='Random Allocation', color='red')
        plt.plot(ma_time, ma_best, label='Advanced DSA (A2C)', color='blue')
        plt.title(f'{user_type} - {title} (Moving Avg)')
        plt.xlabel('Time (s)')
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        for ext in ['pdf', 'eps', 'png']:
            plt.savefig(os.path.join(graphs_folder, f'{filename_prefix}.{ext}'), dpi=500, format=ext)
        plt.show()
        plt.close()

        data = {
            'Time (s)': ma_time,
            'Random Allocation': ma_base,
            'Advanced DSA (A2C)': ma_best
        }
        save_data_to_csv(data, f'{filename_prefix}_data.csv')

    # Load data
    Thrpt_baseline = np.load(f'{saved_folder}/Baseline_Thrpt.npy')     # (20, 61)
    SINR_baseline = np.load(f'{saved_folder}/Baseline_SINR.npy')       # (20, 61)
    Intf_baseline = np.load(f'{saved_folder}/Baseline_Intf.npy')       # (20, 61)
    SE_baseline = np.load(f'{saved_folder}/Baseline_SE.npy')       # (20, 61)

    Thrpt_best = np.load(f'{saved_folder}/Episode_{best_episode_index}_Thrpt.npy')
    SINR_best = np.load(f'{saved_folder}/Episode_{best_episode_index}_SINR.npy')
    Intf_best = np.load(f'{saved_folder}/Episode_{best_episode_index}_Intf.npy')
    SE_best = np.load(f'{saved_folder}/Episode_{best_episode_index}_SE.npy')  # Newly added

    mean_thrpt_baseline = np.mean(Thrpt_baseline, axis=1)
    mean_thrpt_best = np.mean(Thrpt_best, axis=1)
    mean_sinr_baseline = np.mean(SINR_baseline, axis=1)
    mean_sinr_best = np.mean(SINR_best, axis=1)
    mean_intf_baseline = np.mean(Intf_baseline, axis=1)
    mean_intf_best = np.mean(Intf_best, axis=1)


    SE_baseline = SE_baseline.reshape(121,)
    SE_best = SE_best.reshape(121,)

    user_list = np.arange(1, 21)

    # --- Plot vs User Index ---
    print("\n--- Generating Metric vs User Index Plots ---")
    plot_and_save_metric_vs_user(user_list, mean_thrpt_baseline, mean_thrpt_best, 
                                 'Mean Throughput per User', 'Throughput (Mbps)', 'a2c_throughput_plot')
    plot_and_save_metric_vs_user(user_list, mean_sinr_baseline, mean_sinr_best, 
                                 'Mean SINR per User', 'SINR (dB)', 'a2c_sinr_plot')
    plot_and_save_metric_vs_user(user_list, mean_intf_baseline, mean_intf_best, 
                                 'Mean Interference per User', 'Interference (dB)', 'a2c_interference_plot')
    

    # --- Plot vs Time for LEO and GEO ---
    print("\n--- Generating Metric vs Time Plots ---")
    LEO = range(0, 10)
    GEO = range(10, 20)

    metrics = {
        'Throughput': ('Throughput (Mbps)', Thrpt_best, Thrpt_baseline),
        'SINR': ('SINR (dB)', SINR_best, SINR_baseline),
        'Interference': ('Interference (dB)', Intf_best, Intf_baseline),
    }



    for user_type, indices in zip(['LEO', 'GEO'], [LEO, GEO]):
        for metric_name, (ylabel, best_data, base_data) in metrics.items():
            best = best_data[indices].mean(axis=0)
            base = base_data[indices].mean(axis=0)

            prefix = f"{user_type}_{metric_name.replace(' ', '')}"
            prefix_ma = f"{prefix}_MA"

            plot_and_save_metric_vs_time(best, base, user_type, metric_name, ylabel, prefix)
            plot_and_save_metric_vs_time_ma(best, base, user_type, metric_name, ylabel, prefix_ma)

    metrics = {
        'SE': ('Spectral Efficiency (bits/s/Hz)', SE_best, SE_baseline)
    }


    for user_type, indices in zip(['LEO', 'GEO'], [LEO, GEO]):
        for metric_name, (ylabel, best_data, base_data) in metrics.items():
            best = best_data[indices]
            base = base_data[indices]

            prefix = f"{user_type}_{metric_name.replace(' ', '')}"
            prefix_ma = f"{prefix}_MA"

            plot_and_save_metric_vs_time(best, base, user_type, metric_name, ylabel, prefix)
            plot_and_save_metric_vs_time_ma(best, base, user_type, metric_name, ylabel, prefix_ma)

    print("\n✅ Script execution complete. All graphs and data saved.")


    metrics = {
        'SE': ('Spectral Efficiency (bits/s/Hz)', SE_best, SE_baseline)
    }


 

    prefix = f"all_SE"
    prefix_ma = f"{prefix}_MA"

    plot_and_save_metric_vs_time(SE_best, SE_baseline, 'ALL', 'SE', 'Spectral Efficiency (bits/s/Hz)', prefix)
    plot_and_save_metric_vs_time_ma(SE_best, SE_baseline, 'ALL', 'SE', 'Spectral Efficiency (bits/s/Hz)', prefix_ma)

    print("\n✅ Script execution complete. All graphs and data saved.")






import os
import numpy as np
import matplotlib.pyplot as plt

def plot_metrics_vs_user(saved_folder, best_episode_index, graph_folder):
    """
    Loads baseline and best episode data, computes mean metrics across users,
    generates plots for Throughput, SINR, Interference, and Spectral Efficiency per user,
    and saves the plots in multiple formats.
    
    Parameters:
    - saved_folder: str, Path to the folder containing the .npy files.
    - best_episode_index: int, The index of the best episode to use for comparison.
    - graph_folder: str, Path to the folder where the plots will be saved.
    """
    
    # Ensure the save folder exists
    if not os.path.exists(graph_folder):
        os.makedirs(graph_folder)

    # Load baseline data
    Thrpt_baseline = np.load(f'{saved_folder}/Baseline_Thrpt.npy')     # (20, 61)
    SINR_baseline = np.load(f'{saved_folder}/Baseline_SINR.npy')       # (20, 61)
    Intf_baseline = np.load(f'{saved_folder}/Baseline_Intf.npy')       # (20, 61)
    SE_baseline = np.load(f'{saved_folder}/Baseline_SE.npy')       # (20, 61)

    # Load best episode data
    Thrpt_best = np.load(f'{saved_folder}/Episode_{best_episode_index}_Thrpt.npy')
    SINR_best = np.load(f'{saved_folder}/Episode_{best_episode_index}_SINR.npy')
    Intf_best = np.load(f'{saved_folder}/Episode_{best_episode_index}_Intf.npy')
    SE_best = np.load(f'{saved_folder}/Episode_{best_episode_index}_SE.npy')  # Newly added

    # Compute mean across timesteps (axis=1) for each user
    mean_thrpt_baseline = np.mean(Thrpt_baseline, axis=1)
    mean_thrpt_best = np.mean(Thrpt_best, axis=1)

    mean_sinr_baseline = np.mean(SINR_baseline, axis=1)
    mean_sinr_best = np.mean(SINR_best, axis=1)

    mean_intf_baseline = np.mean(Intf_baseline, axis=1)
    mean_intf_best = np.mean(Intf_best, axis=1)

    SE_baseline_transposed = SE_baseline.T
    SE_best_transposed = SE_best.T

    users = np.arange(1, 21)  # User indices (1 to 20)



    print("mean_sinr_baseline :",mean_sinr_baseline)
    print("mean_sinr_best :",mean_sinr_best)

    print("mean_thrpt_baseline :",mean_thrpt_baseline)
    print("mean_thrpt_best :",mean_thrpt_best)

    # Console output
    print(f"✅ Episode Index: {best_episode_index} | Mean Throughput (first 10 users): {np.mean(mean_thrpt_best[:10]):.4f} Mbps")

    # Create figure with 2x2 subplots
    plt.figure(figsize=(18, 10))

    # 1. Throughput
    plt.subplot(2, 2, 1)
    plt.plot(users, mean_thrpt_baseline, label='Baseline', marker='o',color ='r')
    plt.plot(users, mean_thrpt_best, label=f'Episode {best_episode_index}', marker='x',color ='b')
    plt.title('Mean Throughput per User')
    plt.xlabel('User Index')
    plt.ylabel('Throughput (Mbps)')
    plt.legend()
    plt.grid(True)

    # 2. SINR
    plt.subplot(2, 2, 2)
    plt.plot(users, mean_sinr_baseline, label='Baseline', marker='o',color ='r')
    plt.plot(users, mean_sinr_best, label=f'Episode {best_episode_index}', marker='x',color ='b')
    plt.title('Mean SINR per User')
    plt.xlabel('User Index')
    plt.ylabel('SINR (dB)')
    plt.legend()
    plt.grid(True)

    # 3. Interference
    plt.subplot(2, 2, 3)
    plt.plot(users, mean_intf_baseline, label='Baseline', marker='o',color ='r')
    plt.plot(users, mean_intf_best, label=f'Episode {best_episode_index}', marker='x',color ='b')
    plt.title('Mean Interference per User')
    plt.xlabel('User Index')
    plt.ylabel('Interference (dB or mW)')
    plt.legend()
    plt.grid(True)

    # 4. Spectral Efficiency (SE)
    plt.subplot(2, 2, 4)
    plt.plot( np.arange(1, 122), SE_baseline_transposed, label=f'Baseline {best_episode_index}', marker='o',color ='r')
    plt.plot( np.arange(1, 122), SE_best_transposed, label=f'SE Episode {best_episode_index}', marker='x', color ='b')
    plt.title('Mean Spectral Efficiency per User')
    plt.xlabel('User Index')
    plt.ylabel('SE (bits/s/Hz)')
    plt.legend()
    plt.grid(True)

    # Save and display
    plt.tight_layout()
    filename_base = os.path.join(graph_folder, f'Metrics_vs_User_Episode_{best_episode_index}')
    plt.savefig(f'{filename_base}.png', dpi=300)
    plt.savefig(f'{filename_base}.pdf', dpi=300)
    plt.savefig(f'{filename_base}.eps', dpi=300, format='eps')
    print(f"✅ Saved plots to: {filename_base}.{{png, pdf, eps}}")

    plt.show()
    plt.close()


def custom_save_plot_metrics_vs_user(saved_folder, best_episode_index, graphs_folder):
    """
    This function loads baseline and best episode data, computes mean metrics across users,
    generates plots for Throughput, SINR, and Interference per user, and saves the plots.
    
    Parameters:
    - saved_folder: str, Path to the folder containing the .npy files.
    - best_episode_index: int, The index of the best episode to use for comparison.
    - graphs_folder: str, Path to the folder where the plots will be saved.
    """
    
    # Ensure the save folder exists
    if not os.path.exists(graphs_folder):
        os.makedirs(graphs_folder)

    # Step 2: Load baseline data
    Thrpt_baseline = np.load(f'{saved_folder}/Baseline_Thrpt.npy')     # (20, 61)
    SINR_baseline = np.load(f'{saved_folder}/Baseline_SINR.npy')       # (20, 61)
    Intf_baseline = np.load(f'{saved_folder}/Baseline_Intf.npy')       # (20, 61)

    # Step 3: Load best episode data
    Thrpt_best = np.load(f'{saved_folder}/Episode_{best_episode_index}_Thrpt.npy')
    SINR_best = np.load(f'{saved_folder}/Episode_{best_episode_index}_SINR.npy')
    Intf_best = np.load(f'{saved_folder}/Episode_{best_episode_index}_Intf.npy')

    # Step 4: Compute mean across timesteps (axis=1) for each user
    mean_thrpt_baseline = np.mean(Thrpt_baseline, axis=1)
    mean_thrpt_best = np.mean(Thrpt_best, axis=1)

    mean_sinr_baseline = np.mean(SINR_baseline, axis=1)
    mean_sinr_best = np.mean(SINR_best, axis=1)

    mean_intf_baseline = np.mean(Intf_baseline, axis=1)
    mean_intf_best = np.mean(Intf_best, axis=1)

    # Print info about the first 10 users (LEO) for throughput
    print(f"✅ Episode Index: {best_episode_index} with mean throughput (first 10 users): {np.mean(mean_thrpt_best[:10]):.4f} Mbps")

    users = np.arange(1, 21)  # User indices (1 to 20)

    # Step 5: Plot all three metrics
    plt.figure(figsize=(18, 5))

    # Plot Throughput
    plt.subplot(1, 3, 1)
    plt.plot(users, mean_thrpt_baseline, label='Baseline', marker='o')
    plt.plot(users, mean_thrpt_best, label=f'Episode {best_episode_index}', marker='x')
    plt.title('Mean Throughput per User')
    plt.xlabel('User Index')
    plt.ylabel('Throughput (Mbps)')
    plt.legend()
    plt.grid(True)

    # Plot SINR
    plt.subplot(1, 3, 2)
    plt.plot(users, mean_sinr_baseline, label='Baseline', marker='o')
    plt.plot(users, mean_sinr_best, label=f'Episode {best_episode_index}', marker='x')
    plt.title('Mean SINR per User')
    plt.xlabel('User Index')
    plt.ylabel('SINR (dB)')
    plt.legend()
    plt.grid(True)

    # Plot Interference
    plt.subplot(1, 3, 3)
    plt.plot(users, mean_intf_baseline, label='Baseline', marker='o')
    plt.plot(users, mean_intf_best, label=f'Episode {best_episode_index}', marker='x')
    plt.title('Mean Interference per User')
    plt.xlabel('User Index')
    plt.ylabel('Interference (dB or mW)')
    plt.legend()
    plt.grid(True)

    # Create the graph file name by embedding version, case, and run
    parts = saved_folder.split('\\')
    version = parts[1]  # v1 or v2
    case = parts[2]  # Case 1 or Case 2
    run = parts[3]  # Run 1, Run 2, or Run 3
    print(parts)
    
    graph_filename = f'{version}_{case}_{run}_metrics_vs_user.png'
    graph_filepath = os.path.join(graphs_folder, graph_filename)
    
    # Save the figure
    plt.savefig(graph_filepath)
    print(f"Graph saved as {graph_filepath}")
    plt.close()


