import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_data_from_tensorboard(file_path, tag):
    """Extract data for a specific tag from a TensorBoard file."""
    event_acc = EventAccumulator(file_path)
    event_acc.Reload()
    if tag not in event_acc.Tags()['scalars']:
        raise ValueError(f"Tag {tag} not found in {file_path}")
    scalar_data = event_acc.Scalars(tag)
    steps = np.array([s.step for s in scalar_data])
    values = np.array([s.value for s in scalar_data])
    return steps, values

def aggregate_data(file_paths, tag):
    """Aggregate data for a specific tag across multiple files."""
    all_steps = []
    all_values = []

    for file_path in file_paths:
        steps, values = extract_data_from_tensorboard(file_path, tag)
        all_steps.append(steps)
        all_values.append(values)

    # Align data by the minimum length
    min_steps = min([len(steps) for steps in all_steps])
    aligned_values = np.array([values[:min_steps] for values in all_values])
    steps = all_steps[0][:min_steps]

    mean_values = np.mean(aligned_values, axis=0)
    std_values = np.std(aligned_values, axis=0)

    return steps, mean_values, std_values

def plot_with_error_bands(steps, mean_values, std_values, label, color):
    """Plot data with error bands."""
    plt.plot(steps, mean_values, label=label, color=color, linewidth=2)
    plt.fill_between(steps, mean_values - std_values, mean_values + std_values, alpha=0.1, color=color)

def main():
    # TensorBoard file groups
    file_groups = {
        "Standard": [
            "results/motion_im/back_flip_seed1/tb",
            "results/motion_im/back_flip2/tb",
            "results/motion_im/back_flip3/tb"
        ],
        "Demo Replay": [
            "results/motion_im/back_flip_demo_seed1/tb",
            "results/motion_im/back_flip_demo_seed2/tb",
            "results/motion_im/back_flip_demo_seed3/tb"
        ],
        "No Entropy": [
            "results/motion_im/back_flip_noentropy/tb",
            "results/motion_im/back_flip_noentropy2/tb",
            "results/motion_im/back_flip_noentropy3/tb"
        ]
    }

    tags = {"reward_5": "Pose Error", "total_reward": "Total Reward", "episode_len": "Episode Length"}

    colors = {"Standard": "blue", "Demo Replay": "red", "No Entropy": "lightgreen"}

    for tag, tag_label in tags.items():
        for group_name, files in file_groups.items():
            steps, mean_values, std_values = aggregate_data(files, tag)
            plt.figure()
            plot_with_error_bands(steps, mean_values, std_values, label=f"{group_name} (std dev)", color=colors[group_name])
            plt.title(f"{tag_label} - {group_name}")
            plt.xlabel("Steps")
            plt.ylabel(tag_label)
            plt.legend()
            plt.grid()
            plt.savefig(f"{tag}_{group_name}.png")
            plt.close()

    # Combined plots with error bands
    for tag, tag_label in tags.items():
        plt.figure()
        for group_name, files in file_groups.items():
            steps, mean_values, std_values = aggregate_data(files, tag)
            plot_with_error_bands(steps, mean_values, std_values, label=f"{group_name} (std dev)", color=colors[group_name])
        plt.title(f"{tag_label} - Combined with Error Bands")
        plt.xlabel("Steps")
        plt.ylabel(tag_label)
        plt.legend(title="Legend (std dev: shaded area)")
        plt.grid()
        plt.savefig(f"{tag}_all_with_error_bands.png")
        plt.close()

if __name__ == "__main__":
    main()
