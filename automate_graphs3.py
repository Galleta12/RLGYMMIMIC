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

    return steps, aligned_values

def plot_without_error_bands(steps, values, label, color):
    """Plot data without error bands."""
    plt.plot(steps, values, label=label, color=color, linewidth=2)

def main():
    # TensorBoard file groups
    file_groups = {
        "Standard AMP": [
            "results/motion_im/amp_location_fastreal1/tb",
        ],
        "AMP Residual": [
            "results/motion_im/amp_location_fastrealimplicit/tb",
        ]
    }

    tags = {
        "reward_3": "Task Reward",
        "reward_4": "Style Reward",
        "total_reward": "Total Reward",
        "episode_len": "Episode Length"
    }

    colors = {
        "Standard AMP": "blue",
        "AMP Residual": "red",
    }

    y_limits = {}

    # Step 1: Generate combined plots to determine y-axis limits
    for tag, tag_label in tags.items():
        plt.figure()
        y_min, y_max = float('inf'), float('-inf')

        for group_name, files in file_groups.items():
            steps, values = aggregate_data(files, tag)
            for v in values:
                plot_without_error_bands(steps, v, label=f"{group_name}", color=colors[group_name])
                y_min = min(y_min, np.min(v))
                y_max = max(y_max, np.max(v))

        # Save the y-axis limits for this tag
        y_limits[tag] = (y_min, y_max)

        plt.title(f"{tag_label} - Combined")
        plt.xlabel("Steps Epoch")
        plt.ylabel(tag_label)
        plt.legend(title="Legend", loc="upper right", fontsize="small")
        plt.grid()
        plt.savefig(f"amp_{tag}_all.png")
        plt.close()

    # Step 2: Generate individual plots using the captured y-axis limits
    for tag, tag_label in tags.items():
        y_min, y_max = y_limits[tag]

        for group_name, files in file_groups.items():
            steps, values = aggregate_data(files, tag)
            plt.figure()
            for v in values:
                plot_without_error_bands(steps, v, label=f"{group_name}", color=colors[group_name])
            plt.ylim(y_min, y_max)  # Apply the same y-axis limits
            plt.title(f"{tag_label} - {group_name}")
            plt.xlabel("Steps Epoch")
            plt.ylabel(tag_label)
            plt.legend()
            plt.grid()
            plt.savefig(f"amp_{tag}_{group_name.replace(' ', '_')}.png")
            plt.close()

if __name__ == "__main__":
    main()
