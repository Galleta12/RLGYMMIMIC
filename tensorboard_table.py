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
    values = np.array([s.value for s in scalar_data])
    return values

def calculate_max_min(file_paths, tag):
    """Calculate the maximum and minimum values for a specific tag across multiple files."""
    all_values = []

    for file_path in file_paths:
        values = extract_data_from_tensorboard(file_path, tag)
        all_values.append(values)

    # Combine all values
    combined_values = np.concatenate(all_values)
    max_value = np.max(combined_values)
    min_value = np.min(combined_values)

    return max_value, min_value

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

    tags = ["reward_5", "total_reward", "episode_len"]

    results = {}

    for tag in tags:
        results[tag] = {}
        for group_name, files in file_groups.items():
            max_value, min_value = calculate_max_min(files, tag)
            results[tag][group_name] = {
                "max": max_value,
                "min": min_value
            }

    # Create three separate tables close together in a single figure
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))

    for idx, tag in enumerate(tags):
        ax = axes[idx]
        ax.axis('tight')
        ax.axis('off')

        # Prepare table data for the current tag
        table_data = [["Group", "Max", "Min"]]
        for group_name, values in results[tag].items():
            table_data.append([group_name, f"{values['max']:.2f}", f"{values['min']:.2f}"])

        # Create table
        table = ax.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.4, 0.3, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width([0, 1, 2])
        ax.set_title("Pose Error" if tag == "reward_5" else tag)

    plt.subplots_adjust(wspace=0.05)  # Reduce space between tables
    plt.savefig("max_min_all_metrics_side_by_side.png")
    plt.close()

if __name__ == "__main__":
    main()
