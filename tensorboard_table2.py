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
        "Residual and Entropy": [
            "results/motion_im/chicken_01_realreal/tb",
        ],
        "Residual and Entropy and Demo": [
            "results/motion_im/chicken_01_realreal2/tb",
        ],
        "Entropy and No Residual and Root": [
            "results/motion_im/chicken_01_realreal3/tb",
        ],
        "No Entropy and No Residual and No Demo": [
            "results/motion_im/chicken_01_realreal4/tb",
        ]
    }

    tags = ["reward_6", "total_reward", "episode_len"]

    results = {}

    for tag in tags:
        for group_name, files in file_groups.items():
            # Special condition for "Entropy and No Residual and Root"
            current_tag = "reward_4" if group_name == "Entropy and No Residual and Root" and tag == "reward_6" else tag

            max_value, min_value = calculate_max_min(files, current_tag)
            if tag not in results:
                results[tag] = {}
            results[tag][group_name] = {
                "max": max_value,
                "min": min_value
            }

    # Create a figure with three separate tables in a horizontal grid layout
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    for idx, tag in enumerate(tags):
        ax = axs[idx]
        ax.axis('tight')
        ax.axis('off')

        # Prepare table data for the current tag
        table_data = [["Group", "Max", "Min"]]
        for group_name, values in results.get(tag, {}).items():
            table_data.append([group_name, f"{values['max']:.2f}", f"{values['min']:.2f}"])

        # Create table
        table = ax.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.4, 0.3, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width([0, 1, 2])

        # Set the title for each table
        ax.set_title("Pose Error" if tag in ["reward_6", "reward_4"] else tag.replace('_', ' ').title())

    # Adjust layout to ensure clarity
    plt.subplots_adjust(wspace=0.2)

    # Save the figure
    plt.savefig("residual_max_min_all_metrics_horizontal_grid.png")
    plt.close()

if __name__ == "__main__":
    main()
