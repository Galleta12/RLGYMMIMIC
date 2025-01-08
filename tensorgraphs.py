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

def aggregate_and_plot(file_paths, tags, output_file=None):
    """Aggregate data for multiple TensorBoard files and plot with error bands."""
    for tag in tags:
        all_steps = []
        all_values = []

        # Extract data from all files
        for file_path in file_paths:
            steps, values = extract_data_from_tensorboard(file_path, tag)
            all_steps.append(steps)
            all_values.append(values)
        
        # Align data by steps
        min_steps = min([len(steps) for steps in all_steps])
        aligned_values = np.array([values[:min_steps] for values in all_values])

        # Calculate averages and standard deviations
        mean_values = np.mean(aligned_values, axis=0)
        std_values = np.std(aligned_values, axis=0)
        steps = all_steps[0][:min_steps]

        # Plot with error bands
        plt.figure()
        plt.plot(steps, mean_values, label=f"Average {tag}")
        plt.fill_between(steps, mean_values - std_values, mean_values + std_values, alpha=0.3, label=f"Std Dev {tag}")
        plt.xlabel("Steps")
        plt.ylabel("Values")
        plt.title(f"{tag} with Error Bands")
        plt.legend()
        plt.grid()

        # Save or show the plot
        if output_file:
            plt.savefig(f"{output_file}_{tag}.png")
        else:
            plt.show()

# Paths to TensorBoard files
tensorboard_files = [
    "results/motion_im/back_flip_seed1/tb",
    "results/motion_im/back_flip2/tb",
    "results/motion_im/back_flip3/tb"
]

# Tags of interest
tags_to_plot = ["reward_5", "total_reward"]

# Call the function
aggregate_and_plot(tensorboard_files, tags_to_plot, output_file="tensorboard_plot")
