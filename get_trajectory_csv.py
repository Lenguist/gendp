import os
import csv
import numpy as np

# Change this to your top-level directory path
DIRECTORY_PATH = "/home/maksymbondarenko/Desktop/gendp/cube_picking_processed"
OUTPUT_DIR = "./episode_csvs"  # output directory for CSVs

# Create output directory if needed
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Process each episode
for episode_name in sorted(os.listdir(DIRECTORY_PATH)):
    episode_path = os.path.join(DIRECTORY_PATH, episode_name)
    robot_dir = os.path.join(episode_path, "robot")

    if not os.path.isdir(robot_dir):
        continue

    rows = []
    for fname in sorted(os.listdir(robot_dir)):
        if not fname.endswith(".txt"):
            continue

        frame_idx = int(fname.replace(".txt", ""))
        fpath = os.path.join(robot_dir, fname)

        try:
            data = np.loadtxt(fpath)
            flat_data = data.flatten().tolist()
            rows.append([frame_idx] + flat_data)
        except Exception as e:
            print(f"Failed to load {fpath}: {e}")

    if rows:
        # Determine column names dynamically
        num_values = len(rows[0]) - 1
        header = ["frame_idx"] + [f"v{i}" for i in range(num_values)]
        output_path = os.path.join(OUTPUT_DIR, f"{episode_name}.csv")

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

        print(f"Saved {episode_name}.csv with {len(rows)} frames.")
