
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "data/multitask/"
y_entry = np.load(DATA_DIR + "y_entry.npy")
y_direction = np.load(DATA_DIR + "y_direction.npy")

def plot_label_distribution(y, title, labels=["0", "1"], filename="dist.png"):
    counts = np.bincount(y)
    plt.figure(figsize=(5, 4))
    plt.bar(labels, counts, color="skyblue")
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Count")
    for i, c in enumerate(counts):
        plt.text(i, c + max(counts) * 0.01, f"{c}", ha='center')
    plt.tight_layout()
    plt.savefig(filename)
    print(f"✅ 저장 완료: {filename}")

plot_label_distribution(y_entry, "Entry Label Distribution", filename="entry_dist.png")
plot_label_distribution(y_direction, "Direction Label Distribution", filename="direction_dist.png")
