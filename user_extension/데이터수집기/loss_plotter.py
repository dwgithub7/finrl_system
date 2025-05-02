
import matplotlib.pyplot as plt

def plot_losses(entry_losses, dir_losses, save_path="loss_plot.png"):
    epochs = list(range(1, len(entry_losses) + 1))
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, entry_losses, label="Entry Loss", marker="o")
    plt.plot(epochs, dir_losses, label="Direction Loss", marker="o")
    plt.title("Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ 저장 완료: {save_path}")
