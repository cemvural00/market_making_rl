import matplotlib.pyplot as plt
import os

def plot_pnl_distribution(pnls, out_dir):
    plt.figure(figsize=(8,4))
    plt.hist(pnls, bins=50, alpha=0.7)
    plt.title("PnL Distribution")
    plt.xlabel("PnL")
    plt.ylabel("Frequency")

    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "pnl_distribution.png"))
    plt.close()
