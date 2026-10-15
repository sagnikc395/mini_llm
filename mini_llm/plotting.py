"""Loss curves, saved to plots/ rather than shown.

Saving instead of calling plt.show() keeps the CLI non-interactive; the
filename carries a timestamp down to the second so successive runs on the same
day do not overwrite each other.
"""

from datetime import datetime
from pathlib import Path

import matplotlib

# a non-interactive backend: nothing here ever opens a window
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from mini_llm.paths import PLOTS_DIR  # noqa: E402


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses) -> Path:
    """Plot train/val loss against epochs and save the figure.

    The second x-axis is purely cosmetic: it shares the y-axis and plots an
    invisible series so the top ticks read as tokens seen.
    """
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="validation loss")
    ax1.set_xlabel("epochs")
    ax1.set_ylabel("loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("tokens seen")
    fig.tight_layout()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    out_path = PLOTS_DIR / f"loss_{timestamp}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"saved loss plot to {out_path}")
    return out_path
