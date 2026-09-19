from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLOTS_DIR = PROJECT_ROOT / "plots"


# a plottler for plotting losses
#
def plot_losses(epochs_seen,tokens_seen,train_losses,val_losses):
    fig,ax1 = plt.subplots(figsize=[5,3])
    ax1.plot(epochs_seen,train_losses,label="Training loss")
    ax1.plot(
        epochs_seen,val_losses,linestyle="-.",label="Validation loss"
    )
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    # create a second x-axis that shares the same y-axis
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen,train_losses,alpha=0)# invisible plot for aligning tikcs
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()

    # save the plots instead of showing it; the timestamp goes down to seconds so
    # several runs on the same day don't overwrite each other
    PLOTS_DIR.mkdir(parents=True,exist_ok=True)
    now = datetime.now()
    filename = "gpt_model_epochs_vs_loss_"+now.strftime("%d-%m-%Y_%H-%M-%S")+".png"
    out_path = PLOTS_DIR / filename
    fig.savefig(out_path,dpi=150)
    plt.close(fig)
    print(f"Saved loss plot to {out_path}")
    return out_path
