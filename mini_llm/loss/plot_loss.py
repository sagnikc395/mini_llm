from pathlib import Path
from datetime import datetime
import torch

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from mini_llm.generate import softmax_with_temperate

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

    vocab = {
    "closer": 0,
    "every": 1,
    "effort": 2,
    "forward": 3,
    "inches": 4,
    "moves": 5,
    "pizza": 6,
    "toward": 7,
    "you": 8,
    }
    inverse_vocab = {v: k for k, v in vocab.items()}

    # temperatures and scaled probabilities
    temperatures = [1,0.1,5]
    next_token_logits = torch.tensor([4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79])
    scaled_probas = [softmax_with_temperate(next_token_logits,T) for T in temperatures]

    x = torch.arange(len(vocab))
    bar_width = 0.15
    fig, ax = plt.subplots(figsize=(5, 3))
    for i, T in enumerate(temperatures):
        rects = ax.bar(x + i * bar_width, scaled_probas[i],bar_width, label=f'Temperature = {T}')
    ax.set_ylabel('Probability')
    ax.set_xticks(x)
    ax.set_xticklabels(vocab.keys(), rotation=90)
    ax.legend()
    plt.tight_layout()


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
