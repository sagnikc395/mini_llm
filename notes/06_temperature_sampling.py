"""How temperature reshapes the next-token distribution.

Below 1 the distribution sharpens toward the argmax (deterministic, repetitive);
above 1 it flattens toward uniform (varied, incoherent). Saves a bar chart to
plots/.
"""

from datetime import datetime

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from mini_llm.paths import PLOTS_DIR  # noqa: E402

VOCAB = [
    "closer",
    "every",
    "effort",
    "forward",
    "inches",
    "moves",
    "pizza",
    "toward",
    "you",
]
# made-up logits over that nine-word vocabulary
NEXT_TOKEN_LOGITS = torch.tensor(
    [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
)
TEMPERATURES = [1, 0.1, 5]


def main():
    scaled = [torch.softmax(NEXT_TOKEN_LOGITS / t, dim=0) for t in TEMPERATURES]
    for t, probs in zip(TEMPERATURES, scaled):
        top = VOCAB[int(probs.argmax())]
        print(f"T={t}: max probability {probs.max():.3f} on {top!r}")

    x = torch.arange(len(VOCAB))
    bar_width = 0.15
    fig, ax = plt.subplots(figsize=(5, 3))
    for i, (t, probs) in enumerate(zip(TEMPERATURES, scaled)):
        ax.bar(x + i * bar_width, probs, bar_width, label=f"temperature = {t}")
    ax.set_ylabel("probability")
    ax.set_xticks(x)
    ax.set_xticklabels(VOCAB, rotation=90)
    ax.legend()
    fig.tight_layout()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOTS_DIR / f"temperature_{datetime.now():%d-%m-%Y_%H-%M-%S}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
