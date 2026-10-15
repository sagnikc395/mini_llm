"""Why every transformer block wraps its sublayers in a residual.

Print the gradient magnitude at each layer of a five-layer network, with the
shortcut connections on and off. Without them the gradient shrinks by roughly
an order of magnitude per layer on the way back -- the vanishing gradient that
makes deep stacks untrainable.
"""

import torch
from torch import nn

from mini_llm.model import GELU


class DeepNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList(
            nn.Sequential(nn.Linear(a, b), GELU())
            for a, b in zip(layer_sizes, layer_sizes[1:])
        )

    def forward(self, x):
        for layer in self.layers:
            out = layer(x)
            # the shortcut only applies where the shapes line up
            x = x + out if self.use_shortcut and x.shape == out.shape else out
        return x


def print_gradients(model, x):
    loss = nn.MSELoss()(model(x), torch.tensor([[0.0]]))
    loss.backward()
    for name, param in model.named_parameters():
        if "weight" in name:
            print(f"  {name}: {param.grad.abs().mean().item():.8f}")


def main():
    layer_sizes = [3, 3, 3, 3, 3, 1]
    sample_input = torch.tensor([[1.0, 0.0, -1.0]])

    for use_shortcut in (False, True):
        torch.manual_seed(123)
        print(f"shortcut={use_shortcut}:")
        print_gradients(DeepNetwork(layer_sizes, use_shortcut), sample_input)


if __name__ == "__main__":
    main()
