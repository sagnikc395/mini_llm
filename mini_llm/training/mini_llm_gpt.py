import torch
import torch.nn as nn

from mini_llm.config import GPT_CONFIG_124M as cfg
from mini_llm.attention.multi_head_attention import MultiHeadAttention


class GPTModel(nn.Module):
    def __init__(self, cfg=cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.emb_dim)
        self.drop_emb = nn.Dropout(cfg.drop_rate)
        # this acts as a placeholder for TransformerBlock
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        # this act as a placeholder for LayerNorm
        self.final_norm = LayerNorm(cfg.emb_dim)
        self.out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


class TransformerBlock(nn.Module):
    # placeholder for a real transformerblock
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg.emb_dim,
            d_out=cfg.emb_dim,
            context_length=cfg.context_length,
            num_heads=cfg.n_heads,
            dropout=cfg.drop_rate,
            qkv_bias=cfg.qkv_bias,
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg.emb_dim)
        self.norm2 = LayerNorm(cfg.emb_dim)
        self.drop_shortcut = nn.Dropout(cfg.drop_rate)

    def forward(self, x):

        # shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        # adding back the original input back
        x = x + shortcut

        # shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        # adding back the original input back
        x = x + shortcut
        return x


class LayerNorm(nn.Module):
    # placeholder for a real layernorm block
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        # torch.set_printoptions(sci_mode=False)
        # mean = out.mean(dim=-1, keepdim=True)
        # var = out.var(dim=-1, keepdim=True)
        # print(f"Mean: {mean}")
        # print(f"Variance: {var}")

        # out_norm = (out - mean) / torch.sqrt(var)
        # mean = out_norm.mean(dim=-1, keepdim=True)
        # var = out_norm.var(dim=-1, keepdim=True)
        # print(f"normalized layer outputs: {out_norm}\n")
        # print(f"mean: {mean}\n")
        # print(f"variance: {var}\n")
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

    def simple_layer_norm_example(self):
        torch.manual_seed(123)
        batch_example = torch.randn(2, 5)
        layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
        out = layer(batch_example)
        return out


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


# use GELU to implement the small NN network module , FeedForward ,
# using it layer in the LLMs transfoer block later
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg.emb_dim, 4 * cfg.emb_dim),
            GELU(),
            nn.Linear(4 * cfg.emb_dim, cfg.emb_dim),
        )

    def forward(self, x):
        return self.layers(x)


# neural network with shortcut connections
class DeepNeuralNetworkShortcut(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
                nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
                nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
                nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
                nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU()),
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            # compute the output of the current layer
            layer_output = layer(x)
            # check if the shortcut can be applied
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output

        return x


# function that will compute the gradients in the model's backward pass
def print_gradients(model, x):
    output = model(x)
    # forward pass
    target = torch.tensor([[0.0]])

    # calculate the loss based on how close the target and outputs are
    loss = nn.MSELoss()
    loss = loss(output, target)
    loss.backward()

    for name, param in model.named_parameters():
        if "weight" in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")


if __name__ == "__main__":
    # LayerNorm(emb_dim=6).simple_layer_norm_example()
    # plot gelu vs relu side by side
    # import matplotlib.pyplot as plt

    # gelu, relu = GELU(), nn.ReLU()

    # x = torch.linspace(-3, 3, 100)
    # y_gelu, y_relu = gelu(x), relu(x)
    # plt.figure(figsize=(8, 3))
    # for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
    #    plt.subplot(1, 2, i)
    #    plt.plot(x, y)
    #    plt.title(f"{label} activation function")
    #    plt.xlabel("x")
    #    plt.ylabel(f"{label} (x)")
    #    plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    # init a new FeedForward module with a token embedding size of 768 and feed it batch input with two samples
    ffn = FeedForward(cfg())
    x = torch.rand(2, 3, 768)
    # create a smaple input with batch dimension 2
    out = ffn(x)
    print(out.shape)

    ## nn with shortcut connections
    layer_sizes = [3, 3, 3, 3, 3, 1]
    sample_input = torch.tensor([[1.0, 0.0, -1.0]])
    torch.manual_seed(123)
    model_without_shortcut = DeepNeuralNetworkShortcut(layer_sizes, use_shortcut=False)

    print_gradients(model_without_shortcut, sample_input)

    torch.manual_seed(123)
    model_with_shortcut = DeepNeuralNetworkShortcut(layer_sizes, use_shortcut=True)
    print_gradients(model_with_shortcut, sample_input)

    ## instantiating a transformer block and feed it some sample data
    torch.manual_seed(123)

    # create sample input of shape [batch_size, num_tokens,emb_dim]
    x = torch.rand(2, 4, 768)

    block = TransformerBlock(cfg)
    output = block(x)

    print(f"input shape: {x.shape}")
    print(f"output shape: {output.shape}")
