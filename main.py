import tiktoken
import torch

# from mini_llm.training.mini_llm_gpt import GPTModel
from mini_llm.config import GPT_CONFIG_124M
from mini_llm.training.gpt_model import GPTModel


def main():
    # 1. tokenization stage
    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"

    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch, dim=0)
    print(batch)

    # 2. initialize instance and feed it the tokenized batch
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    total_params = sum(p.numel() for p in model.parameters())
    logits = model(batch)
    print(f"Input batch: \n{batch}")
    print(f"Output shape: \n{logits.shape}")
    print(
        f"Total number of parameters: {total_params}\n, is it 124 million ? {True if (124_000_000 == total_params) else False}"
    )
    print(logits)

    # print the outputs , the weight tensors for both of these layers and checking if they have the same shape or not
    print(f"token embedding layer shape: {model.tok_emb.weight.shape}")
    print(f"output layer shape: {model.out_head.weight.shape}")

    total_params_gpt2 = (
        total_params - sum(p.numel() for p in model.out_head.parameters())
    )

    print(f"Number of trainable parameters, considering weight tying {total_params_gpt2}")

    ## memory requirements calculation
    total_size_bytes = total_params * 4
    # calcs the total size in bytes( assuming flaot32, 4 bytes per parameter)
    total_size_mb = total_size_bytes / (1024 * 1024)
    # converting to bytes
    print(f"Total size of the model is : {total_size_mb:.2f} MB")






if __name__ == "__main__":
    main()
