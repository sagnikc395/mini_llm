import tiktoken
import torch
import torch.nn.functional as F


# from mini_llm.architecture.mini_llm_gpt import GPTModel
from mini_llm.config import GPT_CONFIG_124M as cfg
from mini_llm.architecture.gpt_model import GPTModel
from mini_llm import generate


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
    model = GPTModel(cfg)
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

    total_params_gpt2 = total_params - sum(
        p.numel() for p in model.out_head.parameters()
    )

    print(
        f"Number of trainable parameters, considering weight tying {total_params_gpt2}"
    )

    ## memory requirements calculation
    total_size_bytes = total_params * 4
    # calcs the total size in bytes( assuming flaot32, 4 bytes per parameter)
    total_size_mb = total_size_bytes / (1024 * 1024)
    # converting to bytes
    print(f"Total size of the model is : {total_size_mb:.2f} MB")

    start_context = "Hello, I am"
    encoded = tokenizer.encode(start_context)
    print(f"encoded: {encoded}")
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # adding batch dimension
    print(f"encoded_tensor.shape {encoded_tensor.shape}")

    # model eval
    model.eval()
    out = generate.generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=6,
        context_size=cfg.context_length,
    )
    print(f"output: {out}")
    print(f"output length: {len(out[0])}")

    start_context = "Every effort moves you"
    tokenizer = tiktoken.get_encoding("gpt2")

    torch.manual_seed(123)
    model = GPTModel(cfg)
    model.eval()  # disables dropout for deterministic inference

    token_ids = generate.generate_text_simple(
        model=model,
        idx=generate.text_to_token_ids(start_context, tokenizer),
        max_new_tokens=10,
        context_size=cfg.context_length,
    )
    print(f"output text: \n{generate.token_ids_to_text(token_ids, tokenizer)}")

    # feed the inputs to the model to calculate logits vector for the 2 input examples
    inputs = torch.tensor([[16833, 3626, 6100], [40, 1107, 588]])
    targets = torch.tensor([[3626, 6100, 345], [1107, 588, 11311]])

    with torch.no_grad():
        logits = model(inputs)
    # prob of each token in vocabulary
    probas = torch.softmax(logits, dim=-1)
    print(probas.shape)

    # applying the argmax function to the probability scores to obtain the corresponding token IDS
    token_ids = torch.argmax(probas, dim=-1, keepdim=True)
    print(f"Token IDS:\n {token_ids}")

    print(f"Targets batch 1: {generate.token_ids_to_text(targets[0], tokenizer)}")
    print(
        f"Outputs batch 1: {generate.token_ids_to_text(token_ids[0].flatten(), tokenizer)}"
    )

    # for each of the two input texts, now we can print the initial softmax probability scores corresponding to the target tokens using the code
    text_idx = 0
    target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
    print(f"Text 1: {target_probas_1}")

    text_idx = 1
    target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
    print(f"Text 2: {target_probas_2}")

    # loss for it
    log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
    print(log_probas)

    avg_log_probas = torch.mean(log_probas)
    print(f"average log loss : {avg_log_probas}")

    neg_avg_log_probas = avg_log_probas * -1
    print(neg_avg_log_probas)

    # shapes of the logits and the target tensors
    print(f"Logits Shape: {logits.shape}")
    print(f"Targets Shape: {targets.shape}")

    # flatten these tensors by combining them over the batch dimensions
    logits_flat = logits.flatten(0, 1)
    targets_flat = targets.flatten()
    print(f"Flattened logits: {logits_flat.shape}")
    print(f"flattened targets: {targets_flat.shape}")

    loss = F.cross_entropy(logits_flat, targets_flat)
    print(f"Loss : {loss}\n")


if __name__ == "__main__":
    main()
