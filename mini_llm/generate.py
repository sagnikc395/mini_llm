import torch

# utility function for text to token ids conversion
#
import tiktoken
from mini_llm.config import GPT_CONFIG_124M as cfg
from mini_llm.architecture.gpt_model import GPTModel


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # adding the batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # removes the batch dimension
    return tokenizer.decode(flat.tolist())


# function for the GPT model to generate the next text
def generate_text_simple(model, idx, max_new_tokens, context_size):
    # iterates for a specified number of new tokens to be generated , crop the curent context
    # # to fit the model's maximum context size, compute the predictions and then select the next token based
    # # on the highest probability prediction.
    # idx -> (batch,n_tokens) array of indices in the current context
    #
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)

            # focus only on the last time step so that
            logits = logits[:, -1, :]
            # (batch,n_token,vocab_size) becomes (batch,vocab_size)
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.argmax(probs, dim=-1, keepdim=True)
            idx = torch.cat((idx, idx_next), dim=1)

    return idx
