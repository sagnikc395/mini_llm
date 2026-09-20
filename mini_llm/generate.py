import torch

from mini_llm.attention.compact_self_attention import contenxt_length


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # adding the batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # removes the batch dimension
    return tokenizer.decode(flat.tolist())

# def softmax_with_temperate(logits,temperature):
#     scaled_logits = logits / temperature
#     return torch.softmax(scaled_logits,dim=0)

# def top_k_sampling(next_token_logits,top_k):
#     top_logits,top_pos = torch.topk(next_token_logits,top_k)
#     print(f"Top logits: {top_logits}")
#     print(f"Top positions: {top_pos}")

#     new_logits = torch.where(
#         # identify logits less than the mini
#         condition=next_token_logits < top_logits[-1],
#         # assign -inf to these lower logits
#         input=torch.tensor(float('-inf')),
#         # retains the original logits for all other tokens
#         other=next_token_logits
#     )

#     # apply the softmax
#     topk_probas = torch.softmax(new_logits,dim=0)
#     return topk_probas


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

## the core handler for everything that we can call directly and this would chain these and used
def generate(model,idx,max_new_tokens,context_size,temperature=0.0,top_k=None,eos_id=None):
    for _ in range(max_new_tokens):
        # for loop is the same as befor, gets logits and only focuses on the last time step
        idx_cond = idx[:,-context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:,-1,:]
        if top_k is not None:
            # filter logits with top_k sampling
            top_logits,_ = torch.topk(logits,top_k)
            min_val = top_logits[:,-1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits,
            )
        # apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits,dim=-1)
            idx_next = torch.multinomial(probs,num_samples=1)
        else:
            # carry out greedy next-token selection as before when temperature scaling is disabled
            idx_next = torch.argmax(logits,dim=-1,keepdim=True)
        # stop generating early if eos token is encoutered
        if idx_next == eos_id:
            break
        idx = torch.cat((idx,idx_next),dim=1)
    return idx
