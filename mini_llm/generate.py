import torch


# function for the GPT model to generate the next text
def generate_text_simple(model,idx,max_new_tokens,context_size):
    # iterates for a specified number of new tokens to be generated , crop the curent context
    # # to fit the model's maximum context size, compute the predictions and then select the next token based
    # # on the highest probability prediction.
    # idx -> (batch,n_tokens) array of indices in the current context
    #
    for _ in range(max_new_tokens):
        idx_cond = idx[:,-context_size:]
        with torch.no_grad():
            logits = model(idx_cond)

            # focus only on the last time step so that
            logits = logits[:,-1,:]
            # (batch,n_token,vocab_size) becomes (batch,vocab_size)
            probs = torch.softmax(logits,dim=-1)
            idx_next = torch.argmax(probs,dim=-1,keepdim=True)
            idx = torch.cat((idx,idx_next),dim=1)

        return idx
