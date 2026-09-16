# utility function for text to token ids conversion
#
import tiktoken
from mini_llm.generate import generate_text_simple
import torch
from mini_llm.config import GPT_CONFIG_124M as cfg
from mini_llm.architecture.gpt_model import GPTModel


def text_to_token_ids(text,tokenizer):
    encoded = tokenizer.encode(text,allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # adding the batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # removes the batch dimension
    return tokenizer.decode(flat.tolist())

start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

torch.manual_seed(123)
model = GPTModel(cfg)
model.eval() # disables dropout for deterministic inference

token_ids = generate_text_simple(
    model = model,
    idx = text_to_token_ids(start_context, tokenizer),
    max_new_tokens=10,
    context_size=cfg.context_length
)
print(f"output text: \n{token_ids_to_text(token_ids,tokenizer)}")
