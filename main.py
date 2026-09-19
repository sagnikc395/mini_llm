from pathlib import Path

# import tiktoken
# import torch
# import torch.nn.functional as F

from mini_llm.pretraining.tokenizer_run import tokenizer

PROJECT_ROOT = Path(__file__).resolve().parent

def main():
    file_path = PROJECT_ROOT / "data" / "the_verdict.txt"
    text_data = file_path.read_text(encoding="utf-8")

    # check the number of characters and tokens in the dataset
    total_characters = len(text_data)
    total_tokens = len(tokenizer.encode(text_data))
    print(f"Characters: {total_characters}")
    print(f"Tokens: {total_tokens}")


if __name__ == "__main__":
    main()
