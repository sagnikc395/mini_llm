"""A word-level vocabulary vs. GPT-2 BPE, on the same text."""

import tiktoken

from mini_llm.data import load_verdict
from mini_llm.tokenizer import SimpleTokenizer, build_vocab, tokenize


def main():
    raw_text = load_verdict()

    tokens = tokenize(raw_text)
    vocab = build_vocab(raw_text)
    print(f"{len(tokens)} tokens, {len(vocab)} distinct entries in the vocabulary")

    tokenizer = SimpleTokenizer(vocab)
    ids = tokenizer.encode(raw_text)
    print(f"first 50 ids: {ids[:50]}")
    print(f"decoded back: {tokenizer.decode(ids[:50])}")

    # the failure mode this tokenizer has and BPE does not: every word outside
    # the_verdict.txt collapses to a single <|unk|> id
    unseen = "Akwirw ier, do you like tea?"
    print(f"\nunseen text: {unseen}")
    print(f"round trip:  {tokenizer.decode(tokenizer.encode(unseen))}")

    # BPE never needs <|unk|>: anything it has not seen splits into subwords
    print("\n--- tiktoken (GPT-2 BPE) ---")
    enc = tiktoken.get_encoding("gpt2")
    bpe_ids = enc.encode(unseen)
    print(f"ids:        {bpe_ids}")
    print(f"as pieces:  {[enc.decode([i]) for i in bpe_ids]}")
    print(f"round trip: {enc.decode(bpe_ids)}")


if __name__ == "__main__":
    main()
