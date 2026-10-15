# notes

Step-by-step scripts from working through the book: each one prints its way
through a single idea. They import from `mini_llm/` but nothing in `mini_llm/`
imports *them*, so they can be messy and chatty without that leaking into the
library.

Run one directly:

```bash
uv run python notes/01_attention_from_scratch.py
```

| script | what it shows |
| --- | --- |
| `01_attention_from_scratch.py` | dot-product scores -> softmax weights -> context vectors, then trainable Q/K/V and the causal mask |
| `02_tokenizer_from_scratch.py` | a word-level vocabulary vs. GPT-2 BPE on the same text |
| `03_embeddings_and_batching.py` | token + positional embeddings, and what a batch out of the dataloader looks like |
| `04_model_walkthrough.py` | parameter count, memory footprint, and cross-entropy computed by hand vs. `F.cross_entropy` |
| `05_residual_connections.py` | gradient magnitudes with and without residual connections |
| `06_temperature_sampling.py` | how temperature reshapes the next-token distribution |
| `07_sft_loss_sketch.py` | the shift-by-one masked log-prob that supervised finetuning optimizes |
