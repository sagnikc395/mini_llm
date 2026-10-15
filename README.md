## mini-llm

following along the  { sebastian raschka (Build a Large Language Model (From Scratch)) + nathan lambert(Reinforcement Learning from Human Feedback - aka the post-training) } books.

## mental model of the whole flow

```
TRAINING SYSTEM                         INFERENCE SYSTEM

raw text
   ↓
tokenizer
   ↓
pretraining
   ↓
base checkpoint
   ↓
SFT / DPO / RLHF / RLVR
   ↓
final checkpoint
   │
   └───────────────────────→ model loader
                                ↓
                           inference runtime
                                ↓
                         KV cache + scheduler
                                ↓
                           sampling / decode
                                ↓
                              server
```


## running it

```bash
# train the small GPT from scratch on data/the_verdict.txt (downloaded on first run)
uv run python main.py train --epochs 10
uv run python main.py train --epochs 1 --debug --no-save   # quick sanity run

# download OpenAI's original GPT-2 weights (124M / 355M / 774M / 1558M)
uv run python main.py download-weights --size 124M

# load those weights into our GPTModel and sample from them
uv run python main.py pretrained --prompt "Every effort moves you" --top-k 50 --temperature 1.0

# build the spam-classification splits for finetuning
uv run python main.py spam-data
```

## layout

`main.py` is a four-line shim over `mini_llm/cli.py`; everything reusable is a
flat module under `mini_llm/`, roughly in the order the pieces stack up:

```
mini_llm/
  config.py        GPTConfig, and the four GPT-2 sizes
  tokenizer.py     a word-level tokenizer (reference only; we use tiktoken)
  attention.py     CausalAttention, MultiHeadAttention
  model.py         GELU, LayerNorm, FeedForward, TransformerBlock, GPTModel
  generate.py      text <-> token ids, and the sampling loop
  data.py          the_verdict.txt, sliced into training windows
  training.py      losses, the training loop, device selection
  gpt2_weights.py  OpenAI's pretrained checkpoints -> our GPTModel
  spam.py          the spam-classification finetuning dataset
  plotting.py      loss curves
  paths.py         every on-disk location, derived from the repo root
  cli.py           the click group behind main.py
notes/             step-by-step scripts from working through the book
```

Importing anything under `mini_llm/` is free of side effects — nothing
downloads, trains or prints at import time.

## decisions made along the way

- **flat modules, one concept each.** the package used to be
  `architecture/`, `attention/`, `loss/`, `pretraining/`, `finetuning/`, most of
  them holding a single file, with two different `GPTModel` classes and three
  competing config objects. ~1700 lines does not need a directory tree; it needs
  `model.py`, `attention.py`, `config.py`.
- **the library never runs anything at import time.** several modules used to
  print, train or *download a file* just from being imported — `import
  mini_llm.pretraining.download_data` fetched the-verdict.txt as a side effect.
  the exploratory scripts now live in `notes/`, import from `mini_llm/`, and
  nothing imports them back.
- **one config object, `GPTConfig`.** `gpt2_config("355M")` derives the
  pretrained variants from it. the old `finetuning/config.py` dict had
  `contenxt_length` and `n_heades` typos that silently did nothing, which is
  exactly the failure a single typed dataclass prevents.
- **one `generate()`, not `generate` plus `generate_text_simple`.** temperature
  0 with no top-k *is* greedy decoding, so the "simple" variant was the same
  function with two branches removed.
- **paths come from `paths.py`, never the cwd.** the spam pipeline used to write
  its csvs wherever you happened to run it from, and the splits ended up
  committed inside the package. they are generated artifacts, so they live in
  the gitignored `data/spam/`.
- **click, not argparse, for anything with a CLI.** `main.py` is one group
  (`train` / `pretrained` / `download-weights` / `spam-data`) rather than a pile
  of scripts, and `click.Choice` gives the model-size validation for free.
- **256-token context when training on the_verdict.txt.** the book is only ~5k
  tokens, so a 1024-token context leaves the 10% validation split with *zero*
  full-length windows -> empty loader -> nan validation loss. 256 is short
  enough that both splits actually yield batches. the pretrained path uses 1024,
  because that's the size of the checkpoint's positional embedding table.
- **weight loading sits next to the downloader that feeds it**, in
  `gpt2_weights.py`. two conversions matter there: TF stores dense layers as
  `(in, out)` while `nn.Linear` wants `(out, in)` (so every weight matrix is
  transposed on the way in), and the three attention projections live in one
  fused `c_attn` tensor that gets split into q/k/v thirds.
- **`_assign()` refuses a shape mismatch instead of reshaping.** a silent
  mismatch produces a model that runs and emits fluent-looking garbage, which is
  much harder to debug than an exception at load time. same reasoning behind the
  upfront check that the model was built with `qkv_bias=True` — GPT-2 trained the
  attention projections with biases, so a `qkv_bias=False` model has nowhere to
  put them.
- **tensorflow is imported inside the function that reads the checkpoint.** it's
  only needed for OpenAI's TF v1 files; the `train` path shouldn't pay a
  multi-second import for a dependency it never touches.
- **sampling always runs under `model.eval()`.** otherwise dropout is live
  during generation and quietly adds noise to the sampled distribution.
- **hyperparameters are module constants at the top of `cli.py`**, not scattered
  through the call sites — easier to see what the run actually was.
- **labels are mapped to ints (`ham`->0, `spam`->1) before the splits are written
  to csv**, not inside `SpamDataset.__getitem__`. the dataset hands
  `self.data["Label"]` straight to `torch.tensor(..., dtype=torch.long)`, which
  raises `TypeError: new(): invalid data type 'str'` on the raw strings. doing
  the mapping once means the csvs on disk are already model-ready.
- **the spam set is undersampled to balance it before splitting.** ~4800 ham vs
  ~750 spam means a classifier can hit 87% accuracy by always answering ham, so
  accuracy would say nothing. balancing before the split keeps train/val/test all
  balanced.
- **matplotlib uses the Agg backend and saves to `plots/`.** no run ever blocks
  on an open window, and the timestamped filenames mean successive runs don't
  overwrite each other.

## Additional Resources that I found helpful throughout this 

1. Reading the [Scaling Laws, Carefully](https://lilianweng.github.io/posts/2026-06-24-scaling-laws/) post helped me see that scaling laws are way less settled than I thought - the whole Kaplan vs Chinchilla disagreement mostly comes down to boring procedural stuff (whether you count embedding params, which loss region you fit, how far you extrapolate) rather than some deep conflict. Also got introduced to data-constrained scaling, which feels like the regime that actually matters now. Before this I had only learnt the Kaplan scaling laws from CS685. Blog post on my learning soon.

2. [PyTorch in One Hour: From Tensors to Training Neural Networks on Multiple GPUs](https://sebastianraschka.com/teaching/pytorch-1h/) is a amazing tutorial helped in getting speed with torch. Previously I had relegated a lot of coding to the agents especially in writing NN modules and writing the code, but writing dl code isnt that hard and is digestible. Give a read through this. 

3. Inference 
    - how to serve the model effectively ? 
    - [Paged Attention from First Principles](https://hamzaelshafie.bearblog.dev/paged-attention-from-first-principles-a-view-inside-vllm/) by Hamza Elshafie - walks through KV cache fragmentation and how vLLM borrows OS virtual memory ideas to fix it.
    - [Tutorial on LLM inference](https://github.com/garg-aayush/building-from-scratch/tree/main/llm-inference) - a separate from-scratch repo by Aayush Garg covering sampling (top-k / top-p), penalties, kv-cache and speculative decoding. this seems like a cool inference repo, checking this out.

4. [Post Training Course](https://rlhfbook.com/course) this is the companion course (13 lectures) to the same nathan lambert book above, that I plan to do after I finish this, as I am very interested in AI Alignment , but post training is something that I haven't done very handson, but rather read it mostly as a chapter in CS685. Excited to go over this lectures soon !

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">just received a great bed time reading book by the mail looking forward for tonight <a href="https://t.co/aKoXDIVVV3">pic.twitter.com/aKoXDIVVV3</a></p>&mdash; Yacine Mahdid (@yacinelearning) <a href="https://x.com/yacinelearning/status/2091988693984719353?ref_src=twsrc%5Etfw">August 24, 2026</a></blockquote> <script async src="https://platform.x.com/widgets.js" charset="utf-8"></script>

## Future Work 

1.[How to Scale Your Model](https://jax-ml.github.io/scaling-book/) (the "scaling book" by the google deepmind folks) , tbh this is a future reading for me that I plan to read (sometime ig). its a systems view of running LLMs on TPUs/GPUs and all the examples are in Jax, so its also my excuse to finally pick up Jax.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">This is a genuine question. How are folks using JAX for their production use cases doing it?<br><br>If I get enough signals from Twitter, I am happy to work on a little project that provides a streamlined workflow. <a href="https://x.com/algo_diver?ref_src=twsrc%5Etfw">@algo_diver</a> will be interested to collaborate, I know. <a href="https://t.co/q9tNaBguoj">https://t.co/q9tNaBguoj</a></p>&mdash; Sayak Paul (@RisingSayak) <a href="https://x.com/RisingSayak/status/1750731172131619137?ref_src=twsrc%5Etfw">January 26, 2024</a></blockquote> <script async src="https://platform.x.com/widgets.js" charset="utf-8"></script>

<br/> 
 (though going by this thread, Jax in production is more of a TPU/deepmind-shaped thing than a default) - seems interesting , would have a look through it later ig !
