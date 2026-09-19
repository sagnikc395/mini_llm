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
