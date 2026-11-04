# Attention Basics: Q, K, V and Heads, with an Inference Lens

Study notes written while starting Part 1. They cover what self-attention computes, where Q, K and V come from, what heads are, and why each of these matters for an inference engine.

## The shapes: B, T, D

The input to attention is a tensor `x` of shape `(B, T, D)`:

| Letter | Meaning | Example |
|---|---|---|
| `B` | Batch size: how many sequences are processed at once | 2 sentences |
| `T` | Tokens in each sequence for this forward pass | 8 tokens each |
| `D` | Numbers per token (the embedding size, `d_model`; `n_embd` in nanoGPT; `C` in Karpathy's video) | 64 numbers per token |

Before attention runs, text is tokenized into IDs, shorter sequences are padded so that every sequence in the batch has the same `T`, and each ID is turned into a vector of `D` numbers. Padding tokens are not real words, so attention needs a padding mask as well as the causal mask. Inference engines try to avoid padding, because the padded slots are computed but do no useful work.

`T` is not a fixed property of a model. It can change on every forward pass:

```text
Prefill: a 50-token prompt          → T = 50
Decode:  the next token only        → T = 1   (earlier K and V come from the KV cache)
Later:   a new 1,000-token prompt   → T = 1000
```

The model's limit is its context length (40,960 for Qwen3-0.6B). `T` can be anything up to it. The weights only depend on `D`, not `T`: `nn.Linear(D, D)` never mentions `T`, and every token passes through the same weights. That is why one model accepts inputs of any length.

## Single-head causal self-attention

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

B, T, D = 2, 8, 64
x = torch.randn(B, T, D)                             # (B, T, D)

to_q = nn.Linear(D, D, bias=False)
to_k = nn.Linear(D, D, bias=False)
to_v = nn.Linear(D, D, bias=False)

q, k, v = to_q(x), to_k(x), to_v(x)                  # each (B, T, D)

scores = q @ k.transpose(-2, -1) / math.sqrt(D)      # (B, T, T)
mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
scores = scores.masked_fill(~mask, float("-inf"))    # hide future tokens
weights = scores.softmax(dim=-1)                     # (B, T, T)
out = weights @ v                                    # (B, T, D)

check = F.scaled_dot_product_attention(q, k, v, is_causal=True)
print(torch.allclose(out, check, atol=1e-6))         # True
```

Three ideas:

1. **Q and K** score how much each token should look at every other token.
2. **The mask** stops a token from looking at future tokens.
3. **V** is what gets mixed together, using those scores as weights.

Exercises: print every shape and match it to the comments; delete the mask lines and see the check fail; then write the five lines from `scores` to `out` from memory.

## The score table

A score answers: "When token `i` is updated, how much should it look at token `j`?"

```text
score(i, j) = q[i] · k[j]
```

Every token's query is compared with every token's key, so each sequence gets a `T × T` table, and the batch gets `(B, T, T)`:

```text
q            @  k.transpose(-2, -1)  =  scores
(B, T, D)       (B, D, T)               (B, T, T)
```

The `D` in the middle disappears because the matrix multiplication computes all the dot products at once.

An example for "The cat sat", with invented numbers. Rows are queries, columns are keys:

```text
              key: The   cat   sat
query The          2.0   0.5  -1.0
query cat          0.3   2.5   1.0
query sat          0.1   3.0   2.0   ← "sat" looks mostly at "cat"
```

After the causal mask and softmax, the `sat` row becomes about `0.0, 0.73, 0.27`, so the new `sat` vector is about 73% of `cat`'s V and 27% of its own.

This table is the core idea of the Transformer: any token can look directly at any earlier token in one step, and the model learns which ones matter. In an RNN, information has to pass step by step along the sequence.

## What "similar" means

Similarity is the dot product: multiply matching numbers and add them up.

```text
q = [1, 2, 0]
k = [3, 1, 5]
q · k = 1×3 + 2×1 + 0×5 = 5
```

Vectors pointing in a similar direction score high, unrelated ones score near 0, and opposite ones score negative. The score is not dictionary similarity. It means "how useful is this token to that one right now", so a verb can score high with its subject.

The scores are divided by `sqrt(D)` (or the head size, with several heads) because dot products tend to grow as the vectors get longer, which would push softmax toward putting almost all the weight on one token.

## Where Q, K and V come from

Each is the token's embedding multiplied by a learned matrix:

```text
q[i] = W_q × x[i]
k[i] = W_k × x[i]
v[i] = W_v × x[i]
```

- `W_q` and `W_k` are separate because relevance is not symmetric: "sat" may need "cat" more than "cat" needs "sat". With one shared vector, `score(i, j)` would always equal `score(j, i)`.
- **V is the content that gets passed along.** Q and K decide how much to look; V decides what is taken. In a library, Q is your question, K is the label on each book's spine, and V is what is inside the book. What makes a good match is not necessarily what is useful to pass on, so V has its own matrix.

## Training versus inference

Training learns the matrices `W_q`, `W_k`, `W_v`, and every other weight, not Q, K and V themselves. Q, K and V are recalculated from the matrices for every input.

| | Training | Inference |
|---|---|---|
| Weights | Changing | Frozen |
| Q, K, V | Calculated, then used to update the weights | Calculated, then used for output only |
| Passes | Forward, then backward | Forward only |
| Goal | Make the model better | Generate tokens fast and cheaply |

- **Pre-training** is the first and largest stage: next-token prediction over a huge amount of text. Weights start random, the loss measures how wrong each prediction is, and backpropagation adjusts all the weights. Useful attention patterns come out of this without anyone writing rules.
- **Post-training** (fine-tuning, RL) adjusts the same weights to follow instructions and chat.
- **Inference** only runs the forward pass with frozen weights. Because the weights never change, K and V for earlier tokens stay valid, which is what makes the KV cache possible.

Other learned weights: `W_o` (combines the heads), the MLP layers that follow attention in each layer (most of the parameters), and the token embeddings. Every layer has its own set of attention matrices.

## Heads

A head is one of several smaller attention calculations run side by side. Each gets its own slice of Q, K and V, so it can look for something different.

```python
H = 4                      # number of heads
d_head = D // H            # 64 / 4 = 16 numbers per head

# (B, T, D) → (B, H, T, d_head)
q = q.view(B, T, H, d_head).transpose(1, 2)
k = k.view(B, T, H, d_head).transpose(1, 2)
v = v.view(B, T, H, d_head).transpose(1, 2)

out = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (B, H, T, d_head)
out = out.transpose(1, 2).reshape(B, T, D)   # join the heads back together
out = W_o(out)                               # learned mix of all heads (W_o = nn.Linear(D, D))
```

The scores now have shape `(B, H, T, T)`, one table per head.

**Who decides what:**

- Designers choose the number of heads and the head size before training. They are hyperparameters, found mostly by experiment, and fixed after that.
- Training decides what each head attends to. Nobody assigns roles.
- In the basic version, `D` must divide evenly by `H`. Some models set the head size separately and use `W_o` to bring the result back to `D`. Common head sizes are 64 and 128.

**What heads learn:** relationships, not just properties. A head finds which tokens matter to this token for one purpose and pulls their information in. Some clear examples found by researchers:

- **Previous-token heads** always look at the token just before.
- **Induction heads** find where the current token appeared earlier and look at what came after it, which lets a model repeat patterns.
- **Syntax heads** connect a verb to its subject or a pronoun to its noun.

"One head, one relationship" is only the clearest case. A head can mix several jobs, one job can be spread across several heads, and many heads have no nameable role.

**Too many heads:** nothing forces every head to be useful. Some heads duplicate each other, some sit idle (putting almost all their weight on a harmless token such as the first one), and some share one job. Michel et al. (2019), "Are Sixteen Heads Really Better than One?", removed many heads from trained models with little loss in accuracy. Removing unneeded heads is called pruning. With `D` fixed, more heads also means each head gets fewer numbers:

```text
D = 1024,  8 heads → 128 numbers per head
D = 1024, 64 heads →  16 numbers per head
```

**Layers build on each other.** Each layer adds its findings to the token's vector rather than replacing it (the residual stream). As a general tendency, not a rule:

```text
Early layers   →  word pieces, positions, nearby words
Middle layers  →  grammar, which words relate to which
Later layers   →  meaning, facts, what the next token should be
```

## Qwen3-0.6B as a real example

From the model's `config.json`:

| Setting | Value |
|---|---|
| `hidden_size` (D) | 1024 |
| `num_hidden_layers` | 28 |
| `num_attention_heads` (query heads) | 16 |
| `num_key_value_heads` | 8 |
| `head_dim` | 128 |
| `max_position_embeddings` | 40,960 |
| `torch_dtype` | bfloat16 |

Two things to notice:

- 16 query heads × 128 = 2048, not 1024. The head size is set separately, and `W_o` maps the result back to D.
- There are only 8 K/V heads for 16 query heads, so pairs of query heads share one K/V head. This is **grouped-query attention (GQA)**, and it halves the KV cache compared with 16 K/V heads.

## Why this matters for inference

- The score table is `T × T`, so the work grows quickly with sequence length. In decode, `T = 1`: only one new row is computed, using cached K and V for the earlier tokens.
- Every K/V head in every layer stores its K and V for every token. KV cache bytes per token are:

  ```text
  layers × kv_heads × head_dim × 2 (K and V) × bytes per number
  ```

  Exercise (Week 4): work this out for Qwen3-0.6B in bfloat16, then predict how many tokens fit in the memory SGLang reserves, and compare with the `max_total_num_tokens` it logs at startup.
- GQA, head pruning and head size are design choices that trade model quality against KV cache size and compute.
