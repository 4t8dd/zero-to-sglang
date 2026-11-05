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

Some texts write the same shape as `[batchSize, seqLen, d_model]`.

### What X actually looks like

A tiny example with `batchSize = 2`, `seqLen = 3`, `d_model = 4`. Text becomes token IDs (invented here), and each ID picks one row from the embedding table, a learned matrix of shape `[vocabulary size, d_model]`:

```text
"The cat sat"  → [ 791, 8415, 7731]
"I like tea"   → [  40, 1093, 9725]

X = [
  [                                  # sequence 1: "The cat sat"
    [ 0.12, -0.40,  0.88,  0.05],    # "The"
    [-0.73,  0.31,  0.10,  0.66],    # "cat"
    [ 0.27,  0.94, -0.52, -0.18],    # "sat"
  ],
  [                                  # sequence 2: "I like tea"
    [ 0.45, -0.09, -0.33,  0.71],    # "I"
    [-0.16,  0.58,  0.82, -0.47],    # "like"
    [ 0.90, -0.25,  0.14,  0.39],    # "tea"
  ],
]
shape: [2, 3, 4]
```

```python
embedding = nn.Embedding(vocab_size, d_model)   # the learned table
ids = torch.tensor([[791, 8415, 7731],
                    [ 40, 1093, 9725]])          # [batchSize, seqLen]
X = embedding(ids)                               # [batchSize, seqLen, d_model]
```

- `X[0]` is the first sentence, `[3, 4]`; `X[0][1]` is "cat", 4 numbers; `X[0][1][2]` is one number.
- The same token always starts with the same row, wherever it appears.
- In Qwen3-0.6B each token is 1024 numbers, so a 50-token prompt gives `[1, 50, 1024]`. In later layers, `X` is the previous layer's output: same shape, different numbers.
- The embedding does not record word order. Some models add position information to `X`; Qwen3 uses RoPE, which rotates Q and K by each token's position, so `X` itself has no position numbers.

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

## Heads and knowledge

Heads are not where knowledge is stored, and adding heads is not how a model gets better at a field.

- **Heads decide how information moves.** Each one finds which tokens in the input matter to the current token and pulls their information in. Heads limit how many relationships the model can track at once within the text it is reading, not how much it knows.
- **Knowledge is stored mostly in the MLP layers**, which also hold most of the parameters. Geva et al. (2021), "Transformer Feed-Forward Layers Are Key-Value Memories", and Meng et al. (2022, ROME), which changed specific facts by editing specific MLP weights, support this.

Attention brings in the relevant context; the MLP layers bring in what the model already knows.

For a knowledge-heavy field such as medicine or law:

| Approach | Why it helps |
|---|---|
| More and better training data in the field | Matters most: a model cannot learn what it never saw |
| Fine-tuning on the field | Adjusts existing weights without retraining from scratch |
| A larger model overall | More parameters, especially in the MLP, give more room to store knowledge |
| Retrieval (RAG) | Puts relevant documents in the prompt, so attention can read them at inference time |

The head count is fixed when the model is designed and is the same for every topic. Heads do matter for inputs full of long-range connections, such as a contract that refers back to earlier sections or code that calls functions defined far away. Even there, the field's knowledge comes from training.

## Is the usual head count too small for dense data?

The idea: head counts are chosen by experiment, experiments reflect the average case, and the average case has few relationships to compute. Data with many more connections might need more heads.

It is partly right: head counts are compromises tuned on the average of the training data, and no one has shown the usual numbers are best for every kind of data. But the evidence suggests the limit is smaller than it seems:

1. **The training mix includes dense data.** The experiments that set these numbers train on code, math and long documents, which are among the most relationship-heavy data there is.
2. **Relationships are not limited to one per head.**
   - Heads are reused: the same head can do different jobs depending on the input, because Q and K are calculated from the input each time.
   - Layers chain relationships: one layer connects A to B, a later layer connects that result to C. Multi-step reasoning mostly comes from depth, not head count.
   - Each head has width: a 128-number head can encode a fairly rich match.

   Capacity is roughly heads × layers × head size, plus the MLP, not the head count alone.
3. **Head count matters less than expected in experiments.** Kaplan et al. (2020), "Scaling Laws for Neural Language Models", found performance depends mainly on total parameters, data and compute, and only weakly on model shape (heads, depth, width) across a wide range. GQA points the same way: models share K/V heads and lose little quality.

For unusual data with very many connections at once, a different shape might work better. That is still an open question.

**Experiment (Week 2, with nanoGPT on a GPU):**

1. Pick a relationship-heavy dataset, such as code.
2. Train small models with the same total size but different head counts, for example 2, 8 and 32 heads.
3. Predict which one wins, then compare their validation loss.

## What kind of data is rich in connections?

It depends less on the kind of data than on the task: whether predicting each piece requires combining pieces that are far apart.

| Data | Why it is dense |
|---|---|
| Code | A line can depend on a variable, function or type defined thousands of lines away |
| Math proofs | Each step relies on definitions and earlier results from anywhere before it |
| Legal contracts | Clauses refer to definitions and other sections throughout the document |
| Proteins | Amino acids far apart in the chain can touch once the protein folds; AlphaFold relies heavily on attention for this |
| DNA | Regions controlling a gene can be very far from the gene itself |
| Tables and spreadsheets | Each cell relates to its row, its column and the formulas that use it |
| Long conversations | Who said what, and who is replying to whom, spans the whole discussion |

**Telescope data is mixed.**

- Mostly local: in an image, nearby pixels matter most. Convolutional networks (CNNs) have long handled this well, because they look at small neighbourhoods.
- Sometimes long-range:
  - Gravitational lensing: distorted copies of one galaxy can appear in different parts of an image.
  - Light curves: a pattern repeating over a long series of measurements can reveal a planet or a variable star.
  - Survey matching: the same object has to be linked across images, spectra and catalogues.

Transformers are being used in astronomy for these kinds of tasks.

**The catch is size.** One image can have millions of pixels, and the score table is `T × T`, so one token per pixel is far too expensive. Vision models group pixels into patches, often 16 × 16 pixels per token, which brings `T` down to a manageable size.

## The MLP layer

MLP stands for **Multi-Layer Perceptron**: a perceptron is an early, very simple model of a neuron, and stacking layers of them gives an MLP. In Transformers it is also called the **feed-forward network (FFN)**, the name used in the original paper.

The MLP is the second half of every Transformer layer. Attention moves information between tokens; the MLP processes each token on its own.

```text
One layer:
  x = x + Attention(x)   # tokens exchange information
  x = x + MLP(x)         # each token works on what it has gathered
```

The same small network runs on every token's vector separately and never looks at other tokens:

```python
up   = nn.Linear(D, 4 * D)   # expand: 64 → 256
down = nn.Linear(4 * D, D)   # shrink back: 256 → 64

def mlp(x):                      # x: (B, T, D)
    return down(F.gelu(up(x)))   # (B, T, D), each token separately
```

1. **Expand:** more room to work in, usually about 4 times wider.
2. **Activation:** a non-linear function such as GELU or SiLU. Without it, the two layers would collapse into one plain matrix multiplication.
3. **Shrink:** back to `D`, then added to the residual stream.

**Separately, but all at once.** Every token goes through the MLP independently with the same weights, but not one after another. PyTorch treats the `B × T` token vectors as rows of one matrix and multiplies them by the weights in a single operation. The tokens do not affect each other, so each gets the same result as if it ran alone.

This is why batching works so well in decode. Each request adds one token per step, and the tokens from all running requests are stacked into one matrix:

```text
1 request:    read all MLP weights → process 1 token
32 requests:  read all MLP weights → process 32 tokens
```

Reading the weights from GPU memory is the slow part, and it happens once for the whole batch. Attention cannot be stacked as simply, because each request has its own KV cache of a different length, which is why engines use special kernels and memory layouts for it.

What it does:

- **Stores knowledge.** The expand layer acts like a set of detectors that fire on patterns such as "capital of France"; the shrink layer adds a matching piece of information such as "Paris" (the key-value memory view, Geva et al. 2021).
- **Processes what attention gathered.** After attention pulls in "cat" for "sat", the MLP works out what that means for this token.

Qwen3-0.6B uses a gated version, SwiGLU, with three matrices:

```python
def mlp(x):
    return down(F.silu(gate(x)) * up(x))   # 1024 → 3072 → 1024
```

Parameters per layer, from its config (`hidden_size` 1024, `intermediate_size` 3072, 16 query heads and 8 K/V heads of 128):

| Part | Parameters |
|---|---|
| Attention: Q, K, V, O | about 6.3 million |
| MLP: gate, up, down | about 9.4 million |

The MLP holds about 60% of each layer's parameters.

Why it matters for inference:

- **No KV cache.** The MLP does not read other tokens, so it has nothing to cache.
- **Weight reading dominates decode.** Every decode step reads all the MLP weights from GPU memory to process one token per request. That is a big part of why decode is memory-bound, and why batching helps: the weights are read once for the whole batch.
- **Mixture of Experts (MoE)** replaces the single MLP with many expert MLPs and sends each token to only a few of them, so the model stores more knowledge while each token uses a small part of the weights. See Part 3, expert parallelism.

### Does the MLP produce each layer's output?

Partly. It writes the last update in each layer, but:

1. **It adds to the output rather than producing it.** A layer's output is a running total, `x_out = x_in + Attention(...) + MLP(...)`. What earlier layers wrote stays in the vector unless later layers change it. This residual stream is what the next layer receives.
2. **It only sees one token.** That token's vector already includes what attention pulled in from other tokens in this layer, plus what all earlier layers added. The MLP never reaches into other tokens; only attention moves information between tokens. Attention collects; the MLP processes what was collected.
3. **The prediction comes after all the layers.** After the last layer: normalize, then the LM head (a `D × vocabulary` matrix) gives a score for every token in the vocabulary. In decode, only the last token's vector is used. Every attention and MLP step across all layers adds a piece; no single place makes the answer.

The full order inside one layer, with the normalization (RMSNorm in Qwen3) that keeps the numbers at a stable size:

```text
x = x + Attention(Norm(x))
x = x + MLP(Norm(x))
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
