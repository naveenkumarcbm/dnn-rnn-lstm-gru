# Learning guide: recurrent networks and Transformers

This guide explains the ideas behind the two notebooks and how their code fits together. The notebook implementations are embedded in the `.ipynb` files, so they can be uploaded individually to Colab without separate Python modules.

## 1. How to use the notebooks

Each notebook has two modes:

- **Quick-check mode** (`RUN_FULL_EXPERIMENTS = False`) runs correctness tests and parameter-count checks. It does not train the full models, produce measured results, or complete the required figures.
- **Full-experiment mode**: change that setting to `True` in the embedded implementation cell and run all cells. This downloads the requested datasets, performs the experiments, and writes result rows and plots.

Use a Colab T4 GPU for the full experiments. The data, results, and figures are stored in that Colab runtime and can be lost when it disconnects, so download or copy them to Drive before ending the session. Run Problem 1 and Problem 2 in the same project folder/runtime when practical: both use the same root `results.csv`. If the notebooks run in separate Colab runtimes, save both CSVs and combine their data rows without duplicating the header.

## 2. Problem 1: recurrent networks

### 2.1 What a recurrent network computes

A sequence contains ordered observations \(x_1,\ldots,x_T\). A recurrent model carries a hidden state forward:

\[
h_t = f(x_t,h_{t-1}).
\]

The state summarizes useful information from the preceding time steps. For HAR, the sequence is one 128-step sensor window and the final hidden state is used to classify its activity. For the adding task, the final hidden state is used to predict the sum of two marked values.

The notebooks implement the three cell equations directly for verification. The model comparisons use PyTorch's built-in `nn.RNN`, `nn.LSTM`, and `nn.GRU` layers, which are permitted for Problem 1 and avoid an unnecessarily slow Python-level loop in the larger training sweeps.

### 2.2 Vanilla RNN

The custom `MyRNNCell` computes

\[
h_t = \tanh(W_{ih}x_t+b_{ih}+W_{hh}h_{t-1}+b_{hh}).
\]

Its tensors have shapes:

| Tensor | Shape |
|---|---|
| Current input \(x_t\) | `(batch, input_size)` |
| Previous state \(h_{t-1}\) | `(batch, hidden_size)` |
| New state \(h_t\) | `(batch, hidden_size)` |

The nonlinearity is applied at every step. This is simple and parameter-efficient, but repeated multiplication by recurrent weights and repeated `tanh` operations can make distant information or gradients fade or grow.

### 2.3 LSTM

An LSTM keeps both a hidden state \(h_t\) and a cell memory \(c_t\). The affine transform produces four hidden-size vectors, split in the order input, forget, candidate, output:

\[
\begin{aligned}
i_t &= \sigma(a_i), & f_t &= \sigma(a_f),\\
g_t &= \tanh(a_g), & o_t &= \sigma(a_o),\\
c_t &= f_t\odot c_{t-1}+i_t\odot g_t,\\
h_t &= o_t\odot\tanh(c_t).
\end{aligned}
\]

The forget gate controls what memory to retain; the input gate controls how much candidate information is written; and the output gate controls how much cell memory is exposed as the hidden state.

`MyLSTMCell` computes these operations explicitly. The recurrent hidden transform returns `4 * hidden_size` entries; `.chunk(4, dim=-1)` splits those gate vectors. During sequence processing both `h` and `c` have shape `(batch, hidden_size)`.

### 2.4 GRU

The custom GRU follows PyTorch's gate order reset, update, new. In the implementation:

\[
\begin{aligned}
r_t &= \sigma(W_{ir}x_t+b_{ir}+W_{hr}h_{t-1}+b_{hr}),\\
z_t &= \sigma(W_{iz}x_t+b_{iz}+W_{hz}h_{t-1}+b_{hz}),\\
n_t &= \tanh(W_{in}x_t+b_{in}+r_t\odot(W_{hn}h_{t-1}+b_{hn})),\\
h_t &= n_t+z_t\odot(h_{t-1}-n_t).
\end{aligned}
\]

This last equation is equivalent to the common \((1-z_t)n_t+z_th_{t-1}\) form. Reset determines which old state contributes to the candidate; update interpolates between the new candidate and old state.

### 2.5 Why compare custom cells with PyTorch?

The test creates a PyTorch layer and a custom cell, copies the library weights and biases, and unrolls the custom cell over a random input sequence. It asserts the maximum absolute difference is less than \(10^{-5}\). This isolates equation and gate-order errors from training performance. A low test error does not prove that a trained model is accurate; it proves the custom implementation follows the same recurrence for the tested inputs and parameters.

For \(d=9\) inputs and \(h=64\) hidden units, one affine transform per gate gives:

| Cell | Formula with one bias per gate | Hand count | PyTorch count |
|---|---:|---:|---:|
| RNN | \(hd+h^2+h\) | 4,736 | 4,800 |
| LSTM | \(4(hd+h^2+h)\) | 18,944 | 19,200 |
| GRU | \(3(hd+h^2+h)\) | 14,208 | 14,400 |

PyTorch keeps an input-side bias and a recurrent-side bias for every gate, so it has one extra \(h\) parameters per gate. In the custom cells, those two contributions are represented by two `nn.Linear` biases.

### 2.6 HAR data flow

The HAR loader follows the supplied official split:

1. Read the nine inertial-signal arrays and stack channels last, producing `(windows, 128, 9)`.
2. Read labels and convert the official classes from 1–6 to zero-based 0–5.
3. Choose four subjects from the official training subjects using `np.random.default_rng(seed)`. Their windows become validation; all remaining training subjects stay in train. The official test subjects remain untouched.
4. Compute each channel's mean and standard deviation from the training partition only, reducing over window and time axes. Apply those training statistics to train, validation, and test.
5. Wrap arrays in `TensorDataset`/`DataLoader` batches.

Keeping subjects disjoint between train and validation reduces leakage from the same person's movement patterns appearing on both sides. Computing normalization from train only avoids using held-out validation or test information during preprocessing.

`RecurrentClassifier` feeds each `(batch, time, channels)` batch into one recurrent layer and classifies the final output at `outputs[:, -1]`. It uses Adam with learning rate \(10^{-3}\), the same batch size, hidden size, and epoch budget for all three cell types. Accuracy and macro-F1 summarize test performance; macro-F1 gives each activity class equal weight. The confusion matrix gives class-by-class errors. Three seeds expose variability from both subject validation selection and random initialization/shuffle.

### 2.7 Adding-problem data and training

For each example, `make_adding`:

1. Draws values uniformly in \([0,1)\) into channel 0.
2. Draws one marker position in the first half and one in the second half of channel 1.
3. Sums the two values at those marker positions for the target.

Its test asserts two markers per sequence, one marker in each half, and exact agreement between the selected-value sum and the target. Training uses the required fixed data seeds: 0 for 10,000 training rows, 1 for 1,000 validation rows, and 2 for 1,000 test rows. Those generated datasets are reused for model comparisons at a given length, while model seeds control initialization and minibatch shuffle.

`AddingRegressor` returns one scalar from the final recurrent output. `train_adding` trains with mean squared error (MSE), Adam at \(10^{-3}\), and gradient clipping by default. `run_adding_sweep` loops over four lengths, three recurrent model types, and three seeds: 36 model runs. Test MSE is compared to the constant predictor baseline near \(1/6 \simeq 0.167\); the assignment's solved threshold is strictly below 0.05.

### 2.8 Problem 1 gradient and ablation figures

- **Gradient by time step:** a single LSTM adding example batch is unrolled while retaining each hidden state. `torch.autograd.grad` measures the norm of the loss gradient at each hidden state. The horizontal axis is sequence time; the plot illustrates how learning signal propagates backwards through the unroll.
- **Hidden-size sweep:** one LSTM configuration at \(T=200\) is trained at each requested hidden width. The plot relates test MSE to capacity (and parameter count).
- **Clipping ablation:** the same LSTM adding setup, data, seed, length, and epoch budget are run once without clipping and once with norm 1 clipping. This is the chosen interpretation of Task 1.5(c), since the supplied excerpt only specifies an “individual ablation” and does not name its intervention.

When interpreting these plots, report the measured results rather than assuming gates or clipping always improve performance. Discuss what changed, how large the change was, and the compute/parameter cost.

## 3. Problem 2: Transformer from tensor operations

### 3.1 Scaled dot-product attention

Attention lets each query position combine information from key/value positions. For query matrix \(Q\), key matrix \(K\), and value matrix \(V\):

\[
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
\]

The division by \(\sqrt{d_k}\) keeps dot-product magnitudes from growing too large as the key dimension increases. Large magnitudes can make softmax overly concentrated and gradients less useful.

The function accepts `q`, `k`, and `v` shaped `(batch, heads, time, head_dim)`. The score and probability tensors have shape `(batch, heads, query_time, key_time)`; the result has shape `(batch, heads, query_time, head_dim)`. A boolean `allowed_mask` is broadcast over scores; `True` means attention is permitted.

The tests cover three properties:

1. Every unmasked probability row sums to one.
2. The result agrees with `torch.nn.functional.scaled_dot_product_attention` within \(10^{-5}\).
3. A lower-triangular causal mask gives exactly zero probability to future positions.

The hand example uses zero query and key vectors, making all allowed logits equal and the attention uniform across allowed keys. This makes the three output rows straightforward to calculate from the supplied values.

### 3.2 Multi-head attention

The module learns four linear maps: \(W_Q,W_K,W_V,W_O\). Q, K, and V are each projected from `d_model` to `d_model`, split into `n_heads` groups of `head_dim = d_model / n_heads`, and transposed into `(batch, heads, time, head_dim)`. Each head attends independently. The outputs are concatenated back into `(batch, time, d_model)` and passed through \(W_O\).

Multiple heads provide parallel learned attention patterns. The number of heads does not change the total projection dimensions when `d_model` is fixed; it changes how that representation is partitioned.

### 3.3 Pre-LayerNorm encoder block and parameter count

An encoder block applies normalization before each sublayer and adds a residual:

\[
x' = x + \mathrm{Dropout}(\mathrm{MHA}(\mathrm{LN}(x))), \qquad
y = x' + \mathrm{Dropout}(\mathrm{FFN}(\mathrm{LN}(x'))).
\]

The feed-forward network maps `d_model` to `d_ff` and back, with a GELU nonlinearity. The requested default is `d_ff = 4 * d_model`.

The weight-only count is:

\[
4d^2 + 2d\,d_{\mathrm{ff}},
\]

where four \(d\times d\) matrices belong to Q/K/V/output and the two feed-forward matrices have shapes \(d\times d_{\mathrm{ff}}\) and \(d_{\mathrm{ff}}\times d\). At \(d=128,d_{\mathrm{ff}}=512\), this is 196,608 weights. The code separately counts 1,152 linear biases and 512 LayerNorm affine parameters, for 198,272 total trainable parameters in one block. The test compares the derived total with `sum(p.numel() for p in model.parameters())`.

### 3.4 Positional information and the HAR encoder

Self-attention by itself has no inherent notion of sequence order. The classifier therefore has three position settings:

- **none:** only input projection and a learned CLS token are added;
- **sinusoidal:** fixed sine/cosine functions encode positions;
- **learned:** an embedding vector is trained for each position.

Each sensor time step is linearly embedded from 9 channels to `d_model`; a CLS token is prepended. Encoder blocks process the whole sequence, and the transformed CLS output feeds a six-class linear classifier. Position ablations use the same subject split and preprocessing as Problem 1.

### 3.5 Character-level causal language model

The language model predicts the next character from preceding characters:

1. Download Tiny Shakespeare if it is not already present.
2. Build the character vocabulary from the full text and map each character to an integer ID.
3. Use the first 90% of characters for training and the last 10% for validation, preserving order.
4. `CharDataset` produces input windows of length `context` and targets shifted one character forward.
5. The model adds token embeddings to learned position embeddings, passes them through four pre-LayerNorm blocks, and maps each position to vocabulary logits.
6. Cross-entropy compares those logits to the next-character targets.

For a causal model, the lower-triangular mask at each block means position \(t\) can only use positions \(\le t\). This prevents the model from cheating by looking at future characters. The specified main model has context 128, `d_model=128`, four heads, and four layers.

Validation cross-entropy is reported in nats. Convert it to bits per character by dividing by \(\ln 2\). The learning curve stores training loss at evaluation steps and validation loss averaged over up to 16 validation batches; it is a sampled evaluation, not a full pass over every validation window.

Generation crops to the most recent context window, divides the final-position logits by temperature, retains the highest-probability `top_k` characters, samples one, and appends it. Lower temperature makes the distribution sharper and outputs more predictable; higher temperature spreads probability and increases variation. Compare the three actual samples—these are tendencies, not guarantees about coherence.

### 3.6 Transformer analysis experiments

- **Heatmaps:** two heads from the first layer are plotted for a short sentence. A cell at query position \(i\), key position \(j\) indicates the attention weight from \(i\) to \(j\). In a causal model, weights above the diagonal should be zero. Treat heatmaps as learned routing patterns, not proof of semantic causality.
- **Head ablation:** compare 1, 4, and 8 heads while holding model width, depth, context, optimizer, and training steps fixed. The head dimension changes as `d_model / heads`.
- **No-mask comparison:** train the same next-character setup without a causal mask. This allows positions to see future characters, including the next-character targets; its loss is a leakage demonstration, not a valid language-model score.
- **Context benchmark:** measure forward/backward time and peak allocated memory at 64, 128, 256, and 512 tokens. Attention score storage scales as \(O(n^2)\) in context \(n\), multiplied by batch and head count; projection and feed-forward work also depends on \(d\). CUDA memory is measured when CUDA is active. On CPU/MPS, this code reports memory as unavailable (`NaN`), so interpret the memory column accordingly.

## 4. Reading the results and avoiding common mistakes

1. Do not treat the quick-check pass as a model result. It verifies implementation properties; it does not train the marked experiments.
2. Use the recorded seeds and compare mean and standard deviation for the required multi-seed groups. A single best seed is not the whole result.
3. Distinguish validation data (for monitoring/model choices) from the official HAR test set (for final reporting).
4. For the adding task, show the 0.167 reference and the strict 0.05 solved criterion.
5. For the character model, report nats and bits/character consistently; do not score the no-mask leakage model as a causal model.
6. Include the generated figures and refer to them and the result tables in the final PDF. The current report is a draft until the full experiments are run and its pending values are replaced with measured values.
7. Colab disconnects can erase local runtime files. Save `results.csv`, the figures, and `problem2/generated_samples.txt` before ending the session.

## 5. Useful code-flow map

| Notebook area | Main functions/classes | Purpose |
|---|---|---|
| Problem 1 cell checks | `make_adding`, `test_make_adding`, `test_cells_match_torch` | Validate deterministic generated data and custom recurrence equations |
| Problem 1 HAR | `load_har`, `RecurrentClassifier`, `train_har` | Subject-wise classification experiment |
| Problem 1 adding | `AddingRegressor`, `train_adding`, `run_adding_sweep` | Sequence-length and seed experiment |
| Problem 1 analyses | `plot_gradient_norm`, `run_hidden_size_sweep`, `run_clipping_ablation` | Gradient, capacity, and intervention evidence |
| Problem 2 attention | `scaled_dot_product_attention`, `MultiHeadAttention` | Handwritten attention and head decomposition |
| Problem 2 block | `EncoderBlock`, `SinusoidalPositionalEncoding`, `encoder_block_parameter_count` | Encoder architecture, positions, and count verification |
| Problem 2 HAR | `HARTransformerClassifier`, `train_har_transformer` | Positional encoding classifier comparison |
| Problem 2 language model | `CharDataset`, `CausalTransformerLM`, `train_language_model`, `generate` | Next-character training and sampling |
| Problem 2 analyses | `plot_attention_heatmaps`, `run_heads_ablation`, `run_context_benchmark` | Attention interpretation and architecture/context studies |

