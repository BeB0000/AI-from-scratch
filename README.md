# beb0 (I called it this :D)

`beb0` is a small GPT-style transformer language model written from scratch in C. It implements the model, training loop, optimizer, tokenizer lookup, checkpointing, sampling, and terminal chat runtime without a machine-learning framework or BLAS dependency.

The project also includes Python utilities for collecting a text corpus, cleaning it, training a byte-level BPE tokenizer, and converting the corpus into the binary format expected by the C trainer.

Created by **@BeB0000**. Licensed under the MIT License.

## Features

- Decoder-only transformer with token and position embeddings.

- Multi-head causal self-attention, GELU feed-forward layers, layer normalization, and residual connections.

- Hand-written forward and backward passes; no automatic differentiation.

- Adam optimizer with bias correction, gradient clipping, and gradient accumulation.

- POSIX-threaded batch processing when the physical batch contains more than one sample.

- Greedy longest-match tokenizer lookup with optional SentencePiece-style `▁` word-start pieces.

- Memory-mapped binary training corpus, so the complete token file does not need to be copied into RAM.

- Atomic checkpoints containing model weights, Adam state, and the current training step.

- Automatic checkpoint resume.

- Temperature sampling, top-k sampling, and repetition control.

- CPU thermal protection on systems exposing `/sys/class/thermal/thermal_zone0/temp`.

- Interactive terminal chat with `/help`, `/joke`, `/clear`, `/restart`, `/quit`, and `/exit` commands.

- SIGINT/SIGTERM handling that saves a checkpoint before training exits.

## Repository layout

```
.
├── include/
│   └── beb0.h                 # Model and training configuration
├── src/
│   └── beb0.c                 # Model, trainer, tokenizer loader, and chat loop
├── python/
│   ├── prepare_corpus.py      # Download Gutenberg and Wikipedia text
│   ├── clean_corpus.py        # Remove metadata and unwanted text
│   ├── build_tokenizer.py     # Train the BPE tokenizer and write token files
│   └── cleaner.py             # Remove a selected character from a text file
├── data/                      # Generated locally; not included by default
│   ├── corpus.txt
│   ├── corpus_clean.txt
│   ├── corpus.bin
│   └── vocab.txt
├── checkpoints/               # Generated locally
│   └── checkpoint.bin
├── Makefile
└── README.md
```

Run commands from the repository root. The executable expects `data/vocab.txt` and `checkpoints/checkpoint.bin` relative to the current working directory.

## Requirements

### C runtime

- Linux or another POSIX-like operating system.

- GCC or a compatible C compiler.

- POSIX threads, `libm`, `mmap`, and standard POSIX file APIs.

### Corpus preparation tools

The Python pipeline requires Python 3 and these packages:

```
python3 -m pip install requests tqdm tokenizers
```

The C program itself does not require Python at runtime after `data/vocab.txt` and a binary corpus have been created.

## Build

The Makefile is the recommended build path:

```
make
```

This produces `./beb0` with `-O3`, native CPU optimizations, warnings enabled, pthread support, and the math library. The equivalent direct command is:

```
gcc -O3 -march=native -Wall -Wextra -Iinclude \\
    src/beb0.c -o beb0 -pthread -lm
```

Useful Make targets:

| Target | Description |
| --- | --- |
| `make` | Build the executable. |
| `make run` | Build and start chat mode. |
| `make train` | Build and train on `data/corpus.bin`. |
| `make clean` | Remove object files. |
| `make clear` | Remove object files and the executable. |

## Prepare a training corpus

The included scripts form a sequential pipeline.

### 1. Collect raw text

```
python3 python/prepare_corpus.py
```

This creates `data/corpus.txt` from Project Gutenberg books and Wikipedia articles. The script performs network requests and may take a while because it deliberately pauses between requests.

### 2. Clean the corpus

```
python3 python/clean_corpus.py
```

This reads `data/corpus.txt`, removes source markers, Gutenberg headers and footers, illustration blocks, citation markers, and excess blank lines, then writes `data/corpus_clean.txt`.

### 3. Train the tokenizer and encode the corpus

```
python3 python/build_tokenizer.py
```

This trains a byte-level BPE tokenizer with a target vocabulary size of 16,000 and writes:

- `data/vocab.txt`: one token per line, ordered by token ID.

- `data/corpus.bin`: little-endian signed `int32` token IDs.

The C configuration uses the same default vocabulary size through `VOCAB_SIZE` in `include/beb0.h`. If the tokenizer vocabulary size or special-token ordering changes, update the C configuration and start from a fresh checkpoint.

### Optional character cleaner

`cleaner.py` removes every occurrence of one string from a file in place:

```
python3 python/cleaner.py path/to/file.txt character
```

Use it carefully because it modifies the input file directly.

## Vocabulary and corpus formats

`data/vocab.txt` is read one line at a time. The line number, starting at zero, is the token ID. A line may contain a token by itself or a token followed by an optional score. Tokens enclosed in angle brackets are treated as special tokens by the loader. The default special-token IDs are:

| Token | ID |
| --- | --- |
| `<unk>` | 0 |
| `<bos>` | 1 |
| `<eos>` | 2 |
| `<pad>` | 3 |

The training corpus is a raw sequence of little-endian `int32` token IDs. It must use the same vocabulary and special-token ordering as `data/vocab.txt`. The trainer samples random input/target windows of `MAX_SEQ_LEN` tokens from this file.

## Train

After building the executable and generating `data/corpus.bin`, start training with either command:

```
./beb0 data/corpus.bin
```

or:

```
make train
```

Training behavior:

- A new model is initialized if `checkpoints/checkpoint.bin` does not exist.

- An existing checkpoint is loaded automatically and training resumes from its saved step.

- Checkpoints are written every `SAVE_EVERY` steps and again when training finishes.

- Press `Ctrl+C` to save the current model and exit cleanly.

- The default configuration trains for 50,000 steps with a physical batch size of 1 and four gradient-accumulation steps.

- The learning rate is reduced to `MIN_LEARNING_RATE` after step 50,000; thermal protection can reduce it further when the CPU is too hot.

The checkpoint directory must exist before the first save. Create it with:

```
mkdir -p checkpoints
```

The trainer also expects `data/vocab.txt` to exist, even when the corpus itself has already been prepared.

## Chat

Start interactive chat mode without a corpus argument:

```
./beb0
```

or:

```
make run
```

Chat loads `checkpoints/checkpoint.bin` when available. If no checkpoint exists, it starts with a randomly initialized model, which will not produce useful language until trained.

| Command | Action |
| --- | --- |
| `/help` | Show the command list. |
| `/joke` | Print a built-in joke. |
| `/clear` | Clear the terminal screen. |
| `/restart` | Restart the process and clear conversation history. |
| `/quit`, `/exit` | Leave chat mode. |

Generated responses use the configured `TOP_K`, temperature, and repetition-penalty settings in the source. The current chat loop generates up to 200 tokens per response and keeps a bounded conversation history.

## Configuration

Edit `include/beb0.h` and rebuild after changing configuration. Important settings include:

| Constant | Purpose |
| --- | --- |
| `VOCAB_SIZE` | Maximum vocabulary size and output-logit count. |
| `HIDDEN_DIM` | Embedding and transformer hidden width. |
| `NUM_HEADS`, `HEAD_DIM` | Number and width of attention heads. `HEAD_DIM` is derived from `HIDDEN_DIM / NUM_HEADS`. |
| `NUM_LAYERS` | Number of transformer blocks. |
| `FF_MULTIPLIER`, `FF_DIM` | Feed-forward width. `FF_DIM` is derived from `FF_MULTIPLIER * HIDDEN_DIM`. |
| `MAX_SEQ_LEN` | Context length used for training and generation. |
| `PHYSICAL_BATCH`, `GRAD_ACCUM_STEPS` | Physical batch size and gradient-accumulation count. |
| `NUM_STEPS`, `LOG_EVERY`, `SAVE_EVERY` | Training length and reporting/checkpoint intervals. |
| `LEARNING_RATE`, `MIN_LEARNING_RATE`, `FIRE_LEARNING_RATE` | Learning-rate configuration. |
| `ADAM_BETA1`, `ADAM_BETA2`, `ADAM_EPS` | Adam optimizer parameters. |
| `CLIP_GRAD_NORM` | Global gradient-norm limit. |
| `TOKEN_UNK`, `TOKEN_BOS`, `TOKEN_EOS`, `TOKEN_PAD` | Special-token IDs. |
| `TOP_K`, `REP_PENALTY_WINDOW`, `REP_PENALTY_FACTOR` | Generation sampling controls. |
| `THERMAL_PATH`, `THERMAL_WARN_C`, `THERMAL_CRIT_C`, `THERMAL_SLEEP_US` | CPU temperature protection. |

Changing tensor shapes, vocabulary size, token IDs, or other model dimensions invalidates existing checkpoints. Remove the old checkpoint before retraining:

```
rm -f checkpoints/checkpoint.bin checkpoints/checkpoint.bin.tmp
```

## Model overview

For each input sequence, `beb0`:

1. Encodes token IDs using token and learned position embeddings.

1. Applies `NUM_LAYERS` transformer blocks, each with pre-layer normalization, causal multi-head self-attention, a residual connection, GELU feed-forward layers, and a second residual connection.

1. Applies a final layer normalization and a linear language-model head.

1. Computes next-token cross-entropy loss.

1. Computes gradients manually through the complete network.

1. Clips gradients and updates parameters with Adam.

The implementation is intentionally simple and CPU-oriented. Matrix multiplications are plain loops, so compiler optimization and realistic model sizes matter for training speed.

## Checkpoints and memory

Checkpoints are raw binary files. They contain the training step followed by every parameter tensor in a fixed order, including its weights and Adam `m` and `v` buffers. They are not portable across incompatible hyperparameter or source-layout changes.

At startup, the program prints an approximate memory requirement based on parameters, gradients, and optimizer state. Attention scratch buffers and activations require additional memory, especially as `MAX_SEQ_LEN`, `HIDDEN_DIM`, `NUM_LAYERS`, or `VOCAB_SIZE` increase.

## Thermal protection

On systems with the configured thermal file, the program checks the CPU temperature every `LOG_EVERY` steps. Above `THERMAL_WARN_C`, it pauses briefly; above `THERMAL_CRIT_C`, it pauses longer and reduces the active learning rate. If the thermal file is unavailable, the temperature is reported as unavailable and training continues.

If your system exposes temperature data at another path, update `THERMAL_PATH` in `include/beb0.h` before rebuilding.

## Troubleshooting

### `cannot open data/vocab.txt`

Run the tokenizer step and make sure you are launching `beb0` from the repository root:

```
python3 python/build_tokenizer.py
./beb0
```

### `failed to load corpus`

Confirm that the corpus is a binary `int32` token file, that it is larger than at least one training window, and that the path passed to `beb0` is correct.

### The model produces poor or random responses

A fresh model is random. Train it on a sufficiently large, clean corpus, verify that the tokenizer and C special-token IDs agree, and ensure that the checkpoint was created with the current `beb0.h` settings.

### Checkpoint loading fails after a configuration change

Delete the old checkpoint and train again with the new configuration:

```
rm -f checkpoints/checkpoint.bin checkpoints/checkpoint.bin.tmp
./beb0 data/corpus.bin
```

## License

MIT License. See the source header in `src/beb0.c` for the project notice.
:wq
