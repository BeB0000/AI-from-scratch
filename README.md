# beb0 (i called it this :D)

A tiny GPT-style transformer language model written from scratch in plain C.
No frameworks, no BLAS, no python. Just `malloc`, loops and a lot of patience.

It can **train** on a pre-tokenized corpus and **chat** with you in the terminal,
all from a single source file (`beb0.c`).

Created by **(@BeB0000)**, MIT licensed.

---

## features

- decoder-only transformer (token + position embeddings, multi-head causal self-attention, GELU feed-forward, layer norm, residual connections)
- hand written forward **and** backward passes for every layer (no autograd)
- Adam optimizer with bias correction
- gradient clipping by global norm
- gradient accumulation (small physical batch, bigger effective batch)
- 2 thread training (the batch is split in half, one half per thread)
- BPE style vocab loader with a greedy longest-match encoder
- corpus loaded with `mmap`, so big token files do not need to fit in ram
- atomic checkpoint saving (writes `.tmp` then renames), weights + adam state + step count
- resume training from the last checkpoint automatically
- sampling with temperature, top-k and a repetition penalty
- cpu thermal protection: reads the cpu temperature and slows training down if it gets too hot
- interactive chat mode with a few commands
- clean ctrl+c handling (saves a checkpoint before exiting)

---

## project layout

```
.
├── src/
│   └── beb0.c            # the whole model, trainer and chat loop
├── include/
│   └── beb0.h            # all the hyperparameters and constants
├── data/
│   └── vocab.txt         # tokenizer vocab (you create this)
├── checkpoints/
│   └── checkpoint.bin    # created automatically when training
└── Makefile
```

`beb0.c` includes `../include/beb0.h`, so keep that folder structure.

### what lives in `beb0.h`

The header holds the constants the code relies on. Tweak these to change the model size or training behaviour:

| constant | what it does |
| --- | --- |
| `VOCAB_SIZE` | max number of tokens in the vocab |
| `HIDDEN_DIM` | embedding / hidden size |
| `NUM_HEADS`, `HEAD_DIM` | attention heads and size of each head |
| `FF_DIM` | size of the feed-forward layer |
| `NUM_LAYERS` | number of transformer blocks |
| `MAX_SEQ_LEN` | context length |
| `PHYSICAL_BATCH`, `GRAD_ACCUM_STEPS` | batch size per step and accumulation steps |
| `NUM_STEPS`, `LOG_EVERY`, `SAVE_EVERY` | training length, log and checkpoint frequency |
| `LEARNING_RATE`, `MIN_LEARNING_RATE`, `FIRE_LEARNING_RATE` | learning rate values |
| `ADAM_BETA1`, `ADAM_BETA2`, `ADAM_EPS` | adam settings |
| `CLIP_GRAD_NORM` | gradient clipping threshold |
| `LN_EPS`, `GELU_C`, `GELU_SQ` | layer norm epsilon and gelu constants |
| `TOKEN_PAD`, `TOKEN_UNK`, `TOKEN_EOS` | special token ids |
| `TOP_K`, `REP_PENALTY_WINDOW`, `REP_PENALTY_FACTOR` | sampling settings |
| `THERMAL_PATH`, `THERMAL_WARN_C`, `THERMAL_CRIT_C`, `THERMAL_SLEEP_US` | thermal protection |

It also needs to pull in the standard headers the code uses (`stdio`, `stdlib`, `string`, `math`, `signal`, `pthread`, `fcntl`, `sys/mman`, `sys/stat`, `unistd`, `time`, `stdint`).

---

## building

Needs a POSIX system (linux) and a C compiler.

```sh
gcc -O3 -march=native -o beb0 src/beb0.c -lm -lpthread
```

`-O3` matters a lot here, the matmuls are plain loops.

---

## usage

### 1. make a vocab

Create `data/vocab.txt`, one token per line:

```
<piece> <score>
```

- the line number (starting at 0) is the token id
- the score is optional
- tokens that look like `<something>` are marked as special
- sentencepiece style word-start pieces (prefixed with `▁`) are supported, the chat encoder tries both the plain piece and the `▁piece` version

### 2. make a corpus

The corpus is a raw binary file of **`int32` token ids**, one after another, already tokenized with the same vocab. The trainer picks random windows of `MAX_SEQ_LEN` tokens from it.

### 3. train

```sh
./beb0 path/to/corpus.bin
```

- if `checkpoints/checkpoint.bin` exists, training resumes from it
- a checkpoint is saved every `SAVE_EVERY` steps and when the run ends
- press `ctrl+c` any time, it will save and exit cleanly
- the learning rate drops to `MIN_LEARNING_RATE` after step 50000

### 4. chat

```sh
./beb0
```

Loads the checkpoint (or a fresh random model if there is none) and drops you in a chat prompt.

| command | what it does |
| --- | --- |
| `/help` | show the help menu |
| `/joke` | tell a terrible joke |
| `/clear` | clear the screen |
| `/restart` | restart the program (clears the chat history) |
| `/quit` or `/exit` | leave |

---

## how it works

1. **tokenize**: text is turned into ids with a greedy longest-match search over the vocab
2. **embed**: token embedding + learned position embedding
3. **transformer blocks** (x `NUM_LAYERS`): layer norm, causal multi-head attention, residual, layer norm, GELU feed-forward, residual
4. **head**: final layer norm and a linear layer to vocab logits
5. **loss**: cross entropy against the next token
6. **backward**: gradients are computed by hand through every step above
7. **update**: gradient clipping then Adam

### checkpoint format

A raw binary file: the step count (`int`), then for every parameter tensor in a fixed order the weights, the adam `m` and the adam `v`. The file is only valid for the exact same hyperparameters it was trained with, so if you change anything in `beb0.h` start with a fresh checkpoint.

### thermal protection

Every `LOG_EVERY` steps the cpu temperature is read from `THERMAL_PATH` (usually `/sys/class/thermal/thermal_zone0/temp`). Above the warning temperature training sleeps a bit, above the critical temperature it sleeps longer and halves the learning rate. Made for an old dual core machine that was not happy about this.

---

## memory

The startup message prints the parameter count and a rough ram estimate (weights + gradients + adam m + adam v). Make sure you have enough before raising `HIDDEN_DIM`, `NUM_LAYERS` or `VOCAB_SIZE`.

---

## license

MIT. See the header of `beb0.c`.
