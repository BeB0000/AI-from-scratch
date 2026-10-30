#!/usr/bin/env python3

"""
build_tokenizer.py

Trains a BPE tokenizer (vocab size 16000) on the cleaned corpus.
Produces:
- data/vocab.txt       (list of tokens, one per line)
- data/corpus.bin      (token IDs as int32, little‑endian)
"""

import struct
import sys
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

CORPUS_TXT = DATA_DIR / "corpus_clean.txt"
VOCAB_TXT = DATA_DIR / "vocab.txt"
CORPUS_BIN = DATA_DIR / "corpus.bin"

VOCAB_SIZE = 16000
SPECIAL_TOKENS = ["<unk>", "<bos>", "<eos>", "<pad>"]

def main():

	DATA_DIR.mkdir(parents=True, exist_ok=True)

	if not CORPUS_TXT.exists():
		print(f"❌ {CORPUS_TXT} not found. Run clean_corpus.py first.")
		sys.exit(1)

	print("🚀 Training BPE tokenizer...")
	tokenizer = ByteLevelBPETokenizer()
	tokenizer.train(
		files=[str(CORPUS_TXT)],
		vocab_size=VOCAB_SIZE,
		min_frequency=2,
		special_tokens=SPECIAL_TOKENS,
	)

	tokenizer.save_model(str(DATA_DIR), "bpe")

	print("📝 Writing vocab.txt...")
	vocab = tokenizer.get_vocab()

	tokens_by_id = [None] * len(vocab)
	for token, idx in vocab.items():
		tokens_by_id[idx] = token

	with open(VOCAB_TXT, "w", encoding="utf-8") as f:
		for token in tokens_by_id:

			f.write(f"{token}\n")

	print("🔢 Encoding corpus to binary...")
	with open(CORPUS_TXT, "r", encoding="utf-8") as f:
		text = f.read()

	encoding = tokenizer.encode(text)
	ids = encoding.ids

	with open(CORPUS_BIN, "wb") as f:
		for token_id in ids:
			f.write(struct.pack("<i", token_id))

	print(f"✅ Done! Wrote {len(ids)} tokens to {CORPUS_BIN}")
	print(f"   Vocab size: {len(vocab)} (should be {VOCAB_SIZE})")
	print(f"   Vocab saved to {VOCAB_TXT}")

if __name__ == "__main__":
	main()
