#!/usr/bin/env python3

"""
clean_corpus.py – aggressively remove all metadata, headers, footers, and junk
from the raw corpus, leaving only clean text for training.
"""

import re
import sys
from pathlib import Path

INPUT_FILE = "data/corpus.txt"
OUTPUT_FILE = "data/corpus_clean.txt"

def remove_source_lines(text):
	"""Remove lines that start with '=== SOURCE:' (the document separators)."""
	lines = text.split('\n')
	lines = [line for line in lines if not line.startswith('=== SOURCE:')]
	return '\n'.join(lines)

def strip_gutenberg_headers_footers(text):
	"""
	Remove everything before the '*** START OF' line and after the '*** END OF' line.
	Also remove common standalone Gutenberg lines like 'EBOOK', 'Produced by', etc.
	"""

	start_match = re.search(r'\*\*\* START OF (?:THE |THIS )?PROJECT GUTENBERG.*?\*\*\*', text, re.IGNORECASE)
	end_match   = re.search(r'\*\*\* END OF (?:THE |THIS )?PROJECT GUTENBERG.*?\*\*\*', text, re.IGNORECASE)

	if start_match and end_match:
		text = text[start_match.end():end_match.start()]
	elif start_match:

		text = text[start_match.end():]
	elif end_match:

		text = text[:end_match.start()]

	lines = text.split('\n')
	cleaned_lines = []
	skip_patterns = [
		r'^EBOOK.*\*\*\*',
		r'^Produced by',
		r'^Transcriber’s Note',
		r'^\[Illustration',
		r'^ +RUSKIN HOUSE',
		r'^CHISWICK PRESS',
		r'^TOOKS COURT',
		r'^ +LONDON',
		r'^ +George Allen',
		r'^PAGE$',
		r'^Frontispiece',
		r'^Title-page',
		r'^Dedication',
		r'^Heading to',
		r'^“.*”\s+\d+$',
		r'^\d+$',
		r'^FINIS$',
		r'^THE END$',
	]
	for line in lines:
		if any(re.match(p, line) for p in skip_patterns):
			continue
		cleaned_lines.append(line)
	return '\n'.join(cleaned_lines)

def remove_illustration_blocks(text):
	"""Remove anything that looks like [Illustration: ... ]."""
	return re.sub(r'\[Illustration:.*?\]', '', text, flags=re.DOTALL)

def remove_citation_markers(text):
	"""Remove [1], [2] etc. (optional)."""
	return re.sub(r'\[\d+\]', '', text)

def collapse_blank_lines(text):
	"""Replace three or more consecutive newlines with two."""
	return re.sub(r'\n{3,}', '\n\n', text)

def main():
	if not Path(INPUT_FILE).exists():
		print(f"❌ {INPUT_FILE} not found. Run prepare_corpus.py first.")
		sys.exit(1)

	with open(INPUT_FILE, "r", encoding="utf-8") as f:
		text = f.read()

	text = remove_source_lines(text)
	text = strip_gutenberg_headers_footers(text)
	text = remove_illustration_blocks(text)
	text = remove_citation_markers(text)
	text = collapse_blank_lines(text)

	with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
		f.write(text)

	print(f"✅ Cleaned corpus written to {OUTPUT_FILE}")
	print("   Metadata, headers, footers, and illustration lists removed.")

if __name__ == "__main__":
	main()
