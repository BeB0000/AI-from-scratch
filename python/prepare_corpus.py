#!/usr/bin/env python3

"""
prepare_corpus.py

Downloads text from:
- Project Gutenberg (popular books)
- Wikipedia (random articles)

Output: data/corpus.txt (one document per block, separated by blank lines)
"""

import os
import re
import time
import random
import requests
from pathlib import Path
from tqdm import tqdm

OUTPUT_FILE = "data/corpus.txt"
MIN_TEXT_LENGTH = 500
REQUEST_DELAY = 0.5

GUTENBERG_IDS = [
	1342, 11, 1661, 98, 76, 1952, 84, 2701, 345, 1400,
	174, 16328, 2542, 4300, 5200, 41, 1260, 46, 1080, 829,
	35, 36, 768, 244, 1727, 2600, 2554, 1232, 3207, 4363,
	7370, 5827, 1251, 45, 514, 55, 2814, 1184, 135, 2148,
	103, 2500, 1399, 600, 996, 3160, 1257, 2097, 161, 158
]
GUTENBERG_MIRROR = "https://www.gutenberg.org/files/{id}/{id}-0.txt"
GUTENBERG_FALLBACK = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"

def fetch_gutenberg(book_id, session):
	for url_tmpl in (GUTENBERG_MIRROR, GUTENBERG_FALLBACK):
		url = url_tmpl.format(id=book_id)
		try:
			r = session.get(url, timeout=15)
			if r.status_code == 200 and len(r.text) > 1000:
				text = r.text

				title_match = re.search(r'Title:\s*(.+)', text)
				title = title_match.group(1).strip() if title_match else f"Book #{book_id}"

				start = re.search(r'\*\*\* START OF (THE|THIS) PROJECT GUTENBERG', text)
				end   = re.search(r'\*\*\* END OF (THE|THIS) PROJECT GUTENBERG', text)
				if start:
					text = text[start.end():]
				if end:
					text = text[:end.start()]
				return title, text
		except Exception:
			pass
	return None

WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_SEEDS = [
	"Artificial intelligence", "Machine learning", "Computer science",
	"Mathematics", "Physics", "Chemistry", "Biology", "History",
	"Philosophy", "Literature", "Economics", "Psychology", "Linguistics",
	"Astronomy", "Programming language", "Operating system", "Internet"
]

def fetch_wiki_article(title, session):
	params = {
		"action": "query",
		"titles": title,
		"prop": "extracts",
		"explaintext": True,
		"exsectionformat": "plain",
		"format": "json",
		"redirects": 1,
	}
	try:
		r = session.get(WIKI_API, params=params, timeout=15)
		data = r.json()
		pages = data["query"]["pages"]
		page = next(iter(pages.values()))
		if "extract" in page and len(page["extract"]) > MIN_TEXT_LENGTH:
			return title, page["extract"]
	except Exception:
		pass
	return None

def fetch_random_wiki(session):
	params = {
		"action": "query",
		"list": "random",
		"rnnamespace": 0,
		"rnlimit": 5,
		"format": "json",
	}
	try:
		r = session.get(WIKI_API, params=params, timeout=10)
		data = r.json()
		for article in data["query"]["random"]:
			title = article["title"]
			res = fetch_wiki_article(title, session)
			if res:
				return res
	except Exception:
		pass
	return None

def clean_text(text):
	text = re.sub(r'\r\n', '\n', text)
	text = re.sub(r'[ \t]+', ' ', text)
	text = re.sub(r'\n{3,}', '\n\n', text)

	text = ''.join(ch for ch in text if ch.isprintable() or ch == '\n')
	return text.strip()

def write_doc(f, source, title, text):
	text = clean_text(text)
	if len(text) < MIN_TEXT_LENGTH:
		return False
	f.write(f"=== SOURCE: {source} | TITLE: {title} ===\n")
	f.write(text)
	f.write("\n\n")
	return True

def main():
	os.makedirs("data", exist_ok=True)
	session = requests.Session()
	session.headers.update({"User-Agent": "CorpusBuilder/1.0 (educational)"})

	total_docs = 0
	with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
		f.write("# RAW TEXT CORPUS\n# Sources: Project Gutenberg, Wikipedia\n\n")

		print("\n📚 Collecting Project Gutenberg books...")
		for book_id in tqdm(GUTENBERG_IDS, desc="Gutenberg"):
			res = fetch_gutenberg(book_id, session)
			if res:
				title, text = res
				if write_doc(f, "Gutenberg", title, text):
					total_docs += 1
			time.sleep(REQUEST_DELAY)

		print("\n🌐 Collecting Wikipedia articles (seeds)...")
		for topic in tqdm(WIKI_SEEDS, desc="Wikipedia seeds"):
			res = fetch_wiki_article(topic, session)
			if res:
				title, text = res
				if write_doc(f, "Wikipedia", title, text):
					total_docs += 1
			time.sleep(REQUEST_DELAY)

		target_random = 200
		random_count = 0
		pbar = tqdm(total=target_random, desc="Wikipedia random")
		while random_count < target_random:
			res = fetch_random_wiki(session)
			if res:
				title, text = res
				if write_doc(f, "Wikipedia", title, text):
					random_count += 1
					pbar.update(1)
			time.sleep(REQUEST_DELAY)
		pbar.close()

	size_mb = Path(OUTPUT_FILE).stat().st_size / 1_000_000
	print(f"\n✅ Done! {total_docs} documents written to {OUTPUT_FILE} ({size_mb:.1f} MB)")

if __name__ == "__main__":
	main()
