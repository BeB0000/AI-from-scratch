
import sys

def main():

	if len(sys.argv) != 3:
		print("Usage: python3 cleanTxt.py <filename> <letter_to_remove>")
		sys.exit(1)

	filename = sys.argv[1]
	letter = sys.argv[2]

	try:
		with open(filename, 'r', encoding='utf-8') as f:
			content = f.read()
	except FileNotFoundError:
		print(f"Error: File '{filename}' not found.")
		sys.exit(1)
	except Exception as e:
		print(f"Error reading file: {e}")
		sys.exit(1)

	cleaned_content = content.replace(letter, '')

	try:
		with open(filename, 'w', encoding='utf-8') as f:
			f.write(cleaned_content)
	except Exception as e:
		print(f"Error writing file: {e}")
		sys.exit(1)

	print(f"Removed all occurrences of '{letter}' from {filename}.")

if __name__ == "__main__":
	main()
