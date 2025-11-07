import random
from pathlib import Path

def subsample_fasta():
	keep_prob = 0.30

	input_path = Path("train_sequences.fasta")
	output_path = Path("train_sequences_tiny.fasta")

	with input_path.open("r") as inp, output_path.open("w") as out:
		header = None
		seq_lines = []

		def flush_record():
			nonlocal header, seq_lines
			if header is None:
				return
			if random.random() < keep_prob:
				if not header.endswith("\n"):
					header_to_write = header + "\n"
				else:
					header_to_write = header
				out.write(header_to_write)

				for line in seq_lines:
					if line.endswith("\n"):
						out.write(line)
					else:
						out.write(line + "\n")

			header = None
			seq_lines = []

		for line in inp:
			if line.startswith(">"):
				flush_record()
				header = line.rstrip("\n")
			else:
				if header is not None:
					seq_lines.append(line.rstrip("\n"))

		flush_record()
	print("finished")

subsample_fasta()
