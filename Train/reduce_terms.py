import random

input_file = "train_terms.tsv"
output_file = "train_terms_tiny.tsv"

keep_ratio = 0.09

with open(input_file, "r", encoding = "utf-8") as fin, open(output_file, "w", encoding = "utf-8") as fout:
	header = fin.readline()
	fout.write(header)

	for line in fin:
		if random.random() < keep_ratio:
			fout.write(line)

print("done")
