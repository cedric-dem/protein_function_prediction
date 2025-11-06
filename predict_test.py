import pandas as pd

fasta_path = "Test/testsuperset.fasta"

pro_ids = []

with open(fasta_path, "r") as f:
	for line in f:
		line = line.strip()
		if line.startswith(">"):
			prot_id = line[1:].split(" ")[0]
			pro_ids.append(prot_id)

df = pd.DataFrame(pro_ids, columns = ["protein_ids"])

print(df)
