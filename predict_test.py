from misc import *

if __name__ == "__main__":
	# go_basic = read_obo_file("Train/go-basic.obo")
	# train_fasta = read_fasta_file("Train/train_sequences.fasta")
	# train_taxonomy = read_taxonomy("Train/train_taxonomy.tsv")
	train_terms = read_terms("Train/train_terms.tsv")

	# print(len(train_terms))
	print(train_terms.head())
