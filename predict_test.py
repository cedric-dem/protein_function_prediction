from misc import *

if __name__ == "__main__":
	# go_basic = read_obo_file("Train/go-basic.obo")
	train_fasta = read_fasta_file("Train/train_sequences.fasta")

	print(train_fasta.head())
