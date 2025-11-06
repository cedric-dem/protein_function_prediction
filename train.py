from misc import *

if __name__ == "__main__":
	# Train ######

	go_basic = read_obo_file("Train/go-basic.obo")  # ontology graph structure
	print('=> Go basic', go_basic.head())

	train_fasta = read_fasta_file("Train/train_sequences.fasta", False)  # amino acid sequences for proteins in the training set
	print('=> train fasta', train_fasta.head())

	train_taxonomy = read_taxonomy("Train/train_taxonomy.tsv")  # taxon IDs for proteins in the training set
	print('=> train taxonomy', train_taxonomy.head())

	train_terms = read_terms("Train/train_terms.tsv")  # the training set of proteins and corresponding annotated GO terms
	print('=> train terms', train_terms.head())

	ia = read_ia_file("IA.tsv")  # information accretion for each term (used to weight precision and recall)
	print('=> ia ', ia.head())

	train_model(go_basic, train_fasta, train_taxonomy, train_terms, ia)
