from misc import *

if __name__ == "__main__":
	# Train ######
	# go_basic = read_obo_file("Train/go-basic.obo") # ontology graph structure
	train_fasta = read_fasta_file("Train/train_sequences.fasta") # amino acid sequences for proteins in the training set
	# train_taxonomy = read_taxonomy("Train/train_taxonomy.tsv") # taxon IDs for proteins in the training set
	# train_terms = read_terms("Train/train_terms.tsv") # the training set of proteins and corresponding annotated GO terms

	# Test ######
	# test_fasta = read_fasta_file("Test/testsuperset.fasta") # amino acid sequences for proteins on which predictions should be made
	# train_taxonomy = read_taxons_list("Test/testsuperset-taxon-list.tsv")  # taxon IDs for proteins in the test superset

	ia = read_ia_file("IA.tsv") # information accretion for each term (used to weight precision and recall)


	print(len(train_fasta))
	print(train_fasta.head())

	print(len(ia))
	print(ia.head())
