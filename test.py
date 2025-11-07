from misc import *

if __name__ == "__main__":
	# Test ######

	test_fasta = read_fasta_file("Test/testsuperset.fasta", True)  # amino acid sequences for proteins on which predictions should be made
	print('=> finished reading test ')  # , test_fasta.head())

	test_taxonomy = read_taxons_list("Test/testsuperset-taxon-list.tsv")  # taxon IDs for proteins in the test superset
	print('=> finished reading taxonomy ')  # , test_taxonomy.head())

	ia = read_ia_file("IA.tsv")  # information accretion for each term (used to weight precision and recall)
	print('=> finished reading  ia ')  # , ia.head())

	produce_test_result(test_fasta, test_taxonomy, ia)
