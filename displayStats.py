from misc import *

def get_training_protein_list():
	training_df = read_fasta_file("Train/train_sequences.fasta", False)
	return training_df['sequence'].to_numpy()

def get_test_protein_list():
	test_df = read_fasta_file("Test/testsuperset.fasta", True)
	return test_df['sequence'].to_numpy()

def display_probability_each_amino_acid(protein_list):
	occurrences_total = np.zeros(len(positions))
	occurrences_start = np.zeros(len(positions))
	occurrences_end = np.zeros(len(positions))

	total = 0
	for current_protein_index in range(len(protein_list)):
		occurrences_start[positions.index(protein_list[current_protein_index][0])] += 1
		occurrences_end[positions.index(protein_list[current_protein_index][-1])] += 1

		for j in range(len(protein_list[current_protein_index])):
			index_in_positions = positions.index(protein_list[current_protein_index][j])
			occurrences_total[index_in_positions] += 1
			total += 1

	print('====> Probabilities for each amino acid')
	for current_position_index in range(len(positions)):
		prob_total = round(100 * occurrences_total[current_position_index] / total, 2)
		prob_start = round(100 * occurrences_start[current_position_index] / len(protein_list), 2)
		prob_end = round(100 * occurrences_end[current_position_index] / len(protein_list), 2)
		print("===> Amino acid  : ", positions[current_position_index], ":", prob_total, " %, start ", prob_start, " %, end ", prob_end, " %")

def display_heatmap_amino_acid_couples(protein_list):
	total_occurrences = np.zeros((len(positions), len(positions)), dtype = np.float32)

	for protein in protein_list:
		total_occurrences += get_occurences_vector_from_matrix_2d(protein, False)

	plt.xticks(ticks = np.arange(len(positions)), labels = positions, rotation = 45, ha = "right")
	plt.yticks(ticks = np.arange(len(positions)), labels = positions)

	plt.imshow(total_occurrences, cmap = 'hot', interpolation = 'nearest')
	plt.colorbar(label = 'Nb Occurrences')
	plt.title("Heatmap pairs")
	plt.show()

def display_stats_chains_length(protein_list):
	protein_length = []
	for protein in protein_list:
		protein_length.append(len(protein))

	describe_list(protein_length, "protein chain lengths")

def get_proteins_functions_quantities():
	df_train = read_terms("Train/train_terms.tsv")
	function_quantity_per_protein = df_train['EntryID'].value_counts().to_dict()

	functions_quantity = []
	for protein in function_quantity_per_protein:
		functions_quantity.append(function_quantity_per_protein[protein])

	return functions_quantity

def describe_list(l, name):
	l.sort()

	print('======> ', name)
	print('==> Min Element : ', min(l))
	print('==> median element : ', l[len(l) // 2])
	print('==> Max Element : ', max(l))

	print('==> Average Element : ', sum(l) / len(l))
	print('==> Standard Deviation : ', np.std(l))

	plt.plot(l)
	plt.title("plot evolution of " + name)
	plt.xlabel("number of elem")
	plt.ylabel("value")

	plt.show()

train_amino_acid_list = get_training_protein_list()
test_amino_acid_list = get_test_protein_list()

all_amino_acids = np.concatenate((train_amino_acid_list, test_amino_acid_list))

display_probability_each_amino_acid(all_amino_acids)
display_heatmap_amino_acid_couples(all_amino_acids)
display_stats_chains_length(all_amino_acids)

lengths = get_proteins_functions_quantities()
describe_list(lengths, "protein functions quantities")
