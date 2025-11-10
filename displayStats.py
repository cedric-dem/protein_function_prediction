from misc import *

def get_training_protein_list():
	df = read_fasta_file("Train/train_sequences.fasta", False)
	return df['sequence'].to_numpy()

def get_test_protein_list():
	df = read_fasta_file("Test/testsuperset.fasta", True)
	return df['sequence'].to_numpy()

def display_probability_each_amino_acid(l):
	# 3 : prob prot starts with each aa, contains, and ends
	occurrences_total = np.zeros(len(positions))
	occurrences_start = np.zeros(len(positions))
	occurrences_end = np.zeros(len(positions))

	total = 0
	for i in range(len(l)):
		occurrences_start[positions.index(l[i][0])] += 1
		occurrences_end[positions.index(l[i][-1])] += 1

		for j in range(len(l[i])):
			index_in_positions = positions.index(l[i][j])
			occurrences_total[index_in_positions] += 1
			total += 1

	print('====> Probabilities for each amino acid')
	for i in range(len(positions)):
		prob_total = round(100 * occurrences_total[i] / total, 2)
		prob_start = round(100 * occurrences_start[i] / len(l), 2)
		prob_end = round(100 * occurrences_end[i] / len(l), 2)
		print("===> Amino acid  : ", positions[i], ":", prob_total, " %, start ", prob_start, " %, end ", prob_end, " %")

def display_heatmap_amino_acid_couples(l):
	total_occurrences = np.zeros((len(positions), len(positions)), dtype = np.float32)

	for elem in l:
		total_occurrences += get_occurences_vector_from_matrix_2d(elem, False)

	plt.xticks(ticks = np.arange(len(positions)), labels = positions, rotation = 45, ha = "right")
	plt.yticks(ticks = np.arange(len(positions)), labels = positions)

	plt.imshow(total_occurrences, cmap = 'hot', interpolation = 'nearest')
	plt.colorbar(label = 'Nb Occurrences')
	plt.title("Heatmap pairs")
	plt.show()

def display_stats_chains_length(l):
	lengths = []
	for elem in l:
		lengths.append(len(elem))

	describe_list(lengths, "protein chain lengths")

def get_proteins_functions_quantities():
	x = read_terms("Train/train_terms.tsv")
	dict_out = x['EntryID'].value_counts().to_dict()

	result = []
	for elem in dict_out:
		result.append(dict_out[elem])

	return result

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

# display_probability_each_amino_acid(all_amino_acids)
# display_heatmap_amino_acid_couples(all_amino_acids)
# display_stats_chains_length(all_amino_acids)

lengths = get_proteins_functions_quantities()
describe_list(lengths, "protein functions quantities")
