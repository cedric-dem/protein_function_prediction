import re
from pathlib import Path
import random

import pandas as pd
from dataclasses import dataclass
from typing import List
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def get_list_of_raw_terms():
	complete_df = read_obo_file("Train/go-basic.obo")
	raw_terms_list = []
	for index, rows in complete_df.iterrows():
		raw_terms_list.append(rows.id)
	raw_terms_list.sort()
	return raw_terms_list

def _obo_row(term):
	row = {}
	for key in ["id", "name", "namespace", "def", "synonym", "is_a"]:
		value = term.get(key)
		if value is None:
			row[key] = ""
		elif isinstance(value, list):
			row[key] = "|".join(value)
		else:
			row[key] = value
	return row

def _append_obo_term(terms, term):
	if term:
		terms.append(_obo_row(term))

def read_obo_file(path):
	wanted_keys = {"id", "name", "namespace", "def", "synonym", "is_a"}
	term = {}
	terms = []

	rx_def = re.compile(r'^def:\s*"(.*?)"')
	rx_syn = re.compile(r'^synonym:\s*"(.*?)"')
	rx_is_a = re.compile(r'^is_a:\s*(\S+)')
	rx_kv = re.compile(r'^(\w+):\s*(.*)$')

	with open(path, "r", encoding = "utf-8") as f:
		for raw in f:
			line = raw.rstrip("\n")

			if line.strip() == "[Term]":
				_append_obo_term(terms, term)
				term = {}
				continue

			if not line.strip():
				continue

			match = rx_def.match(line)
			if match:
				term["def"] = match.group(1)
				continue

			match = rx_syn.match(line)
			if match:
				term.setdefault("synonym", [])
				term["synonym"].append(match.group(1))
				continue

			match = rx_is_a.match(line)
			if match:
				term.setdefault("is_a", [])
				term["is_a"].append(match.group(1))
				continue

			match = rx_kv.match(line)
			if match:
				key, value = match.group(1), match.group(2)
				if key in wanted_keys:
					if key in ("synonym", "is_a"):
						term.setdefault(key, [])
						term[key].append(value)
					else:
						term[key] = value

	df = pd.DataFrame(terms, columns = ["id", "name", "namespace", "def", "synonym", "is_a"])
	return df

def _normalized_sequence(chunks):
	sequence = "".join(chunks).replace(" ", "").replace("\n", "").upper()
	return re.sub(r"[^A-Z]", "", sequence)

def _append_fasta_record(records, current, seq_chunks):
	if current is None:
		return
	sequence = _normalized_sequence(seq_chunks)
	record = current.copy()
	record["sequence"] = sequence if sequence else None
	record["length"] = len(sequence) if sequence else 0
	records.append(record)

def read_fasta_file(file_path, only_species):
	file_path = Path(file_path)

	if not file_path.exists():
		raise FileNotFoundError(f"FASTA not found: {file_path}")

	header_re = re.compile(r"""
        ^
        >(?P<db>[^|]+)\|
        (?P<accession>[^|]+)\|
        (?P<entry_name>\S+)
        \s*
        (?P<rest>.*)
        $
    """, re.VERBOSE)

	kv_re = re.compile(r"""
        (?P<key>[A-Z]{2,})=
        (?P<val>.*?)
        (?=(?:\s[A-Z]{2,}=)|$)
    """, re.VERBOSE)

	records = []
	current = None
	seq_chunks = []

	with file_path.open("r", encoding = "utf-8", errors = "replace") as fh:
		for raw_line in fh:
			line = raw_line.rstrip("\n")

			if not line:
				continue

			if line.startswith(">"):
				_append_fasta_record(records, current, seq_chunks)
				current = None
				seq_chunks = []

				if only_species:
					header_body = line[1:].strip()
					protein_name = None
					os = None

					if header_body:
						parts = header_body.split(maxsplit = 1)
						if parts:
							protein_name = parts[0] or None
							if len(parts) > 1:
								os = parts[1] or None

					current = {
						"db": None,
						"accession": None,
						"entry_name": None,
						"protein_name": protein_name,
						"os": os,
						"ox": None,
						"gn": None,
						"pe": None,
						"sv": None,
					}
					continue

				match = header_re.match(line)
				if not match:
					header_body = line[1:].strip()
					current = {
						"db": None,
						"accession": None,
						"entry_name": None,
						"protein_name": header_body,
						"os": None,
						"ox": None,
						"gn": None,
						"pe": None,
						"sv": None,
					}
					continue

				db = match.group("db")
				accession = match.group("accession")
				entry_name = match.group("entry_name")
				rest = match.group("rest").strip()

				protein_name = None
				os = ox = gn = pe = sv = None

				if rest:
					first_kv = re.search(r"\b[A-Z]{2,}=", rest)
					if first_kv:
						protein_name = rest[: first_kv.start()].strip() or None
						kv_part = rest[first_kv.start():]
					else:
						protein_name = rest or None
						kv_part = ""

					if kv_part:
						for km in kv_re.finditer(kv_part):
							key = km.group("key")
							val = km.group("val").strip()
							if key == "OS":
								os = val or None
							elif key == "OX":
								ox = val or None
							elif key == "GN":
								gn = val or None
							elif key == "PE":
								pe = val or None
							elif key == "SV":
								sv = val or None

				current = {
					"db": db or None,
					"accession": accession or None,
					"entry_name": entry_name or None,
					"protein_name": protein_name,
					"os": os,
					"ox": ox,
					"gn": gn,
					"pe": pe,
					"sv": sv,
				}
			else:
				if current is None:
					continue
				seq_chunks.append(line.strip())

	_append_fasta_record(records, current, seq_chunks)

	df = pd.DataFrame.from_records(records, columns = [
		"db", "accession", "entry_name", "protein_name",
		"os", "ox", "gn", "pe", "sv",
		"sequence", "length"
	])
	return df

def read_taxonomy(path):
	df = pd.read_csv(path, sep = '\t', header = None, names = ['id', 'taxon_name'], usecols = [0, 1])
	return df

def read_terms(path):
	df = pd.read_csv(path, sep = '\t', header = 0, names = ["EntryID", "term", "aspect"], usecols = [0, 1])
	return df

def read_taxons_list(path):
	df = pd.read_csv(path, sep = '\t', header = 0, names = ["ID", "Species"], usecols = [0, 1])
	return df

def read_ia_file(path):
	df = pd.read_csv(path, sep = '\t', header = None, names = ["term", "weight"], usecols = [0, 1])
	return df

class dataPoint(object):
	def __init__(self, raw_input, raw_output):
		self.raw_input = raw_input
		self.input = get_shaped_input(raw_input)

		self.raw_output = raw_output

		self.output = None
		if raw_output:
			self.output = get_shaped_output(raw_output)

def get_matrix_occurences(amino_acid_list):
	matrix_occurences = [[0 for i in range(len(positions))] for j in range(len(positions))]

	for i in range(1, len(amino_acid_list)):
		old_car = amino_acid_list[i - 1]
		new_car = amino_acid_list[i]

		old_position = positions.index(old_car)
		new_position = positions.index(new_car)

		matrix_occurences[old_position][new_position] += 1

	# todo try in percentage
	# TODO try non linear like project 1 to 0.5, 10 to 0.9, 100 to 0.9999 etc

	# for line in matrix_occurences:
	#	print(line)
	return matrix_occurences

def get_shaped_input(amino_acid_list):
	matrix_occurences = get_matrix_occurences(amino_acid_list)
	occurences_as_vector = [x / 1300 for line in matrix_occurences for x in line]

	start = [0 for _ in range(len(matrix_occurences))]
	end = [0 for _ in range(len(matrix_occurences))]

	start[positions.index(amino_acid_list[0])] = 1
	end[positions.index(amino_acid_list[1])] = 1

	size = len(amino_acid_list)  # todo split in size size_perentile, like 10 values, 1 at the proportion

	return occurences_as_vector + start + end + [size / 36000]

def get_shaped_output(raw_output):
	result = [0 for _ in range(len(all_terms))]
	for current_output in raw_output:
		result[all_terms.index(current_output)] = 1
	return result

def get_terms_as_dict(terms):
	result = {}
	for index, row in terms.iterrows():
		if row["EntryID"] not in result:
			result[row["EntryID"]] = []

		result[row["EntryID"]].append(row["term"])

	return result

def get_dataset(fasta, terms):
	terms_as_dict = get_terms_as_dict(terms)
	dataset = []

	for index, row in fasta.iterrows():
		sequence = row['sequence']
		name = row['accession']

		list_terms = []
		if name in terms_as_dict:  # for debug, should not be outside if
			list_terms = terms_as_dict[name]

		new_datapoint = dataPoint(sequence, list_terms)
		dataset.append(new_datapoint)

	return dataset

def train_model(go_basic, train_fasta, train_taxonomy, train_terms, ia):
	dataset = get_dataset(train_fasta, train_terms)

	train_nn(dataset)

def train_nn(dataset):
	N = len(dataset[0].input)
	M = len(dataset[0].output)

	for dp in dataset:
		if len(dp.input) != N or len(dp.output) != M:
			raise ValueError("duhhh")

	X = np.array([dp.input for dp in dataset], dtype = np.float32)
	Y = np.array([dp.output for dp in dataset], dtype = np.float32)

	model = keras.Sequential([
		layers.Input(shape = (N,)),
		layers.Dense(M, activation = None)
	])

	model.compile(
		optimizer = keras.optimizers.Adam(learning_rate = 0.001),
		loss = "mse",
		metrics = ["mae"]
	)

	model.fit(
		X, Y,
		epochs = 100,
		batch_size = 16,
		verbose = 1
	)

	model.save("model_v0.keras")

def produce_test_result(test_fasta, test_taxonomy, ia):
	entry_names = test_fasta.get("protein_name")
	if entry_names is None:
		raise KeyError("test_fasta must contain an 'entry_name' column")

	term_column = ia.get("term")
	if term_column is None:
		raise KeyError("ia must contain a 'term' column")

	available_terms = term_column.dropna()
	if available_terms.empty:
		raise ValueError("ia 'term' column must contain at least one non-null value")

	term_pool = available_terms.tolist()

	rows = []
	for entry in entry_names.dropna():
		entry_id = str(entry)
		if len(term_pool) >= 2:
			selected_terms = random.sample(term_pool, k = 3)
		else:
			selected_terms = random.choices(term_pool, k = 3)

		for term in selected_terms:
			score = round(random.uniform(0.1, 0.2), 3)
			rows.append({
				"EntryID": entry_id,
				"term": term,
				"score": score,
			})

	submission = pd.DataFrame(rows, columns = ["EntryID", "term", "score"])
	submission.to_csv("submission.tsv", header = False, sep = "\t", index = False)
	return submission

######

# positions = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
# Dont know why but some have b, u,x
positions = ['A', "B", 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', "U", 'V', 'W', "X", 'Y']
all_terms = get_list_of_raw_terms()

######
