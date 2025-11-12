import pickle
import re
from pathlib import Path
import random

import pandas as pd
from config import *
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb

print(xgb.__version__)

params = {
	'tree_method': 'gpu_hist',
	'predictor': 'gpu_predictor'
}

xgb_r = xgb.XGBRegressor(**params)
print(xgb_r.get_params())

PAD_TOKEN_INDEX = 0
CHAR_TO_INDEX = {char: idx + 1 for idx, char in enumerate(positions)}
UNKNOWN_TOKEN_INDEX = len(CHAR_TO_INDEX) + 1
RNN_VOCAB_SIZE = UNKNOWN_TOKEN_INDEX + 1

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

def encode_sequence_for_rnn(sequence: str) -> np.ndarray:
	if not isinstance(sequence, str):
		return np.array([], dtype = np.int32)

	cleaned_sequence = sequence.upper()
	encoded = [CHAR_TO_INDEX.get(char, UNKNOWN_TOKEN_INDEX) for char in cleaned_sequence if char.strip()]

	if not encoded:
		return np.array([], dtype = np.int32)

	return np.asarray(encoded, dtype = np.int32)

def pad_encoded_sequences(sequences: List[np.ndarray]) -> np.ndarray:
	if not sequences:
		raise ValueError("No sequences provided for padding")

	max_length = max(seq.shape[0] for seq in sequences)
	padded = np.full((len(sequences), max_length), PAD_TOKEN_INDEX, dtype = np.int32)

	for idx, seq in enumerate(sequences):
		padded[idx, : seq.shape[0]] = seq

	return padded

class StreamingTrainingSequence(keras.utils.Sequence):
	def __init__(self, fasta_df, terms_dict, batch_size):
		self.fasta_df = fasta_df
		self.terms_dict = terms_dict
		self.batch_size = batch_size

		self.valid_indices = []
		self.sample_count = 0
		self._first_item_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None
		self._first_index: Optional[int] = None
		self._first_batch_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None

		print('====> Start retreive of fassta')
		total_rows = fasta_df.shape[0]
		for position, (index, row) in enumerate(fasta_df.iterrows()):
			if position % 2000 == 0:
				print("======> position in current loop  ", position, "/", total_rows)

			sequence = row['sequence']
			if not isinstance(sequence, str) or len(sequence) < 2:
				continue

			accession = row['accession']

			if self._first_item_cache is None:
				raise IndexError('Batch index out of range')

		if batch_idx == 0 and self._first_batch_cache is not None:
			return self._first_batch_cache

		start = batch_idx * self.batch_size
		end = min(start + self.batch_size, self.sample_count)
		batch_indices = self.valid_indices[start:end]

		batch = self._create_batch(batch_indices)

		if batch_idx == 0:
			self._first_batch_cache = batch

		return batch

	def on_epoch_end(self):
		self._first_batch_cache = None

	def peek_batch(self):
		if self._first_batch_cache is None:
			first_indices = self.valid_indices[: self.batch_size]
			self._first_batch_cache = self._create_batch(first_indices)
		return self._first_batch_cache

class StreamingRNNTrainingSequence(keras.utils.Sequence):
	def __init__(self, fasta_df, terms_dict, batch_size):
		self.fasta_df = fasta_df
		self.terms_dict = terms_dict
		self.batch_size = batch_size

		self.valid_indices: List[int] = []
		self.sample_count = 0
		self._first_item_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None
		self._first_index: Optional[int] = None
		self._first_batch_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None

		print('====> Start retrieve of fasta for RNN')
		total_rows = fasta_df.shape[0]
		for position, (index, row) in enumerate(fasta_df.iterrows()):
			if position % 2000 == 0:
				print("======> position in current loop  ", position, "/", total_rows)

			sequence = row['sequence']
			if not isinstance(sequence, str) or len(sequence) < 1:
				continue

			accession = row['accession']
			encoded_input = encode_sequence_for_rnn(sequence)
			if encoded_input.size == 0:
				continue

			if self._first_item_cache is None:
				terms_list = terms_dict.get(accession, [])
				shaped_output = get_shaped_output(terms_list)
				self._first_item_cache = (encoded_input, shaped_output)
				self._first_index = index

			self.valid_indices.append(index)

		self.sample_count = len(self.valid_indices)

		if self.sample_count == 0:
			raise ValueError("No valid training samples found in FASTA file for RNN strategy")

	def __len__(self):
		if self.sample_count == 0:
			return 0
		return (self.sample_count + self.batch_size - 1) // self.batch_size

	def _create_batch(self, batch_indices):
		batch_inputs: List[np.ndarray] = []
		batch_outputs: List[np.ndarray] = []

		for idx in batch_indices:
			if self._first_item_cache is not None and idx == self._first_index:
				encoded_input, shaped_output = self._first_item_cache
			else:
				row = self.fasta_df.loc[idx]
				sequence = row['sequence']
				accession = row['accession']

				if not isinstance(sequence, str) or len(sequence) < 1:
					continue

				terms_list = self.terms_dict.get(accession, [])
				encoded_input = encode_sequence_for_rnn(sequence)
				if encoded_input.size == 0:
					continue
				shaped_output = get_shaped_output(terms_list)

			batch_inputs.append(encoded_input)
			batch_outputs.append(shaped_output)

		if not batch_inputs:
			raise ValueError("Encountered an empty batch during RNN training generation")

		inputs = pad_encoded_sequences(batch_inputs)
		outputs = np.stack(batch_outputs).astype(np.float32, copy = False)
		return inputs, outputs

	def __getitem__(self, batch_idx):
		if batch_idx < 0 or batch_idx >= len(self):
			raise IndexError('Batch index out of range')

		if batch_idx == 0 and self._first_batch_cache is not None:
			return self._first_batch_cache

		start = batch_idx * self.batch_size
		end = min(start + self.batch_size, self.sample_count)
		batch_indices = self.valid_indices[start:end]

		batch = self._create_batch(batch_indices)

		if batch_idx == 0:
			self._first_batch_cache = batch

		return batch

	def on_epoch_end(self):
		self._first_batch_cache = None

	@property
	def vocab_size(self) -> int:
		return RNN_VOCAB_SIZE

	def peek_batch(self):
		if self._first_batch_cache is None:
			first_indices = self.valid_indices[: self.batch_size]
			self._first_batch_cache = self._create_batch(first_indices)
		return self._first_batch_cache

def get_occurences_vector_from_matrix_1d(amino_acid_list):
	matrix_occurences = np.zeros((len(positions)), dtype = np.float32)

	for i in range(len(amino_acid_list)):
		new_car = amino_acid_list[i]
		new_position = positions.index(new_car)
		matrix_occurences[new_position] += 1

	return (matrix_occurences / 4100) ** (1 / EXPONENT_OCCURENCES)

def get_occurences_vector_from_matrix_2d(amino_acid_list, concatenate_to_list = True):
	matrix_occurences = np.zeros((len(positions), len(positions)), dtype = np.float32)

	for i in range(1, len(amino_acid_list)):
		old_car = amino_acid_list[i - 1]
		new_car = amino_acid_list[i]

		old_position = positions.index(old_car)
		new_position = positions.index(new_car)

		matrix_occurences[old_position, new_position] += 1
	# todo try in percentage
	# TODO try non linear like project 1 to 0.5, 10 to 0.9, 100 to 0.9999 etc
	if concatenate_to_list:
		return ((matrix_occurences / 1300) ** (1 / EXPONENT_OCCURENCES)).reshape(-1)
	else:
		return matrix_occurences

def get_occurences_vector_from_matrix_3d(amino_acid_list):
	matrix_occurences = np.zeros((len(positions), len(positions), len(positions)), dtype = np.float32)

	for i in range(1, len(amino_acid_list) - 1):
		old_car = amino_acid_list[i - 1]
		new_car = amino_acid_list[i]
		next_car = amino_acid_list[i]

		old_position = positions.index(old_car)
		new_position = positions.index(new_car)
		next_position = positions.index(next_car)

		matrix_occurences[old_position, new_position, next_position] += 1

	# todo try in percentage
	# TODO try non linear like project 1 to 0.5, 10 to 0.9, 100 to 0.9999 etc
	return ((matrix_occurences / 500) ** (1 / EXPONENT_OCCURENCES)).reshape(-1)

def get_shaped_input(amino_acid_list):
	occurences_as_vector_1d = get_occurences_vector_from_matrix_1d(amino_acid_list)
	occurences_as_vector_2d = get_occurences_vector_from_matrix_2d(amino_acid_list)
	# occurences_as_vector_3d = get_occurences_vector_from_matrix_3d(amino_acid_list)

	start = np.zeros(len(positions), dtype = np.float32)
	end = np.zeros(len(positions), dtype = np.float32)

	start[positions.index(amino_acid_list[0])] = 1
	end[positions.index(amino_acid_list[1])] = 1

	size = len(amino_acid_list)  # todo split in size size_perentile, like 10 values, 1 at the proportion

	# TODO : cubic root + clamp between  0 and 1
	return np.clip(np.concatenate((occurences_as_vector_1d, occurences_as_vector_2d, start, end, np.array([size / 36000.0], dtype = np.float32))), 0, 1)

def get_shaped_output(raw_output):
	result = np.zeros(len(all_terms), dtype = np.float32)
	for current_output in raw_output:
		result[all_terms.index(current_output)] = 1
	return result

def get_terms_as_dict(terms):
	result = {}
	print("====> start retrieve dict terms")
	for index, row in terms.iterrows():
		if index % 2000 == 0:
			print('======> current pos in loop', index, "/", terms.shape[0])
		if row["EntryID"] not in result:
			result[row["EntryID"]] = []

		result[row["EntryID"]].append(row["term"])

	return result

def get_dataset(fasta, terms):
	terms_as_dict = get_terms_as_dict(terms)
	if STRATEGY == "RNN":
		return StreamingRNNTrainingSequence(fasta, terms_as_dict, BATCH_SIZE_TRAIN)
	return StreamingTrainingSequence(fasta, terms_as_dict, BATCH_SIZE_TRAIN)

def _collect_dataset_arrays(dataset):
	inputs = []
	outputs = []

	for batch_idx in range(len(dataset)):
		batch_inputs, batch_outputs = dataset[batch_idx]
		inputs.append(batch_inputs)
		outputs.append(batch_outputs)

	if not inputs:
		raise ValueError("Dataset did not yield any batches")

	features = np.concatenate(inputs, axis = 0).astype(np.float32, copy = False)
	labels = np.concatenate(outputs, axis = 0).astype(np.float32, copy = False)
	return features, labels

def train_model(go_basic, train_fasta, train_taxonomy, train_terms, ia):
	dataset = get_dataset(train_fasta, train_terms)

	if STRATEGY == "XGB":
		train_xgb(dataset)
	elif STRATEGY == "NN":
		train_nn(dataset)
	elif STRATEGY == "RNN":
		train_rnn(dataset)
	elif STRATEGY == "KNN":
		train_knn(dataset)

def train_nn(dataset):
	first_inputs, first_outputs = dataset.peek_batch()

	INPUT_SIZE = first_inputs.shape[1]
	OUTPUT_SIZE = first_outputs.shape[1]

	print('==> Neural Network I/O Size :', INPUT_SIZE, '/', OUTPUT_SIZE)
	print("==> Dataset size ", dataset.sample_count)

	print('==> building model')
	if HIDDEN_LAYER == None:
		model = keras.Sequential([layers.Input(shape = (INPUT_SIZE,)), layers.Dense(OUTPUT_SIZE, activation = None, use_bias = True)])
	else:
		model = keras.Sequential([layers.Input(shape = (INPUT_SIZE,)), layers.Dense(HIDDEN_LAYER, activation = 'sigmoid'), layers.Dense(OUTPUT_SIZE, activation = None)])

	print('==> compile model')
	model.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.001), loss = "mse", metrics = ["mae"])

	print('==> fitting model')
	model.fit(dataset, epochs = N_EPOCHS, verbose = 1)

	print('==> saving model')
	model.save(NN_MODEL_NAME)

def train_rnn(dataset):
	_, first_outputs = dataset.peek_batch()

	output_size = first_outputs.shape[1]
	vocab_size = dataset.vocab_size

	print('==> RNN Output Size :', output_size)
	print("==> Dataset size ", dataset.sample_count)

	inputs = keras.Input(shape = (None,), dtype = 'int32')

	RNN_EMBEDDING_DIM = 128
	RNN_RECURRENT_UNITS = 128

	x = layers.Embedding(vocab_size, RNN_EMBEDDING_DIM, mask_zero = True)(inputs)
	x = layers.Bidirectional(layers.GRU(RNN_RECURRENT_UNITS))(x)

	if HIDDEN_LAYER is None:
		outputs = layers.Dense(output_size, activation = None)(x)
	else:
		x = layers.Dense(HIDDEN_LAYER, activation = 'relu')(x)
		outputs = layers.Dense(output_size, activation = None)(x)

	model = keras.Model(inputs = inputs, outputs = outputs)

	print('==> compile RNN model')
	model.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.001), loss = "mse", metrics = ["mae"])

	print('==> fitting RNN model')
	model.fit(dataset, epochs = N_EPOCHS, verbose = 1)

	print('==> saving RNN model')
	model.save(RNN_MODEL_NAME)

def train_xgb(dataset):
	first_inputs, first_outputs = dataset.peek_batch()

	input_size = first_inputs.shape[1]
	output_size = first_outputs.shape[1]

	print('==> XGBoost I/O Size :', input_size, '/', output_size)
	print("==> Dataset size ", dataset.sample_count)

	print('==> collecting dataset for XGBoost')
	features, labels = _collect_dataset_arrays(dataset)

	XGB_MAX_DEPTH = 2
	XGB_LEARNING_RATE = 0.1
	XGB_N_ESTIMATORS = 1
	XGB_SUBSAMPLE = 0.3
	XGB_COLSAMPLE = 0.3

	base_regressor = xgb.XGBRegressor(
		objective = 'reg:squarederror',
		n_estimators = XGB_N_ESTIMATORS,
		learning_rate = XGB_LEARNING_RATE,
		max_depth = XGB_MAX_DEPTH,
		subsample = XGB_SUBSAMPLE,
		colsample_bytree = XGB_COLSAMPLE,
		tree_method = 'gpu_hist',
		n_jobs = -1,
		predictor = 'gpu_predictor'
	)

	model = MultiOutputRegressor(base_regressor)

	print('==> training XGBoost model')
	model.fit(features, labels)

	bundle = {
		"model": model,
		"input_size": input_size,
		"output_size": output_size,
		"terms": all_terms,
	}

	print('==> saving XGBoost model')
	with open(XGB_MODEL_NAME, 'wb') as fp:
		pickle.dump(bundle, fp)

def train_knn(dataset):
	first_inputs, first_outputs = dataset.peek_batch()

	input_size = first_inputs.shape[1]
	output_size = first_outputs.shape[1]

	print('==> KNN I/O Size :', input_size, '/', output_size)
	print("==> Dataset size ", dataset.sample_count)

	print('==> collecting dataset for KNN')
	features, labels = _collect_dataset_arrays(dataset)

	KNN_N_NEIGHBORS = 3

	print('==> create regressor')
	base_regressor = KNeighborsRegressor(
		n_neighbors = KNN_N_NEIGHBORS,
		weights = 'distance',
		n_jobs = -1,
	)

	model = MultiOutputRegressor(base_regressor, n_jobs = -1)

	print('==> training KNN model')
	model.fit(features, labels)

	bundle = {
		"model": model,
		"input_size": input_size,
		"output_size": output_size,
		"terms": all_terms,
	}

	print('==> saving KNN model')
	with open(KNN_MODEL_NAME, 'wb') as fp:
		pickle.dump(bundle, fp)

def get_random_submission(test_fasta, test_taxonomy, ia):
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

	return rows

def predict_output(predictor, formatted_input):
	if isinstance(predictor, keras.Model):
		model_inputs = getattr(predictor, "inputs", None)
		input_dtype = None
		if model_inputs:
			input_dtype = model_inputs[0].dtype

		if input_dtype is not None:
			np_dtype = tf.as_dtype(input_dtype).as_numpy_dtype
			vector = np.asarray(formatted_input, dtype = np_dtype)
		else:
			vector = np.asarray(formatted_input)

		if vector.ndim == 1:
			vector = np.expand_dims(vector, axis = 0)

		return predictor.predict(vector, batch_size = 32)

	vector = np.asarray(formatted_input, dtype = np.float32)
	if vector.ndim == 1:
		vector = vector.reshape(1, -1)
	return predictor.predict(vector)

def process_batch(predictor, lowest_value, result, names, inputs):
	if not inputs:
		return
	predictions = predict_output(predictor, inputs)
	for protein_name, vector in zip(names, predictions):
		for output_index, score in enumerate(vector):
			if score > lowest_value:  # todo : try with only take max on that, not every single one
				score = min(round(score, 3), 1.0)
				result.append([protein_name, all_terms[output_index], score])

def get_nn_submission(test_fasta, test_taxonomy, ia):
	result = []
	predictor = keras.models.load_model(NN_MODEL_NAME)

	batch_inputs = []
	batch_names = []

	total_rows = len(test_fasta)

	display_every_percentage = 3

	next_progress = display_every_percentage
	for row_number, (index, row) in enumerate(test_fasta.iterrows(), start = 1):
		this_protein_name = row["protein_name"]

		this_sequence = row["sequence"]

		formatted_input = get_shaped_input(this_sequence)

		batch_inputs.append(formatted_input)
		batch_names.append(this_protein_name)
		while total_rows and next_progress <= 100 and row_number * 100 >= next_progress * total_rows:
			print(f"=========> Processed {row_number}/{total_rows} rows ({next_progress}%): index {index}")
			next_progress += display_every_percentage

		if len(batch_inputs) == BATCH_SIZE_TEST:
			process_batch(predictor, lowest_value, result, batch_names, batch_inputs)
			batch_inputs = []
			batch_names = []

	process_batch(predictor, lowest_value, result, batch_names, batch_inputs)

	return result

def get_rnn_submission(test_fasta, test_taxonomy, ia):
	result = []
	predictor = keras.models.load_model(RNN_MODEL_NAME)

	batch_inputs: List[np.ndarray] = []
	batch_names: List[str] = []

	total_rows = len(test_fasta)

	display_every_percentage = 3

	next_progress = display_every_percentage
	for row_number, (index, row) in enumerate(test_fasta.iterrows(), start = 1):
		this_protein_name = row["protein_name"]

		this_sequence = row["sequence"]

		encoded_sequence = encode_sequence_for_rnn(this_sequence)
		if encoded_sequence.size == 0:
			continue

		batch_inputs.append(encoded_sequence)
		batch_names.append(this_protein_name)
		while total_rows and next_progress <= 100 and row_number * 100 >= next_progress * total_rows:
			print(f"=========> Processed {row_number}/{total_rows} rows ({next_progress}%): index {index}")
			next_progress += display_every_percentage

		if len(batch_inputs) == BATCH_SIZE_TEST:
			padded_inputs = pad_encoded_sequences(batch_inputs)
			process_batch(predictor, lowest_value, result, batch_names, padded_inputs)
			batch_inputs = []
			batch_names = []

	if batch_inputs:
		padded_inputs = pad_encoded_sequences(batch_inputs)
		process_batch(predictor, lowest_value, result, batch_names, padded_inputs)

	return result

def load_knn_model():
	with open(KNN_MODEL_NAME, 'rb') as fp:
		bundle = pickle.load(fp)

	if not isinstance(bundle, dict) or "model" not in bundle:
		raise ValueError("Invalid KNN model bundle")

	return bundle

def load_xgb_model():
	with open(XGB_MODEL_NAME, 'rb') as fp:
		bundle = pickle.load(fp)

	if not isinstance(bundle, dict) or "model" not in bundle:
		raise ValueError("Invalid XGBoost model bundle")

	return bundle

def get_knn_submission(test_fasta, test_taxonomy, ia):
	bundle = load_knn_model()
	predictor = bundle["model"]
	expected_input = bundle.get("input_size")
	if expected_input is None:
		raise ValueError("KNN model is missing input size metadata")

		raise ValueError(f"Input size mismatch for XGBoost model: expected {expected_input}, got {formatted_input.shape[0]}")
		batch_inputs.append(formatted_input)
		batch_names.append(this_protein_name)
		while total_rows and next_progress <= 100 and row_number * 100 >= next_progress * total_rows:
			print(f"=========> Processed {row_number}/{total_rows} rows ({next_progress}%): index {index}")
			next_progress += display_every_percentage

		if len(batch_inputs) == BATCH_SIZE_TEST:
			process_batch(predictor, lowest_value, result, batch_names, batch_inputs)
			batch_inputs = []
			batch_names = []

	process_batch(predictor, lowest_value, result, batch_names, batch_inputs)

	return result

def produce_test_result(test_fasta, test_taxonomy, ia):
	if STRATEGY == "RANDOM":
		rows = get_random_submission(test_fasta, test_taxonomy, ia)

	elif STRATEGY == "NN":
		rows = get_nn_submission(test_fasta, test_taxonomy, ia)
	elif STRATEGY == "RNN":
		rows = get_rnn_submission(test_fasta, test_taxonomy, ia)
	elif STRATEGY == "XGB":
		rows = get_xgb_submission(test_fasta, test_taxonomy, ia)
	elif STRATEGY == "KNN":
		rows = get_knn_submission(test_fasta, test_taxonomy, ia)
	else:
		raise ValueError(f"Unknown strategy '{STRATEGY}'")
	submission = pd.DataFrame(rows, columns = ["EntryID", "term", "score"])
	submission.to_csv(
		SUBMISSION_NAME,
		header = False,
		sep = "\t",
		index = False,
		float_format = "%.3f",
	)
	return submission

######

all_terms = get_list_of_raw_terms()

######
