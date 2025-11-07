import re
from pathlib import Path
import random

import pandas as pd

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

	#_append_obo_term(terms, term)

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

def train_model(go_basic, train_fasta, train_taxonomy, train_terms, ia):
	pass


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