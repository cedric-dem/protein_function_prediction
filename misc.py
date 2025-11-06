import re
from pathlib import Path

import pandas as pd
from typing import List, Dict, Optional

def read_obo_file(path):
	wanted_keys = {"id", "name", "namespace", "def", "synonym", "is_a"}
	term = {}
	terms = []

	rx_def = re.compile(r'^def:\s*"(.*?)"')
	rx_syn = re.compile(r'^synonym:\s*"(.*?)"')
	rx_is_a = re.compile(r'^is_a:\s*(\S+)')
	rx_kv = re.compile(r'^(\w+):\s*(.*)$')

	def flush_current():
		nonlocal term, terms
		if term:
			# Normaliser en str + jointures
			row = {}
			for k in ["id", "name", "namespace", "def", "synonym", "is_a"]:
				v = term.get(k)
				if v is None:
					row[k] = ""
				elif isinstance(v, list):
					row[k] = "|".join(v)
				else:
					row[k] = v
			terms.append(row)
			term = {}

	with open(path, "r", encoding = "utf-8") as f:
		for raw in f:
			line = raw.rstrip("\n")

			if line.strip() == "[Term]":
				flush_current()
				continue

			if not line.strip():
				continue

			m = rx_def.match(line)
			if m:
				term["def"] = m.group(1)
				continue

			m = rx_syn.match(line)
			if m:
				term.setdefault("synonym", [])
				term["synonym"].append(m.group(1))
				continue

			m = rx_is_a.match(line)
			if m:
				term.setdefault("is_a", [])
				term["is_a"].append(m.group(1))
				continue

			m = rx_kv.match(line)
			if m:
				key, value = m.group(1), m.group(2)
				if key in wanted_keys:
					if key in ("synonym", "is_a"):
						term.setdefault(key, [])
						term[key].append(value)
					else:
						term[key] = value

		flush_current()

	df = pd.DataFrame(terms, columns = ["id", "name", "namespace", "def", "synonym", "is_a"])
	return df

def read_fasta_file(file_path):
	file_path = Path(file_path)

	if not file_path.exists():
		raise FileNotFoundError(f"FASTA not found: {file_path}")

	header_re = re.compile(r"""
        ^
        >(?P<db>[^|]+)\|                # sp or tr
        (?P<accession>[^|]+)\|
        (?P<entry_name>\S+)             # e.g., RHG10_HUMAN
        \s*
        (?P<rest>.*)                    # rest of the header line
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

	def flush_current():
		nonlocal current, seq_chunks
		if current is None:
			return
		sequence = "".join(seq_chunks).replace(" ", "").replace("\n", "").upper()
		sequence = re.sub(r"[^A-Z]", "", sequence)
		rec = current.copy()
		rec["sequence"] = sequence if sequence else None
		rec["length"] = len(sequence) if sequence else 0
		records.append(rec)
		current = None
		seq_chunks = []

	with file_path.open("r", encoding = "utf-8", errors = "replace") as fh:
		for raw_line in fh:
			line = raw_line.rstrip("\n")

			if not line:
				continue

			if line.startswith(">"):
				flush_current()

				m = header_re.match(line)
				if not m:
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

				db = m.group("db")
				accession = m.group("accession")
				entry_name = m.group("entry_name")
				rest = m.group("rest").strip()

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

	flush_current()

	df = pd.DataFrame.from_records(records, columns = [
		"db", "accession", "entry_name", "protein_name",
		"os", "ox", "gn", "pe", "sv",
		"sequence", "length"
	])
	return df

def read_taxonomy(path):
	df = pd.read_csv(path, sep = '\t', header = None, names = ['id', 'taxon_name'], usecols = [0, 1])
	return df
