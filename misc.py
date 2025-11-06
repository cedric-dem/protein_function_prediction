import re
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
