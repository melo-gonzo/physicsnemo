"""DeepJEB++: add outlier-filtered splits to the manifest (2026-09-22).

The FEA census found 4.25% of brackets with an unphysical maximum von Mises stress (above 10,000 MPa against a yield of
903 MPa for Ti-6Al-4V; the worst is 1.19e7 MPa). Those brackets are excluded from the clean splits, by bracket (all four
load cases), so seed pairs stay disjoint between train and test. Nested rungs are seed-0 permutation prefixes of the clean
training brackets. The pooled _stats are left as computed over the full train split (a common scale for both models).
"""
import csv, json, os, re, stat, sys
import numpy as np

U = "/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/datasets/deepjebpp_pdmsh"
THRESH = 1.0e4
man_path = os.path.join(U, "manifest.json")
man = json.load(open(man_path))
loads = list(man["_load_cases"].keys())
csv_path = None
for root, _, files in os.walk(os.path.join(U, "deepjebpp_raw")):
    if "deepjebpp_labels.csv" in files:
        csv_path = os.path.join(root, "deepjebpp_labels.csv"); break
assert csv_path, "labels csv not found"
rows = list(csv.DictReader(open(csv_path)))
print("csv columns:", list(rows[0].keys())[:20], "rows", len(rows))
case_col = next(c for c in rows[0] if c.lower() in ("case", "case_id", "id", "name"))
maxvm_cols = [c for c in rows[0] if c.endswith("maxvm")]
assert len(maxvm_cols) == 4, maxvm_cols
def bracket_of(case: str) -> str:
    m = re.match(r"(\d+)-(\d+)-.*IS(\d+)", case)
    assert m, case
    return f"{m.group(1)}-{m.group(2)}-IS{m.group(3)}"
bad = set()
for r in rows:
    if max(float(r[c]) for c in maxvm_cols) > THRESH:
        bad.add(bracket_of(r[case_col]))
train_b = [b for b in man["train_brackets"]]
test_b = [b for b in man["test_brackets"]]
missing = [b for b in train_b[:50] if bracket_of(rows[0][case_col]) is None]
train_c = [b for b in train_b if b not in bad]
test_c = [b for b in test_b if b not in bad]
print(f"bad brackets {len(bad)}; train {len(train_b)} -> {len(train_c)}; test {len(test_b)} -> {len(test_c)}")
assert len(train_c) < len(train_b) and len(test_c) < len(test_b), "filter matched nothing: bracket id mapping is wrong"
files = lambda bs: sorted(f"{b}_{ld}" for b in bs for ld in loads)
perm = np.random.RandomState(0).permutation(len(train_c))
train_c_perm = [train_c[i] for i in perm]
man["train_clean"] = files(train_c); man["test_clean"] = files(test_c)
man["train_clean_brackets"] = train_c; man["test_clean_brackets"] = test_c
for n in (256, 1024, 4096):
    man[f"train_clean_n{n}"] = files(train_c_perm[:n])
for ld in loads:
    man[f"test_clean_{ld}"] = sorted(f"{b}_{ld}" for b in test_c)
man["_clean_note"] = (f"clean splits exclude {len(bad)} brackets whose labels csv reports max von Mises > {THRESH:g} MPa in any load case "
                      f"(census 2026-09-22); nested clean rungs are seed-0 permutation prefixes of the clean train brackets; _stats unchanged (full train)")
os.chmod(man_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP)
json.dump(man, open(man_path, "w"), indent=1)
os.chmod(man_path, stat.S_IRUSR | stat.S_IRGRP)
print({k: len(man[k]) for k in ("train_clean", "test_clean", "train_clean_n256", "train_clean_n1024", "train_clean_n4096")})
