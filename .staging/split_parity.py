# Deterministic routing/parity check for split_dataset_for_evaluation.
import sys
from datasets import Dataset
from core.training.eval_dataset import split_dataset_for_evaluation
bad = 0
for n in [0, 1, 31, 32, 33, 40, 100, 1000, 2560, 2561, 5000]:
    d = split_dataset_for_evaluation(Dataset.from_dict({"i": list(range(n))}))
    try:
        r = split_dataset_for_evaluation([{"i": i} for i in range(n)])
    except AttributeError as e:
        print(f"n={n}: list split raised {e}"); bad += 1; continue
    if (r is None) != (d is None):
        print(f"n={n}: None mismatch"); bad += 1; continue
    if r is None:
        print(f"n={n}: no split (both)"); continue
    ok = [x["i"] for x in r[0]] == d[0]["i"] and [x["i"] for x in r[1]] == d[1]["i"]
    print(f"n={n}: train={len(r[0])} eval={len(r[1])} parity={ok} types={type(r[0]).__name__}/{type(d[0]).__name__}")
    bad += not ok
print("PARITY", "FAIL" if bad else "OK", bad)
sys.exit(1 if bad else 0)
