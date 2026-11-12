"""Recalculate comparisons from archived public-benchmark records, with no model training."""
import csv
from collections import defaultdict
from math import sqrt
from pathlib import Path
from statistics import mean, stdev

RESULTS = Path(__file__).resolve().parent / "results"
T_975_DF4 = 2.7764451051977987


def read(name):
    with (RESULTS / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def interval(values):
    assert len(values) == 5
    center = mean(values)
    half_width = T_975_DF4 * stdev(values) / sqrt(5)
    return center - half_width, center + half_width


def clipping():
    cells = defaultdict(dict)
    for row in read("02_clipping_sweep.csv"):
        c = float(row["clip"])
        if c in (1, 4):
            key = row["dataset"], float(row["alpha"]), int(row["seed"])
            index = c, row["method"]
            assert index not in cells[key]
            cells[key][index] = float(row["test_accuracy"])
    contrasts = []
    groups = defaultdict(list)
    for (dataset, alpha, seed), scores in sorted(cells.items()):
        d1, d4 = [100 * (scores[c, "B1"] - scores[c, "B2"]) for c in (1, 4)]
        entry = d1, d4, d4 - d1, int(d1 > 0 > d4)
        contrasts.append((dataset, alpha, seed, *entry))
        groups[dataset, alpha].append(entry)
    assert len(contrasts) == 20 and len(groups) == 4
    summary = {}
    for key, entries in groups.items():
        d1, d4, changes, reversals = zip(*entries)
        summary[key] = mean(d1), mean(d4), mean(changes), *interval(changes), sum(reversals)
    return contrasts, summary


def main():
    _, summary = clipping()
    print("Clipping: dataset, alpha, gap at 1, gap at 4, change, lower, upper, reversals")
    for key, values in summary.items():
        print(*key, *(round(value, 3) for value in values))
    groups = defaultdict(dict)
    for row in read("01_main_accuracy.csv"):
        key = row["dataset"], row["alpha"], row["epsilon"]
        assert row["seed"] not in groups[key]
        groups[key][row["seed"]] = 100 * (float(row["B4"]) - float(row["B1"]))
    assert len(groups) == 18
    means = [mean(x.values()) for x in groups.values()]
    print("Prior minus learned bias (points): min, mean, max, positive means, intervals above zero")
    print(min(means), mean(means), max(means), sum(x > 0 for x in means),
          sum(interval(list(x.values()))[0] > 0 for x in groups.values()))
    timing = defaultdict(list)
    for row in read("04_cpu_timing.csv"):
        timing[row["dataset"], row["method"]].append(float(row["total_seconds"]))
    print("Stage B mean seconds (3 clients x 5 repetitions)")
    for key, values in sorted(timing.items()):
        assert len(values) == 15
        print(*key, round(mean(values), 6))


if __name__ == "__main__":
    main()
