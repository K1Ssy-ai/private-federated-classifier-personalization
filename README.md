# Clipping sensitivity and private prior correction in federated classifier personalization

This repository contains the experiment code and run-level records for three ten-class image benchmarks. Each comparison starts from a common federated classifier and allocates the remaining record-level privacy budget to one local adaptation.

## Recalculate the reported comparisons

From the repository root, run:

```sh
python summarize_runs.py
```

This command uses only Python's standard library. It reads the CSV files under `results/` and prints the paired threshold contrasts, count-versus-bias comparisons and mean local CPU times. Each accuracy observation averages 20 clients. Each confidence interval uses five paired run differences and four degrees of freedom. The results README maps files to tables and figures. For example, at a clipping threshold of one, the output-bias lead on Fashion-MNIST with alpha 1 is 2.688 percentage points. At four the gap is −1.182 points. All five matched runs changed rank.

## Train an independent replication

The archived numerical checks used Python 3.12, PyTorch 2.5.1 on CPU, NumPy 1.26.4 and Opacus 1.5.4. From the repository root, in an empty Python 3.12 virtual environment, run:

```sh
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r experiments/requirements.txt
```

The loaders fetch the original compressed Fashion-MNIST and Kuzushiji-MNIST files on first use and check their digests. For CIFAR-10, place the original Python archive at `experiments/data/cifar-10-python.tar.gz`. Its MD5 is `c58f30108f718f92721af3b95e74349a`. The original datasets and their terms are available from [Zalando Research](https://github.com/zalandoresearch/fashion-mnist), [ROIS-DS](https://github.com/rois-codh/kmnist) and [the University of Toronto](https://www.cs.toronto.edu/~kriz/cifar.html).

1. To reproduce a single main-matrix condition, use the formal stage below for its dataset or call the two per-condition scripts in the next section. Each dataset's formal stage fits 30 shared checkpoints for three levels of label skew, two budgets and five runs. It fits the alternate heads, prepares two pilot checkpoints and fits the count-correction strength on pilot validation images. The gradient-based methods use the archived pilot choices in `published_settings/`.

   ```sh
   python reproduce.py formal --dataset fashion
   python reproduce.py formal --dataset kmnist
   python reproduce.py formal --dataset cifar
   ```

2. To repeat the gradient hyperparameter search as well, start in a fresh checkout and run `python reproduce.py pilots` **before** the formal commands. This trains the six pilot checkpoints, evaluates the original grid and writes a new selection. Fresh sampling and Gaussian noise can change the winning settings. For a comparison with the published fixed settings, use the previous step in its own checkout.

3. After the corresponding formal stage, run the threshold and whole-budget diagnostics, or measure adaptation time. The historical timing protocol uses four CPU threads and the Windows working-set API. Select each dataset separately for timing.

   ```sh
   python reproduce.py clipping --dataset fashion
   python reproduce.py clipping --dataset cifar
   python reproduce.py timing --dataset fashion
   ```

4. After the KMNIST formal stage, run `python reproduce.py kmnist-extension` for the residual and class-scaling branches and the single-update and finite-step checks.

## One main-matrix condition

For a fresh checkout, copy `published_settings/fashion.json` to `experiments/followup_20261104/unified_v2/pilot/fashion/selection_frozen.json` (create the parent directories). From the repository root run:

```sh
python experiments/followup_20261104/unified_v2/shared_pilot.py --dataset fashion --stage formal --alpha 0.5 --epsilon 4 --seed 2030
python experiments/followup_20261104/unified_v2/formal_adapt.py --dataset fashion --alpha 0.5 --epsilon 4 --seed 2030
```

The first command produces the partition and the shared checkpoint. The second fits the local models and writes `formal_result.json` under `experiments/followup_20261104/unified_v2/formal/fashion/a0.5_e4_s2030/`. It also writes the noisy histogram used for the prior branch. The dataset-wide command evaluates the pilot-selected correction strength and writes those results to `experiments/followup_20261105/E1/`. Fixed weights in the per-condition output represent separate baselines.

## Experiment outputs

| Study | Independent rerun output |
| --- | --- |
| Main accuracy matrix | `unified_v2/formal/` and `followup_20261105/E1/` |
| Threshold sweep and whole-budget comparison | `followup_20261105/E2/` and `E3/` |
| Local adaptation times | `followup_20261105/E4/` |
| Single-update noise and optimization checks | `bias_theory_check_v1/` |
| Prior-initialized adaptations | `prior_residual_pilot_v1/` and `dp_class_scale_confirm_v1/` |

These paths are under `experiments/`; `unified_v2/` is under `followup_20261104/`. Partition and initialization seeds are fixed. Fresh runs generate fresh private updates and histograms. The CSV archive preserves the values used for the paper. The per-branch budget covers one fixed deployment protocol. The clipping and training diagnostics use public benchmark records.

## Recreate the manuscript figures

Install Matplotlib 3.9.2, then run `python redraw_figures.py` from the repository root. This reads the archived CSVs and writes the vector PDFs for Figures 2 and 3 under `figures/`. Font size is set for a final figure width of 119 mm.

Figure 1 uses the original PowerPoint architecture diagram supplied in the manuscript source package.
