# NMMCC

The author's implementation of 'Safety-aware explainable deep reinforcement learning for nephrotoxic medication management in critical care'

## How to use

### Step 0) Data Preparation: 

#### Part 1: MIMIC-IV Sepsis Cohort

Run `preprocess.ipynb`

The code used to define, extract and preprocess the patient cohort from MIMIC-IV. This code produces two CSV files, `preprocess_sepsis.csv` and `demog_sepsis.csv`.

#### Part 2: Preprocess the extracted trajectories

Run `cohort.ipynb`

The patient cohort provided in part 1 is extracted for general purposes. This script places the extracted trajectories into a more convenient format for the DeD learning processes and also constructs independent training, validation and test datasets. It also removes the raw-input columns from the observations (`o:input_total`, `o:input_4hourly`, and `o:max_dose_vaso`), if they are included by the cohort generator.

---

### Step 1) Train the Dead-end (D-) and Recovery (R-) Networks

Run `train_rl.ipynb`

Based on the MDP formulation introduced in the paper, we train two independent DDQN, BCQ, IQL, and CQL models to discover and confirm dead-ends and identify treatments that may lead a patient toward these unfavorable states.

-----

### Step 2) Aggregate and Analyze the Results

#### Part 1) Results aggregation

Run `test_rl.ipynb`

This script takes the trained D- and R-Networks with the embedded states of the test set to aggregate results and identify potentially high-risk states and treatments.

#### Part 2) Generate Figures

Run `make_plot.py` to generate figures in the paper.

### BibTeX
```bibtex
@article{kim2026safety,
  title={Safety-aware explainable deep reinforcement learning for nephrotoxic medication management in critical care},
  author={Kim, Hyunwoo and Kim, Jong Hoon and Lee, Sung Woo and Kim, Su Jin and Han, Kap Su and Lee, Sijin and Song, Juhyun and Lee, Hyo Kyung},
  journal={Biomedical Signal Processing and Control},
  volume={112},
  pages={108577},
  year={2026},
  publisher={Elsevier}
}

