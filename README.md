# Learned Feature Insertion for Quantum Circuits

This repository demonstrates a classical policy for building quantum data-encoding circuits one block at a time. It accompanies the manuscript *Learned Feature Insertion for Quantum Circuits* for the QC+AI 2027 workshop. The policy uses labeled construction examples and the response of the partial circuit to guide its next insertion. Classification is performed with a calibrated readout of the completed circuit.

## Run a local check

Use Python 3.11. The example runs on CPU and generates its own synthetic data.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python demo.py
```

The demonstration samples eight complete proposals from a fresh initialization, selects by validation loss and evaluates the selected circuit on held-out examples. It checks the architecture and block count, the operation budget, feature uniqueness, action masks, finite gradients, and whether changing the validation examples leaves the descriptors and sampled trajectory unchanged under the same seed. The printed BCE values are for this demonstration.

## Mechanism

1. The register starts in the four-qubit zero state. Each step computes 12 descriptors per unused feature and a shared 123-component circuit summary from construction examples.
2. The 21,483-parameter MLP samples stop or selects a feature, one of six block families, a valid placement and discrete rotation weights. Masks enforce the 16-operation budget. The 36 placements are defined on a four-qubit chain.
3. The selected block is applied directly to the simulated state, and the feature is removed from the candidate set. One complete trajectory, including an immediate stop, counts as one proposal. Intermediate state evaluations for descriptors are counted separately.
4. The joint zero-state probability is converted to a class probability using slope-penalized logistic calibration fitted on construction data. Complete proposals are ranked on validation data. The final evaluation set is used after selection.

In the experiments, the MLP was trained across tasks by maximizing the log probability of decisions in high-reward episodes. Reward was the reduction in validation BCE relative to the construction prevalence. A separate set of tasks was used to select checkpoints. In Study A, the acceptance rule could return the prevalence predictor if the best circuit gained at most 0.01 validation BCE. The primary Study B analysis selected the lowest-validation-BCE circuit without this rule. The demonstration exercises the mechanism and a training gradient. The original multi-task training and evaluation drivers and frozen checkpoints are outside this portable release.

## Recorded results

Gain is reference BCE minus policy BCE, so positive values favor the policy. Intervals are paired 95% task-bootstrap intervals. Each method has the indicated number of proposals in the random and earlier-policy comparisons.

| Study and tasks | Reference | Proposals | BCE gain | 95% interval |
| --- | --- | ---: | ---: | --- |
| A, 16 with signal | Random | 1 | 0.0193 | [0.0050, 0.0368] |
| A, 16 with signal | Random | 16 | 0.0246 | [0.0092, 0.0433] |
| A, 16 with signal | Untrained policy | 16 | 0.0233 | [0.0111, 0.0358] |
| A, 16 with signal | Permuted feature descriptors | 16 | 0.0239 | [0.0098, 0.0399] |
| B, 36 within training conditions | Earlier policy | 16 | 0.0052 | [0.0021, 0.0085] |
| B, 60 with prescribed shifts | Earlier policy | 16 | 0.0065 | [0.0029, 0.0106] |
| B, 36 within training conditions | Random | 1 | 0.0164 | [0.0076, 0.0261] |
| B, 60 with prescribed shifts | Random | 1 | 0.0309 | [0.0204, 0.0423] |

Study A's first null-task test fell outside the allowed tolerance. In the subsequent test, both variants, with and without the acceptance rule, met the tolerance. Study B increased training-set size and diversity together. Its distribution shifts remain within the generating library. The comparison with the greedy constructor used a different search procedure and computational budget. Recorded CPU times are for simulation and separate training, descriptor computation and proposal use.

The JSON files in [results/](results/) are the recorded synthetic study outputs used to prepare the manuscript. They are separate from the example output. Study B includes the grouped results, secondary acceptance analysis, and costs. These experiments concern simulated circuits on synthetic data.

## Files and provenance

- `policy.py` adapts the factored sampling and episode likelihood from the research policy.
- `blocks.py` adapts the local research library to the four-qubit chain described in the manuscript. The exact historical block-index mapping has not been checked against original frozen checkpoints.
- `descriptors.py` adapts the research descriptors, excluding the feature index from policy input and defining AUC as 0.5 for single-class construction samples.
- `mechanism.py` supplies a portable statevector environment and fits the same slope-penalized objective with SciPy BFGS.
- `demo.py` supplies a separate synthetic example for checking the mechanism. Reproducing the recorded estimates would require the original drivers, exact library mapping and frozen models.
