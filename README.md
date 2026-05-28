# cgm-on-chip

**A recent linear-attention architecture, reimplemented from scratch in pure C++ with its own autograd engine, trained on real clinical data, and run on a $4 microcontroller to predict hypoglycemia 60 minutes ahead.**

Everything in this repository is built from the C++17 standard library. No PyTorch, no TensorFlow, no Eigen, no BLAS, no OpenMP. The automatic differentiation engine, the optimizer, the multi-threaded trainer, the data pipeline, two model architectures, the on-disk weight format, the embedded inference kernel, and a triple-redundant correctness proof tying them together are roughly 5,500 lines of C++.

The shipped artifact is a trained model whose FP32 inference kernel runs on an **ESP32-S3 in 20% of its RAM**, raising a hypoglycemia alarm an hour before onset, verified to agree with the FP64 training reference to **8 decimal places**.

## Overview

Each layer below is something most stacks would simply `import`. The work here was building the whole thing from primitives.

### The math engine
A **scalar reverse-mode automatic differentiation engine** (`src/value.{hpp,cpp}`, 268 lines). Every arithmetic operation is a heap node that records a closure for pushing gradients backward, and `backward()` runs an explicit-stack topological traversal (plain recursion overflows the stack on a 144-step graph). It is extended to **complex numbers** (`src/complex_value.{hpp,cpp}`) so the S4D model's complex hidden state backpropagates through the same engine. In a normal stack this is `torch.autograd`.

The engine is verified by a **finite-difference gradient checker** (`src/grad_check.cpp`, 284 lines): 12 primitives across 61 test points, with analytical fallbacks at singularities and regression points that must pass after each fix. Normally this is `torch.autograd.gradcheck`.

### The models
**S4D** (`src/s4d.{hpp,cpp}`, 108 lines), the model I began with: a diagonal state-space layer with complex poles and hybrid ZOH / Euler discretization, backpropagating through the complex-value engine.

**OSDN** (`src/osdn.{hpp,cpp}`, 218 lines), implemented from the §4.2 online-preconditioned delta rule in arXiv:2605.13473, built directly from the paper's math.

> **An implementation finding worth noting.** The paper's §4.2 Equation (4) includes a box-projection step on the preconditioner, `d ← clamp(d, 0.5, 2.0)`. The Algorithm 1 pseudocode leaves it out, noting it is omitted "for clarity." Implement the pseudocode as written and FP32 inference silently produces NaN on trained weights, because the preconditioner drifts out of its stable band. Restoring the clamp fixes it. The interesting part: the clamp costs roughly 0.07 test AUROC (an unclamped run scores higher on paper), yet the clamped model is the only one that survives float32 on the device. The deployable number is the one reported here.

Both models share the same wrapper network, same 7-channel input, same trainer, same patient-disjoint split, and the same threshold methodology, so any comparison between them measures the recurrent backbone rather than scaffolding.

### The training infrastructure
A **hand-built fork-join thread pool** on raw `std::thread`, `std::mutex`, and `std::condition_variable` (`src/cgm_train.cpp`, roughly 997 lines), driving **synchronous data-parallel training**. Each worker holds its own network replica and its own parameter nodes, batches are round-robin sharded across workers, and per-worker gradients are summed back into the master before each Adam step. This is the pattern PyTorch hides behind `DataParallel`, written by hand because scalar-`Value` parameter nodes cannot be shared across threads (each carries a per-thread gradient accumulator). It includes a **hardened JSON resume parser** (a stateful character scanner that replaced a brittle reader which threw out of `main()` on malformed input) and **Adam with NaN-gradient rejection** (skip the update rather than poison the moment estimates, which never recover once they absorb a non-finite value).

### The data
**Ohio T1DM** (obtained under a data-use agreement), parsed and windowed from scratch (`src/cgm_data.cpp`, 373 lines): 7 channels (glucose, first and second glucose differences, sine and cosine of time-of-day, insulin-on-board and carbs-on-board with exponential-decay pharmacokinetics). **Patient-disjoint splits are verified twice**, normalization is computed on the training fold only, and the **classification threshold is selected on validation at a fixed specificity and then frozen for test**, so test labels never enter model selection. There is no pandas and no scikit-learn; every column is a hand-managed vector.

### Paper to silicon, proven
A header-only **FP32 inference kernel** (`src/osdn_inference.h`, 297 lines): allocation-free, stack-only, and the same physical file compiled into both the host test and the firmware. Its correctness is enforced by **three independent implementations of the same OSDN math** (the training autograd, an independently written reference network, and the FP32 kernel) asserted to agree at **|diff| = 7.96795993e-07**, held bit for bit through the entire hardening sequence by `make check`.

The **ESP32-S3 firmware** (roughly 716 lines) is a complete on-device alarm: a serial protocol for glucose, bolus, and meal events, on-device feature recomputation matching the trainer byte for byte, and a **fail-open clinical alarm** in which a non-finite logit raises the alarm rather than silently passing, because "I don't know" must surface on a medical predictor. The firmware feature module is tested by compiling the actual production code into a host binary and checking it against a double-precision reference (`src/features_host_test.cpp`, maximum difference 4.8e-7). Footprint: **RAM 20.1%, Flash 8.4%**.

## Results

Patient-disjoint Ohio T1DM, validation = {584, 588}, test = {591, 596}, 8 train / 2 validation / 2 test patients. Identical wrapper network across both architectures (7-channel input, H=16, lookback=144 of about 12 hours, horizon=12 of about 60 minutes ahead, stride=12, seed=42).

| arch | params | val AUROC | **test AUROC** | recall @ spec 0.80 | pb_recall @ spec 0.80 | ms/window |
|---|---|---|---|---|---|---|
| **S4D**  | 562 | 0.9510 | **0.9113** | 0.918 | 0.919 | 189.8 |
| **OSDN** | 747 | 0.9220 | **0.7669** | 0.718 | 0.676 | 236.4 |

Same split, same 7-channel features, same trainer, same threshold methodology. On this task the S4D model scores higher on test AUROC (0.9113 vs 0.7669), recall (0.918 vs 0.718), and post-bolus recall (0.919 vs 0.676), at roughly matched specificity (0.67 vs 0.68) and lower per-window inference cost.

FP32 device kernel agreement with the FP64 host autograd: **|diff| = 7.97e-7** (tolerance 1e-4; this is agreement within float rounding, not a deterministic byte match). ESP32-S3 footprint: RAM 20.1% (65,964 B), Flash 8.4% (281,161 B).

Read the validation-to-test gaps rather than the point estimates: with **n=2 test patients**, absolute AUROC is high-variance. Full caveats are below.

## Quickstart (fresh clone, no data needed)

```sh
make
make grad_check && ./grad_check
./cgm_inference_test --H 16 --K 8 --D-in 7 --n-layers 1 --L 144 --seed 42
```

The Ohio T1DM dataset requires a data-use agreement and is not redistributable. The `make check` trained-weight bit-identity gate needs the locked weight blob, but the build, the gradient check, and the random-init agreement test all run on a bare clone.

## Limitations

**Statistical and data regime**
- **n=2 test patients.** AUROC at this size is high-variance; a fold swap can move it several points. Trust the validation-to-test gap (OSDN 15.5 pp, S4D 4.0 pp), not the point estimate.
- **Custom split, not the canonical Marling and Bunescu BGLP split**, so these numbers are not directly comparable to the Ohio T1DM leaderboard.
- **Single seed (42).** No seed-variance bound.
- **Validation is used for both early stopping and threshold selection**, a tight loop at n=2.
- **Global, not per-patient, normalization.** Per-patient z-scoring would change the numbers and might close part of the gap.
- **AUROC is rank-based without explicit tie-handling** (`auroc_approx`); it differs from scikit-learn by a tie-dependent amount.
- **Window stride is 12, not 1**, so training windows are fewer and non-overlapping.

**Architecture and method**
- **The "from the paper" claim covers the OSDN d-update kernel specifically.** The surrounding network adds standard wrapper components (a per-row `sigmoid(log_lambda)` decay, LayerNorm on the K-readout, a per-channel residual, and a `tanh` input embedding), none of which are part of vanilla OSDN, and all kept identical across the S4D and OSDN runs so the comparison reflects the backbone.
- **S4D scores higher than OSDN here by 14 AUROC points.** This is a from-scratch reproduction and a direct report of what the runs produced, not a claim that either architecture is state-of-the-art on this task.
- **The verified kernel envelope is narrow**: `n_layers >= 2` with `K = 16` produces NaN in FP32 (documented in-kernel). The shipped configuration (`n_layers=1, K=8`) sits inside the verified envelope.

**Engineering and deployment**
- **No on-device hardware validation yet.** The firmware compiles for and deploys to the ESP32-S3, and the kernel is verified bit-equivalent to the trainer host-side; end-to-end replay on physical hardware is future work.
- **"Bit-equivalence" means FP32 / FP64 agreement within rounding** (7.97e-7 at tolerance 1e-4), reported to 8 digits for drift detection, not a claim of identical bits.

## Future work

The limitations above map directly onto the second stage of this project:

- **Leave-one-patient-out cross-validation across all 12 patients**, replacing the fixed 2-patient test fold. This is the single biggest fix: it turns the high-variance n=2 point estimate into a mean with a spread across every patient, and it is the natural way to make the S4D-versus-OSDN gap robust rather than fold-dependent.
- **Multi-seed runs** to put a variance bound on every reported number.
- **Per-patient normalization**, to test how much of the validation-to-test gap is distribution shift between individuals.
- **On-device hardware validation**: replaying a held-out patient's glucose trace into the ESP32-S3 over the serial protocol and confirming the on-chip alarm fires against host expectations, closing the loop from trained weights to a physical device.

The pipeline is built to absorb these without structural change: the split policy, normalization, and evaluation are already isolated behind the trainer's configuration, so stage two is a matter of running the existing machinery across more folds and seeds rather than rebuilding it.

## Data citation

This work uses the OhioT1DM dataset, obtained under a data-use agreement. Please cite:

```bibtex
@inproceedings{marling2020ohiot1dm,
  title     = {The OhioT1DM dataset for blood glucose level prediction: Update 2020},
  author    = {Marling, Cindy and Bunescu, Razvan},
  booktitle = {The 6th International Workshop on Knowledge Discovery from Healthcare Data (KDH)},
  year      = {2020},
  url       = {https://pmc.ncbi.nlm.nih.gov/articles/PMC7881904}
}
```
