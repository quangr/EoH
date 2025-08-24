# COMP64804 Research Project: Re-evaluating LLM-based Heuristic Search – A Case Study on the 3D Packing Problem

This repository contains the code for the COMP64804 research project report: *Re-evaluating LLM-based Heuristic Search: A Case Study on the 3D Packing Problem*.

---

## 1. Install the Library

To use the code, run:

```bash
cd eoh
pip install -e .
```

This will install the library in editable mode.

---

## 2. Download the Dataset

Download the **SSSCSP** instances from [Leandro Coelho’s container loading problems](https://www.leandro-coelho.com/container-loading-problems/) and place them in the `data/ssscsp/INSTANCES/` directory, for example:

```
data/ssscsp/INSTANCES/Iva1.txt
data/ssscsp/INSTANCES/Iva2.txt
...
```

---

## 3. Start EoH

Run the following command to start the EoH algorithm:

```bash
python start_eoh.py --problem ssscsp --exp_output_path output/10_0 --eva_timeout 10
```

### Problem Types

There are four types of problems:

1. **ssscsp** – only basic non-overlap constraints
2. **ssscsp\_support** – includes support constraints
3. **ssscsp\_separation** – includes separation constraints
4. **ssscsp\_all** – includes both support and separation constraints

---

## 4. Run Random Search

You can run a random search as follows:

1. Add a scoring function by inheriting from the `BaseAlgorithm` class in `test_random.py`.
2. Use `plot_pattern.ipynb` to visualize results from the set partition solver.

---

## 5. LLM-generated Heuristics

The original LLM-generated heuristics are listed in the following directories:

* `ssscsp_all`
* `ssscsp_separation`
* `ssscsp_support`
* `paper_output`

---

## 6. Reproducing the Original EoH Baseline

If you want to reproduce the original EoH baseline, switch to the `baseline` branch:

```bash
git checkout baseline
