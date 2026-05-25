# Figure Generation

This folder contains standalone figure code and generated images for the SMC-QA project.

`Ref/` was used only as a visual style reference. No runtime dependency points to `Ref/`, so this folder can regenerate all figures after `Ref/` is deleted.

## Files

- `generate_figures.py`: reads experiment JSON files from `../results/` and writes figures.
- `output/`: generated `.png` and `.pdf` figures.

## Requirements

Use the project conda environment:

```bash
conda activate ml_project_repro
python -m pip install matplotlib
```

## Generate

From the project root:

```bash
python Figure/generate_figures.py
```

or:

```bash
conda run -n ml_project_repro python Figure/generate_figures.py
```

## Generated Figures

- `fig01_oracle_main_results`: Oracle comparison across six methods.
- `fig02_retrieval_tuning`: Base SMC-QA vs retrieval-tuned SMC-QA vs frequency promotion.
- `fig03_s_cleaned_tuning`: Full-haystack S-Cleaned comparison.
- `fig04_question_type_delta`: Per-question-type change from Base SMC-QA to retrieval-tuned SMC-QA.
- `fig05_ltm_quality`: LTM size and evidence-quality diagnostics.
