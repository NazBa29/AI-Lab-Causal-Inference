# AI Lab: Causal Inference Assignment

Investigating the causal effect of AI intensity on firm innovation using:
- **Instrumental Variable (IV) / 2SLS** with distance to fiber backbone as the instrument
- **Regression Discontinuity Design (RDD)** at the 85-point grant eligibility cutoff

## Author
Ilnaz (Nazjoon) Bagheri — UC Irvine MSBA

## Structure
- `analysis.py` — full pipeline: scrape, clean, analyze
- `interpretation.tex` — written interpretation of results
- `data/raw/` — scraped CSVs (untouched)
- `data/clean/` — merged & cleaned master dataset
- `figures/` — RDD plots and density tests

## Setup
```bash
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```