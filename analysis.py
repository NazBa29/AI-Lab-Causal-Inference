# =============================================================================
# AI Lab: Causal Inference Assignment
# Author: Ilnaz Bagheri
# =============================================================================

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import os
import re
from io import StringIO

import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from linearmodels.iv import IV2SLS
from scipy.stats import gaussian_kde

BASE_URL = "https://bana290-assignment4.netlify.app"


# =============================================================================
# PHASE 1: SCRAPE
# =============================================================================

def get_archive_links(base_url):
    """Scrape the homepage and return absolute URLs for each archive page."""
    response = requests.get(base_url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/briefs/" in href:
            full_url = href if href.startswith("http") else base_url + href
            links.append(full_url)
    return links


def scrape_table(url):
    """Download a brief page and return the data table as a DataFrame."""
    response = requests.get(url)
    response.raise_for_status()
    tables = pd.read_html(StringIO(response.text))
    return tables[0]


def run_scrape():
    """Discover all brief links, scrape each table, save to data/raw/."""
    os.makedirs("data/raw", exist_ok=True)
    links = get_archive_links(BASE_URL)
    print(f"Found {len(links)} archive links")
    for url in links:
        slug = url.rstrip("/").split("/")[-1]
        df = scrape_table(url)
        out_path = f"data/raw/{slug}.csv"
        df.to_csv(out_path, index=False)
        print(f"  Saved {len(df)} rows -> {out_path}")


# =============================================================================
# PHASE 2: CLEAN
# =============================================================================

def extract_team_id(text):
    """Extract the SC#### prefix from 'SC001Anteater Cart'."""
    if pd.isna(text):
        return None
    match = re.match(r"(SC\d+)", str(text))
    return match.group(1) if match else None


def extract_team_name(text):
    """Extract the team name from 'SC001Anteater Cart'."""
    if pd.isna(text):
        return None
    match = re.match(r"SC\d+(.*)", str(text))
    return match.group(1).strip() if match else str(text).strip()


def parse_distance(text):
    """Convert any messy distance string into meters (float)."""
    if pd.isna(text):
        return np.nan
    s = str(text).lower().replace(",", "")
    num_match = re.search(r"(\d+\.?\d*)", s)
    if not num_match:
        return np.nan
    value = float(num_match.group(1))
    if "km" in s:
        return value * 1000.0
    elif "m" in s:
        return value
    else:
        return value


def parse_number(text):
    """Extract the first float from any messy text."""
    if pd.isna(text):
        return np.nan
    match = re.search(r"(\d+\.?\d*)", str(text))
    return float(match.group(1)) if match else np.nan


def cap_outliers_iqr(series):
    """Clip values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return series.clip(lower=lower, upper=upper)


def clean_infrastructure(path):
    df = pd.read_csv(path, header=1)
    df["TEAM_NAME"] = df["TEAM_REF"].apply(extract_team_name)
    df["TEAM_REF"] = df["TEAM_REF"].apply(extract_team_id)
    df["DISTANCE_TO_NODE"] = df["DISTANCE_TO_NODE"].apply(parse_distance)
    return df[["TEAM_REF", "TEAM_NAME", "HOME_BASE", "NETWORK_ZONE", "DISTANCE_TO_NODE"]]


def clean_innovation(path):
    df = pd.read_csv(path, header=1)
    df["TEAM_REF"] = df["TEAM_REF"].apply(extract_team_id)
    df["AI_INTENSITY"] = df["AI_INTENSITY"].apply(parse_number)
    df["INNOVATION_SCORE"] = df["INNOVATION_SCORE"].apply(parse_number)
    df["AI_INTENSITY"] = cap_outliers_iqr(df["AI_INTENSITY"])
    df["INNOVATION_SCORE"] = cap_outliers_iqr(df["INNOVATION_SCORE"])
    return df[["TEAM_REF", "TRACK", "AI_INTENSITY", "INNOVATION_SCORE"]]


def clean_grants(path):
    df = pd.read_csv(path, header=1)
    df["TEAM_REF"] = df["TEAM_REF"].apply(extract_team_id)
    df["ELIGIBILITY_SCORE"] = df["ELIGIBILITY_SCORE"].apply(parse_number)
    df["TREATED"] = (df["ELIGIBILITY_SCORE"] >= 85).astype(int)
    return df[["TEAM_REF", "PITCH_TRACK", "ELIGIBILITY_SCORE", "TREATED"]]


def run_clean():
    """Clean each raw CSV, merge them, save to data/clean/master.csv."""
    os.makedirs("data/clean", exist_ok=True)

    print("Cleaning Infrastructure...")
    infra = clean_infrastructure("data/raw/fiber-access-bulletin.csv")

    print("Cleaning Innovation Metrics...")
    innov = clean_innovation("data/raw/builder-metrics-ledger.csv")

    print("Cleaning Grants...")
    grants = clean_grants("data/raw/anteater-fund-panel.csv")

    master = infra.merge(innov, on="TEAM_REF", how="inner")
    master = master.merge(grants, on="TEAM_REF", how="inner")
    print(f"  Final master table: {len(master)} rows, {len(master.columns)} cols")

    out_path = "data/clean/master.csv"
    master.to_csv(out_path, index=False)
    print(f"  Saved -> {out_path}")
    return master


# =============================================================================
# PHASE 3: ANALYZE
# =============================================================================

# Copilot prompt: run a naive OLS regression of innovation on AI intensity.
# This is the biased baseline because innovative teams may use more AI for
# reasons unrelated to the causal effect we are after.
def run_naive_ols(df):
    """OLS: INNOVATION_SCORE ~ AI_INTENSITY (biased baseline)."""
    X = sm.add_constant(df["AI_INTENSITY"])
    y = df["INNOVATION_SCORE"]
    model = sm.OLS(y, X).fit()
    return model


# Copilot prompt: run the first-stage regression of AI intensity on the
# distance instrument. Report the F-statistic to test instrument relevance.
# Rule of thumb: F >= 10 means the instrument is "strong."
def run_first_stage(df):
    """First stage: AI_INTENSITY ~ DISTANCE_TO_NODE."""
    X = sm.add_constant(df["DISTANCE_TO_NODE"])
    y = df["AI_INTENSITY"]
    model = sm.OLS(y, X).fit()
    return model


# Copilot prompt: estimate 2SLS using linearmodels.IV2SLS.
# Endogenous: AI_INTENSITY. Instrument: DISTANCE_TO_NODE.
def run_2sls(df):
    """Second stage / IV: INNOVATION ~ AI_INTENSITY, instrumented by DISTANCE."""
    df2 = df.copy()
    df2["const"] = 1.0
    model = IV2SLS(
        dependent=df2["INNOVATION_SCORE"],
        exog=df2["const"],
        endog=df2["AI_INTENSITY"],
        instruments=df2["DISTANCE_TO_NODE"],
    ).fit(cov_type="robust")
    return model


# Copilot prompt: run a sharp RDD using the 85-point eligibility cutoff.
# Center the running variable around the cutoff and include treatment dummy.
# INNOVATION = a + b*TREATED + c*(ELIGIBILITY-85) + d*TREATED*(ELIGIBILITY-85)
def run_rdd(df, cutoff=85):
    """Sharp RDD with linear trends on each side of the cutoff."""
    df2 = df.copy()
    df2["RUNNING"] = df2["ELIGIBILITY_SCORE"] - cutoff
    df2["TREATED_X_RUN"] = df2["TREATED"] * df2["RUNNING"]
    X = sm.add_constant(df2[["TREATED", "RUNNING", "TREATED_X_RUN"]])
    y = df2["INNOVATION_SCORE"]
    model = sm.OLS(y, X).fit()
    return model


def plot_rdd(df, cutoff=85, out_path="figures/rdd_scatter.png"):
    """Scatter of innovation vs eligibility with separate fits each side."""
    fig, ax = plt.subplots(figsize=(8, 5))

    left = df[df["ELIGIBILITY_SCORE"] < cutoff]
    right = df[df["ELIGIBILITY_SCORE"] >= cutoff]

    ax.scatter(left["ELIGIBILITY_SCORE"], left["INNOVATION_SCORE"],
               color="steelblue", alpha=0.7, label="Below cutoff (control)")
    ax.scatter(right["ELIGIBILITY_SCORE"], right["INNOVATION_SCORE"],
               color="firebrick", alpha=0.7, label="Above cutoff (treated)")

    # Linear fit on each side
    for grp, color in [(left, "steelblue"), (right, "firebrick")]:
        if len(grp) > 1:
            x = grp["ELIGIBILITY_SCORE"].values
            y = grp["INNOVATION_SCORE"].values
            coef = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, np.polyval(coef, xs), color=color, lw=2)

    ax.axvline(cutoff, color="black", linestyle="--", lw=1.5,
               label=f"Cutoff = {cutoff}")
    ax.set_xlabel("Eligibility Score (Running Variable)")
    ax.set_ylabel("Innovation Score")
    ax.set_title("Regression Discontinuity at the 85-Point Grant Cutoff")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved -> {out_path}")


def plot_density(df, cutoff=85, out_path="figures/density_check.png"):
    """McCrary-style density plot of the running variable around the cutoff."""
    fig, ax = plt.subplots(figsize=(8, 5))
    scores = df["ELIGIBILITY_SCORE"].values

    # Histogram + KDE
    ax.hist(scores, bins=15, density=True, alpha=0.5, color="lightgray",
            edgecolor="black", label="Histogram")
    kde = gaussian_kde(scores)
    xs = np.linspace(scores.min(), scores.max(), 200)
    ax.plot(xs, kde(xs), color="navy", lw=2, label="Density (KDE)")
    ax.axvline(cutoff, color="red", linestyle="--", lw=1.5,
               label=f"Cutoff = {cutoff}")

    ax.set_xlabel("Eligibility Score")
    ax.set_ylabel("Density")
    ax.set_title("Density of Eligibility Scores (Manipulation Check)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved -> {out_path}")


def run_analyze():
    """Phase 3: OLS, IV/2SLS, RDD, plots, and saved results file."""
    os.makedirs("figures", exist_ok=True)
    df = pd.read_csv("data/clean/master.csv")

    print("\n--- Naive OLS ---")
    ols = run_naive_ols(df)
    print(f"  AI_INTENSITY coef: {ols.params['AI_INTENSITY']:.4f}")
    print(f"  R-squared: {ols.rsquared:.4f}")

    print("\n--- First Stage (instrument relevance) ---")
    first = run_first_stage(df)
    f_stat = first.fvalue
    print(f"  DISTANCE_TO_NODE coef: {first.params['DISTANCE_TO_NODE']:.6f}")
    print(f"  F-statistic: {f_stat:.2f}  "
          f"({'STRONG' if f_stat >= 10 else 'WEAK'} instrument)")

    print("\n--- 2SLS (causal estimate) ---")
    iv = run_2sls(df)
    ai_coef = iv.params["AI_INTENSITY"]
    print(f"  AI_INTENSITY coef (causal): {ai_coef:.4f}")

    print("\n--- RDD (sharp design at 85) ---")
    rdd = run_rdd(df)
    jump = rdd.params["TREATED"]
    pval = rdd.pvalues["TREATED"]
    print(f"  Estimated jump at cutoff: {jump:+.3f} (p = {pval:.3f})")

    print("\n--- Plots ---")
    plot_rdd(df)
    plot_density(df)

    # Save full text summaries for the LaTeX writeup
    out_path = "data/clean/regression_results.txt"
    with open(out_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("NAIVE OLS\n")
        f.write("=" * 70 + "\n")
        f.write(str(ols.summary()) + "\n\n")

        f.write("=" * 70 + "\n")
        f.write("FIRST STAGE (AI_INTENSITY ~ DISTANCE_TO_NODE)\n")
        f.write("=" * 70 + "\n")
        f.write(str(first.summary()) + "\n")
        f.write(f"\nFirst-stage F-statistic: {f_stat:.2f}\n\n")

        f.write("=" * 70 + "\n")
        f.write("2SLS (instrumental variable)\n")
        f.write("=" * 70 + "\n")
        f.write(str(iv.summary) + "\n\n")

        f.write("=" * 70 + "\n")
        f.write("RDD at cutoff = 85\n")
        f.write("=" * 70 + "\n")
        f.write(str(rdd.summary()) + "\n")
    print(f"\n  Saved full results -> {out_path}")

    print("\n--- Comparison ---")
    print(f"  Naive OLS coef:  {ols.params['AI_INTENSITY']:+.4f}")
    print(f"  2SLS / IV coef:  {ai_coef:+.4f}")
    print(f"  RDD jump (85):   {jump:+.4f}")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    run_scrape()
    print("\n" + "=" * 60 + "\n")
    run_clean()
    print("\n" + "=" * 60 + "\n")
    run_analyze()
