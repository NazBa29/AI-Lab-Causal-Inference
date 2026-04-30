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

BASE_URL = "https://bana290-assignment4.netlify.app"


# =============================================================================
# PHASE 1: SCRAPE
# =============================================================================

# Copilot prompt: function to fetch the index page and return all hrefs that
# point to individual archive briefs (URLs containing '/briefs/')
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


# Copilot prompt: function to fetch a brief page and return its first table as
# a pandas DataFrame using pandas.read_html.
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

# Copilot prompt: helper to split combined team identifiers like
# 'SC001Anteater Cart' into the team id ('SC001') and team name ('Anteater Cart')
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


# Copilot prompt: parse messy distance strings like '~0.59km',
# 'Distance: 827 meters', '1,100 meters (sync route)', '534 m', '0.33 km from
# backbone'. Return distance in meters as float. Detect km vs m by string match.
def parse_distance(text):
    """Convert any messy distance string into meters (float)."""
    if pd.isna(text):
        return np.nan

    s = str(text).lower().replace(",", "")

    # Extract first floating-point number
    num_match = re.search(r"(\d+\.?\d*)", s)
    if not num_match:
        return np.nan
    value = float(num_match.group(1))

    # Detect unit: 'km' overrides 'm' since 'km' contains 'm'
    if "km" in s:
        return value * 1000.0   # convert km to meters
    elif "m" in s:
        return value
    else:
        return value  # assume meters as default


# Copilot prompt: extract the first floating-point number from any string.
# Used for AI_INTENSITY, INNOVATION_SCORE, ELIGIBILITY_SCORE columns that
# have prefixes/suffixes like 'Pitch rating = 81.0', '~57.1 model hrs',
# '61.0 / 100', 'Score: 80.4 pts', 'panel avg 89.5'.
def parse_number(text):
    """Extract the first float from any messy text."""
    if pd.isna(text):
        return np.nan
    match = re.search(r"(\d+\.?\d*)", str(text))
    return float(match.group(1)) if match else np.nan


# Copilot prompt: cap outliers in a numeric series using the IQR method
# (any value below Q1 - 1.5*IQR or above Q3 + 1.5*IQR gets clipped to the bound)
def cap_outliers_iqr(series):
    """Clip values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return series.clip(lower=lower, upper=upper)


def clean_infrastructure(path):
    """Load and clean the Infrastructure table (fiber-access-bulletin)."""
    # header=1 because row 0 of the raw CSV is just '0,1,2,3,4' placeholders
    df = pd.read_csv(path, header=1)
    df["TEAM_NAME"] = df["TEAM_REF"].apply(extract_team_name)
    df["TEAM_REF"] = df["TEAM_REF"].apply(extract_team_id)
    df["DISTANCE_TO_NODE"] = df["DISTANCE_TO_NODE"].apply(parse_distance)
    return df[["TEAM_REF", "TEAM_NAME", "HOME_BASE", "NETWORK_ZONE",
               "DISTANCE_TO_NODE"]]


def clean_innovation(path):
    """Load and clean the Innovation Metrics table (builder-metrics-ledger)."""
    df = pd.read_csv(path, header=1)
    df["TEAM_REF"] = df["TEAM_REF"].apply(extract_team_id)
    df["AI_INTENSITY"] = df["AI_INTENSITY"].apply(parse_number)
    df["INNOVATION_SCORE"] = df["INNOVATION_SCORE"].apply(parse_number)

    # Cap outliers (the assignment specifically calls out AI_INTENSITY and
    # INNOVATION_SCORE for IQR treatment)
    df["AI_INTENSITY"] = cap_outliers_iqr(df["AI_INTENSITY"])
    df["INNOVATION_SCORE"] = cap_outliers_iqr(df["INNOVATION_SCORE"])

    return df[["TEAM_REF", "TRACK", "AI_INTENSITY", "INNOVATION_SCORE"]]


def clean_grants(path):
    """Load and clean the Grant Committee table (anteater-fund-panel)."""
    df = pd.read_csv(path, header=1)
    df["TEAM_REF"] = df["TEAM_REF"].apply(extract_team_id)
    df["ELIGIBILITY_SCORE"] = df["ELIGIBILITY_SCORE"].apply(parse_number)

    # Treatment indicator: cutoff is 85 points (per assignment)
    df["TREATED"] = (df["ELIGIBILITY_SCORE"] >= 85).astype(int)

    return df[["TEAM_REF", "PITCH_TRACK", "ELIGIBILITY_SCORE", "TREATED"]]


def run_clean():
    """Clean each raw CSV, merge them, save to data/clean/master.csv."""
    os.makedirs("data/clean", exist_ok=True)

    print("Cleaning Infrastructure...")
    infra = clean_infrastructure("data/raw/fiber-access-bulletin.csv")
    print(f"  {len(infra)} rows, distance range "
          f"{infra['DISTANCE_TO_NODE'].min():.0f}-"
          f"{infra['DISTANCE_TO_NODE'].max():.0f} m")

    print("Cleaning Innovation Metrics...")
    innov = clean_innovation("data/raw/builder-metrics-ledger.csv")
    print(f"  {len(innov)} rows, AI intensity range "
          f"{innov['AI_INTENSITY'].min():.1f}-{innov['AI_INTENSITY'].max():.1f}")

    print("Cleaning Grants...")
    grants = clean_grants("data/raw/anteater-fund-panel.csv")
    n_treated = grants["TREATED"].sum()
    print(f"  {len(grants)} rows, {n_treated} teams above 85-pt cutoff")

    # Merge on TEAM_REF
    print("\nMerging on TEAM_REF...")
    master = infra.merge(innov, on="TEAM_REF", how="inner")
    master = master.merge(grants, on="TEAM_REF", how="inner")
    print(f"  Final master table: {len(master)} rows, {len(master.columns)} cols")
    print(f"  Columns: {list(master.columns)}")

    out_path = "data/clean/master.csv"
    master.to_csv(out_path, index=False)
    print(f"  Saved -> {out_path}")

    # Summary stats so we can sanity-check the cleaning worked
    print("\nSummary statistics:")
    print(master[["DISTANCE_TO_NODE", "AI_INTENSITY",
                  "INNOVATION_SCORE", "ELIGIBILITY_SCORE"]].describe().round(2))

    return master


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    run_scrape()
    print("\n" + "=" * 60 + "\n")
    run_clean()
