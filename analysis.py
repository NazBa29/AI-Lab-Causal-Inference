# =============================================================================
# AI Lab: Causal Inference Assignment
# Author: Ilnaz Bagheri
# =============================================================================

# -----------------------------------------------------------------------------
# PHASE 1: SCRAPE
# Goal: Extract 3 tables (Infrastructure, Innovation, Grants) from the
# Smart Campus Incubator Archive and save each as a raw CSV.
# -----------------------------------------------------------------------------

# Copilot prompt: import libraries for HTTP requests, HTML parsing, dataframes,
# and StringIO so pandas.read_html can read raw HTML text.
import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
from io import StringIO

# Base URL of the archive portal
BASE_URL = "https://bana290-assignment4.netlify.app"


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
# a pandas DataFrame. Wrap the HTML text in StringIO for compatibility with
# the newest pandas API.
def scrape_table(url):
    """Download a brief page and return the data table as a DataFrame."""
    response = requests.get(url)
    response.raise_for_status()

    tables = pd.read_html(StringIO(response.text))
    df = tables[0]
    return df


# Copilot prompt: main routine. Discover all brief links, scrape each table,
# and save the raw output as CSVs in data/raw/.
if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)

    links = get_archive_links(BASE_URL)
    print(f"Found {len(links)} archive links")

    for url in links:
        slug = url.rstrip("/").split("/")[-1]
        print(f"\nScraping: {slug}")

        df = scrape_table(url)
        print(f"  Got {len(df)} rows, {len(df.columns)} columns")
        print(f"  Columns: {list(df.columns)}")

        out_path = f"data/raw/{slug}.csv"
        df.to_csv(out_path, index=False)
        print(f"  Saved -> {out_path}")
