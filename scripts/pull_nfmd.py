# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas", "pyarrow"]
# ///
"""Field fuel moisture samples from the National Fuel Moisture Database, now the FEMS Field Sample
Database (fems.fs2c.usda.gov). Both calls are the public ones the FEMS web app makes, no login:

- sites: GraphQL `getSites` at /fuelmodel/apis/graphql (needs `fuelSampleMetadataInput: {}`)
- samples: CSV at /fuelmodel/sample/download (the app's "Download Data" / "Copy Data Link")

Writes data/nfmd/sites.parquet (every site in the country) and data/nfmd/samples_<state>.parquet.

    uv run scripts/pull_nfmd.py [--state California] [--start 2015-01-01]
"""
import argparse
import io
import json
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "nfmd"
BASE = "https://fems.fs2c.usda.gov/fuelmodel"

SITES_Q = """query S($returnAll: Boolean!, $perPage: Int, $fuelSampleMetadataInput: FuelSampleMetadataInputDTO) {
 getSites(fuelSampleMetadataInput: $fuelSampleMetadataInput returnAll: $returnAll perPage: $perPage) {
  sites { siteId siteName siteStatus active longitude latitude elevation slope aspect stateName groupName areaName jurisdiction } } }"""


def sites():
    body = json.dumps({"query": SITES_Q, "variables": {"returnAll": False, "perPage": 300000, "fuelSampleMetadataInput": {}}}).encode()
    r = urllib.request.Request(f"{BASE}/apis/graphql", data=body, headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(r, timeout=300))
    return pd.DataFrame(d["data"]["getSites"]["sites"])


def samples(site_ids, start, end):
    u = (f"{BASE}/sample/download?returnAll=true&responseFormat=csv&siteId={','.join(map(str, site_ids))}&sampleId="
         f"&startDate={start}T00:00:00.000Z&endDate={end}T00:00:00.000Z&filterByFuelId=&filterByStatus=Submitted"
         "&filterByCategory=All&filterBySubCategory=All&filterByMethod=All&sortBy=&sortOrder=")
    b = urllib.request.urlopen(u, timeout=900).read()
    df = pd.read_csv(io.BytesIO(b))
    df.columns = ["sample_id", "time", "site_name", "site_id", "fuel_type", "category", "sub_category", "method", "value", "status"]
    df["time"] = pd.to_datetime(df.time, utc=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="California")
    ap.add_argument("--start", default="2015-01-01")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    s = sites()
    s.to_parquet(OUT / "sites.parquet", index=False)
    ids = s.loc[s.stateName == a.state, "siteId"].tolist()
    print(f"{len(s)} sites, {len(ids)} in {a.state}")
    df = samples(ids, a.start, date.today().isoformat())
    tag = a.state.lower().replace(" ", "_")
    df.to_parquet(OUT / f"samples_{tag}.parquet", index=False)
    print(f"{len(df)} samples, {df.site_id.nunique()} sites, {df.time.min():%Y-%m-%d} to {df.time.max():%Y-%m-%d}")


if __name__ == "__main__":
    main()
