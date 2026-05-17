"""
data_acquisition.py – HiVIS Image Download + USGS NWIS Data Fetcher
=====================================================================
Provides:
  - SITE_CATALOG          : known USGS sites with NIMS camId, nwisId, and default ROI
  - USGS_PARAMETERS       : common parameter codes + labels
  - fetch_camera_info()   : get camera record(s) for a site from NIMS API
  - list_hivis_images()   : list available images in a date range
  - download_images()     : download images to local folder with progress
  - fetch_usgs_sensor()   : fetch instantaneous values from waterservices API
  - build_labels_csv()    : join image timestamps with sensor data -> CSV
  - run_acquisition()     : full pipeline in one call

NIMS API (verified working):
  GET https://api.waterdata.usgs.gov/nims/v0/cameras?site_no={nwisId}
  GET https://api.waterdata.usgs.gov/nims/v0/listFiles?camId={camId}&startDT=...&endDT=...
  Images: {overlayDir}{filename}

Filename format: {camId}___YYYY-MM-DDTHH-MM-SSZ.jpg
"""

import os
import time
import json
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Callable, List, Tuple, Dict

# ---------------------------------------------------------------------------
# Site Catalog  (verified from live NIMS API, May 2026)
# nwisId  = the ID used by NIMS (may differ from classic USGS site number)
# site_no = classic 8-digit USGS site number used by NWIS waterservices
# ---------------------------------------------------------------------------

SITE_CATALOG: Dict[str, dict] = {
    "VA Little Neck Creek (Pinewood Rd, VA Beach)": {
        "nwisId":          "0204295505",
        "site_no":         "0204295505",
        "camId":           "VA_Little_Neck_Creek_at_Pinewood_Road_at_Virginia_Beach",
        "default_params":  ["62620", "00045"],   # estuary elevation + precipitation
        "roi":             (951, 0, 1136, 1920),
    },
    "VA Mechumps Creek (Hill Carter Pkwy, Ashland)": {
        "nwisId":          "0167300055",
        "site_no":         "0167300055",
        "camId":           "VA_Mechumps_Creek_at_Hill_Carter_Parkway_at_Ashland",
        "default_params":  ["00065", "00045"],
        "roi":             (400, 0, 900, 1920),
    },
    "VA Bailey Creek (Dock Landing Rd, Chesapeake)": {
        "nwisId":          "0204288905",
        "site_no":         "0204288905",
        "camId":           "VA_Bailey_Creek_at_Dock_Landing_Road_at_Chesapeake",
        "default_params":  ["62620", "00045"],
        "roi":             (400, 0, 900, 1920),
    },
    "VA James River at Buchanan": {
        "nwisId":          "02019500",
        "site_no":         "02019500",
        "camId":           "VA_James_River_at_Buchanan",
        "default_params":  ["00065", "00060"],
        "roi":             (400, 0, 900, 1920),
    },
    "VA Blackwater River at Franklin": {
        "nwisId":          "02050000",
        "site_no":         "02050000",
        "camId":           "VA_Blackwater_River_at_HWY_58_at_Franklin",
        "default_params":  ["00065", "00045"],
        "roi":             (400, 0, 900, 1920),
    },
    "VA Conveyance Channel (Ramsgate Ln, Great Bridge)": {
        "nwisId":          "0204309906",
        "site_no":         "0204309906",
        "camId":           "VA_Conveyance_Channel_at_Ramsgate_Lane_near_Great_Bridge",
        "default_params":  ["62620", "00045"],
        "roi":             (400, 0, 900, 1920),
    },
    "NY Neversink River at Godeffroy": {
        "nwisId":          "01435000",
        "site_no":         "01435000",
        "camId":           None,
        "default_params":  ["00065", "00045"],
        "roi":             (400, 0, 900, 1920),
    },
    "CA Sacramento River at Freeport": {
        "nwisId":          "11447650",
        "site_no":         "11447650",
        "camId":           None,
        "default_params":  ["00065", "00045"],
        "roi":             (400, 0, 900, 1920),
    },
    "SC Congaree River at Columbia": {
        "nwisId":          "02169500",
        "site_no":         "02169500",
        "camId":           None,
        "default_params":  ["00065", "00060"],
        "roi":             (400, 0, 900, 1920),
    },
    "NC McMullen Creek at Charlotte": {
        "nwisId":          "02146300",
        "site_no":         "02146300",
        "camId":           None,
        "default_params":  ["00065", "00045"],
        "roi":             (400, 0, 900, 1920),
    },
}

# ---------------------------------------------------------------------------
# USGS Parameter Codes
# ---------------------------------------------------------------------------

USGS_PARAMETERS: Dict[str, str] = {
    "00065": "Gage height (ft)",
    "00045": "Precipitation (in)",
    "00060": "Discharge (cfs)",
    "62620": "Estuary/ocean water surface elevation NAVD88 (ft)",
    "00010": "Water temperature (°C)",
    "00300": "Dissolved oxygen (mg/L)",
}

WATER_LEVEL_TARGET_CODES = ("62620", "00065", "00060")

# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

NIMS_BASE    = "https://api.waterdata.usgs.gov/nims/v0"
NWIS_IV_BASE = "https://waterservices.usgs.gov/nwis/iv/"

# ---------------------------------------------------------------------------
# NIMS – Camera discovery
# ---------------------------------------------------------------------------

def fetch_camera_info(site_info: dict, api_key: Optional[str] = None,
                      log_cb: Optional[Callable] = None) -> Optional[dict]:
    """
    Return the NIMS camera record for a site.
    Strategy:
      1. If camId is known, fetch full camera list and find by camId.
      2. Otherwise search the full list by nwisId.
    NOTE: the NIMS site_no query param does NOT actually filter —
          it always returns all cameras. We filter locally.
    """
    headers  = {"X-API-Key": api_key} if api_key else {}
    cam_id   = site_info.get("camId")
    nwis_id  = site_info.get("nwisId", "")

    if log_cb:
        log_cb(f"  Fetching full NIMS camera list ...")

    try:
        resp = requests.get(f"{NIMS_BASE}/cameras", headers=headers, timeout=30)
        resp.raise_for_status()
        all_cams = resp.json()
        if not isinstance(all_cams, list):
            all_cams = all_cams.get("cameras", [])
    except Exception as e:
        if log_cb:
            log_cb(f"  Error fetching camera list: {e}")
        raise

    # Match by camId first (exact, fast)
    if cam_id:
        matches = [c for c in all_cams if c.get("camId") == cam_id]
        if matches:
            if log_cb:
                log_cb(f"  Matched by camId: {cam_id}")
            return matches[0]

    # Fall back to nwisId match
    if nwis_id:
        matches = [c for c in all_cams if c.get("nwisId") == nwis_id]
        if matches:
            if log_cb:
                log_cb(f"  Matched by nwisId: {nwis_id} -> {matches[0].get('camId')}")
            return matches[0]

    if log_cb:
        log_cb(f"  No camera found for camId={cam_id}, nwisId={nwis_id}")
    return None


# ---------------------------------------------------------------------------
# NIMS – List available images
# ---------------------------------------------------------------------------

def list_hivis_images(cam: dict, start_date: str, end_date: str,
                      max_images: int = 200,
                      log_cb: Optional[Callable] = None) -> List[dict]:
    """
    List available images for a camera in the given date range.
    cam        : camera dict from fetch_camera_info()
    start_date : 'YYYY-MM-DD'
    end_date   : 'YYYY-MM-DD'
    Returns list of dicts: {filename, timestamp, url}
    """
    cam_id      = cam.get("camId")
    overlay_dir = cam.get("overlayDir", "")

    if not cam_id:
        if log_cb:
            log_cb("  Cannot list images – no camId in camera record.")
        return []

    if log_cb:
        log_cb(f"  Listing images for {cam_id} from {start_date} to {end_date} ...")

    try:
        resp = requests.get(
            f"{NIMS_BASE}/listFiles",
            params={
                "camId":  cam_id,
                "after":  start_date,   # correct param per NIMS OpenAPI spec
                "before": end_date,
                "limit":  max_images,
            },
            timeout=60,
        )
        resp.raise_for_status()
        raw = resp.json()
    except Exception as e:
        if log_cb:
            log_cb(f"  Error listing images: {e}")
        raise

    # Response is a plain list of filename strings
    filenames = raw if isinstance(raw, list) else raw.get("files", raw.get("items", []))

    result = []
    for fname in filenames[:max_images]:
        # fname is a plain string like:
        # "VA_Little_Neck_Creek___2025-01-01T12-00-00Z.jpg"
        if not isinstance(fname, str):
            fname = fname.get("filename", str(fname))
        ts  = _parse_timestamp_from_filename(fname)
        url = overlay_dir.rstrip("/") + "/" + fname if overlay_dir else ""
        result.append({"filename": fname, "timestamp": ts, "url": url})

    if log_cb:
        log_cb(f"  {len(result)} images listed.")
    return result


def _parse_timestamp_from_filename(filename: str) -> Optional[datetime]:
    """
    Extract datetime from HiVIS filename.
    Format: {camId}___YYYY-MM-DDTHH-MM-SSZ.jpg
    """
    base = Path(filename).stem          # strip .jpg
    # find the triple-underscore separator
    sep = "___"
    idx = base.find(sep)
    if idx != -1:
        ts_part = base[idx + len(sep):]  # e.g. "2025-01-01T12-00-00Z"
    else:
        ts_part = base

    # normalise: replace hyphens in time part with colons
    # "2025-01-01T12-00-00Z" -> "2025-01-01T12:00:00Z"
    try:
        if "T" in ts_part:
            date_p, time_p = ts_part.split("T", 1)
            time_p = time_p.rstrip("Z").replace("-", ":") + "+00:00"
            return datetime.fromisoformat(f"{date_p}T{time_p}")
        else:
            return datetime.strptime(ts_part, "%Y-%m-%d_%H-%M-%S")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Download images
# ---------------------------------------------------------------------------

def download_images(image_list: List[dict], dest_dir: str,
                    api_key: Optional[str] = None,
                    log_cb: Optional[Callable] = None) -> List[dict]:
    """
    Download images to dest_dir.
    Returns same list with 'local_path' key added to each successfully
    downloaded (or already-existing) entry.
    """
    os.makedirs(dest_dir, exist_ok=True)
    headers = {"X-API-Key": api_key} if api_key else {}
    downloaded = []

    for i, item in enumerate(image_list):
        filename   = item["filename"]
        url        = item["url"]
        local_path = os.path.join(dest_dir, filename)

        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            if log_cb:
                log_cb(f"  [{i+1}/{len(image_list)}] Cached: {filename}")
            downloaded.append(dict(item, local_path=local_path))
            continue

        if not url:
            if log_cb:
                log_cb(f"  [{i+1}/{len(image_list)}] No URL for {filename}, skip.")
            continue

        try:
            resp = requests.get(url, headers=headers, timeout=30, stream=True)
            resp.raise_for_status()
            with open(local_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=16384):
                    fh.write(chunk)
            if log_cb:
                log_cb(f"  [{i+1}/{len(image_list)}] OK: {filename}")
            downloaded.append(dict(item, local_path=local_path))
        except Exception as e:
            if log_cb:
                log_cb(f"  [{i+1}/{len(image_list)}] FAIL {filename}: {e}")

    return downloaded


# ---------------------------------------------------------------------------
# USGS NWIS – Fetch instantaneous sensor data
# ---------------------------------------------------------------------------

def fetch_usgs_sensor(site_no: str, param_codes: List[str],
                      start_date: str, end_date: str,
                      log_cb: Optional[Callable] = None) -> pd.DataFrame:
    """
    Fetch USGS instantaneous values (IV) for the given site + parameters.
    Returns a DataFrame with columns: datetime, {paramCode}_... per parameter.
    """
    params_str = ",".join(param_codes)
    if log_cb:
        log_cb(f"  Fetching USGS IV data: site={site_no}, params={params_str}")

    try:
        resp = requests.get(
            NWIS_IV_BASE,
            params={
                "format":      "json",
                "sites":       site_no,
                "parameterCd": params_str,
                "startDT":     start_date,
                "endDT":       end_date,
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        if log_cb:
            log_cb(f"  Error fetching NWIS data: {e}")
        raise

    time_series = data.get("value", {}).get("timeSeries", [])
    if not time_series:
        if log_cb:
            log_cb("  No time-series data returned from NWIS.")
        return pd.DataFrame()

    dfs = []
    for ts in time_series:
        var_info   = ts.get("variable", {})
        code       = var_info.get("variableCode", [{}])[0].get("value", "??")
        label      = USGS_PARAMETERS.get(code, var_info.get("variableName", code))
        col_name   = f"{code}_{label.split('(')[0].strip().replace(' ', '_')}"
        values     = ts.get("values", [{}])[0].get("value", [])

        records = []
        for v in values:
            try:
                dt  = datetime.fromisoformat(v["dateTime"].replace("Z", "+00:00"))
                val = float(v["value"]) if v["value"] not in (None, "", "-999999") else None
                records.append({"datetime": dt, col_name: val})
            except Exception:
                continue

        if records:
            df = pd.DataFrame(records)
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
            dfs.append(df)
            if log_cb:
                log_cb(f"  {code}: {len(df)} readings.")

    if not dfs:
        return pd.DataFrame()

    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge_asof(
            merged.sort_values("datetime"),
            df.sort_values("datetime"),
            on="datetime",
            tolerance=pd.Timedelta("15min"),
            direction="nearest",
        )

    if log_cb:
        log_cb(f"  Merged: {len(merged)} rows, cols: {list(merged.columns)}")
    return merged


# ---------------------------------------------------------------------------
# Build labels CSV
# ---------------------------------------------------------------------------

def build_labels_csv(downloaded: List[dict], usgs_df: pd.DataFrame,
                     output_path: str,
                     log_cb: Optional[Callable] = None) -> Tuple[str, int]:
    """
    Join downloaded images with USGS sensor readings on nearest timestamp.
    Returns (csv_path, matched_count).
    """
    if usgs_df.empty:
        raise ValueError("USGS sensor DataFrame is empty.")

    rows = [
        {"image_path": item["local_path"], "dt_image": item["timestamp"]}
        for item in downloaded
        if item.get("timestamp") and item.get("local_path")
    ]
    if not rows:
        raise ValueError("No images with valid timestamps.")

    img_df = pd.DataFrame(rows)
    img_df["dt_image"] = pd.to_datetime(img_df["dt_image"], utc=True)
    img_df = img_df.sort_values("dt_image")

    sensor_df = usgs_df.copy().sort_values("datetime")

    merged = pd.merge_asof(
        img_df,
        sensor_df,
        left_on="dt_image",
        right_on="datetime",
        tolerance=pd.Timedelta("15min"),
        direction="nearest",
    )

    before = len(merged)

    # Rename primary water level column for train_demo compatibility.
    # Precipitation or other environmental readings are useful features in the
    # CSV, but they must not make a row trainable without a numeric target.
    merged = merged.rename(columns={"dt_image": "timestamp"})
    target_col = None
    for code in WATER_LEVEL_TARGET_CODES:
        candidates = [c for c in merged.columns if c.startswith(code + "_")]
        if candidates:
            target_col = candidates[0]
            merged = merged.rename(columns={target_col: "water_level"})
            if log_cb:
                log_cb(f"  Primary label column: '{target_col}' -> 'water_level'")
            break

    if target_col is None:
        requested = ", ".join(WATER_LEVEL_TARGET_CODES)
        raise ValueError(
            "No usable water-level target was returned by USGS. "
            f"Select at least one water-level parameter ({requested}); "
            "precipitation-only data cannot be used for training."
        )

    merged["water_level"] = pd.to_numeric(merged["water_level"], errors="coerce")
    merged = merged.dropna(subset=["water_level"])
    matched = len(merged)

    if matched == 0:
        raise ValueError(
            "No images matched a valid numeric water_level value within 15 minutes. "
            "Try a different date range, site, or water-level parameter."
        )

    if log_cb:
        log_cb(f"  Matched: {matched}/{before} images with valid water_level values.")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    merged.to_csv(output_path, index=False)
    if log_cb:
        log_cb(f"  CSV saved: {output_path}  ({matched} rows)")
    return output_path, matched


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_acquisition(site_name: str, start_date: str, end_date: str,
                    max_images: int, param_codes: List[str],
                    dest_dir: str, api_key: Optional[str] = None,
                    log_cb: Optional[Callable] = None) -> Tuple[str, int, tuple]:
    """
    Complete pipeline: camera lookup → list images → download → USGS data → CSV.
    Returns (csv_path, matched_count, roi_tuple).
    """
    site_info = SITE_CATALOG.get(site_name)
    if not site_info:
        raise ValueError(f"Unknown site: {site_name}")

    roi        = site_info.get("roi", (400, 0, 900, 1920))
    images_dir = os.path.join(dest_dir, "images")
    csv_path   = os.path.join(dest_dir, "labels.csv")

    # 1. Discover camera
    if log_cb:
        log_cb(f"\n[1/5] Looking up HiVIS camera for site '{site_name}' ...")
    cam = fetch_camera_info(site_info, api_key=api_key, log_cb=log_cb)
    if cam is None:
        raise RuntimeError(
            f"No HiVIS camera found for site '{site_name}'.\n"
            "Check that this site has a camera on https://apps.usgs.gov/hivis/"
        )
    if log_cb:
        log_cb(f"  Camera: {cam.get('camName', cam.get('camId', 'unknown'))}")

    # 2. List images
    if log_cb:
        log_cb(f"\n[2/5] Listing images {start_date} → {end_date} ...")
    image_list = list_hivis_images(cam, start_date, end_date,
                                   max_images=max_images, log_cb=log_cb)
    if not image_list:
        raise RuntimeError("No images found in the specified date range.")

    # 3. Download
    if log_cb:
        log_cb(f"\n[3/5] Downloading {len(image_list)} images ...")
    dl = download_images(image_list, images_dir, api_key=api_key, log_cb=log_cb)
    if not dl:
        raise RuntimeError("No images downloaded successfully.")
    if log_cb:
        log_cb(f"  {len(dl)} images ready.")

    # 4. USGS sensor data
    site_no = site_info["site_no"]
    if log_cb:
        log_cb(f"\n[4/5] Fetching USGS sensor data for site {site_no} ...")
    usgs_df = fetch_usgs_sensor(site_no, param_codes, start_date, end_date,
                                log_cb=log_cb)

    # 5. Build CSV
    if log_cb:
        log_cb(f"\n[5/5] Building labels CSV ...")
    csv_path, matched = build_labels_csv(dl, usgs_df, csv_path, log_cb=log_cb)

    if log_cb:
        log_cb(f"\nDone! {matched} labelled samples ready.")
        log_cb(f"  Images : {images_dir}")
        log_cb(f"  CSV    : {csv_path}")
        log_cb(f"  ROI    : {roi}")

    return csv_path, matched, roi
