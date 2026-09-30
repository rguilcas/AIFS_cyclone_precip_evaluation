"""
Daily AIFS Single forecast precipitation over Northern Hemisphere land, for every
forecast that covers a set of valid days (e.g. all winter days), from dynamical.org.

Why this is organised by init time: the store is chunked as one init_time x ALL
lead times x 60x60 deg tiles. Reading one (init, lead) pair still downloads the
whole chunk, so the old per-event loop fetched ~60x more data than it used. Here
each init time is read ONCE (all leads) and turned into daily totals for every
valid day it covers; an event's forecasts at all leads are then a simple selection.

Output: one file per init month, dims (init_time, lead_day, point)
    tp_daily     mm/day, UTC day (four 6 h steps ending 06, 12, 18, 24 UTC)
    valid_date   (init_time, lead_day) the UTC day each value is for
    lead_end_h   (init_time, lead_day) hours from init to the END of that day
    point_lat / point_lon  land points (lon -180..180)
Only complete days are kept; lead 0 (no accumulation) is dropped.

Example:
    python aifs_daily_land_precip.py --valid-start 2024-12-01 --valid-end 2025-02-28 \
        --land-mask land_mask.nc --out-dir aifs_winter --workers 8
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

import dynamical_catalog  # dynamical-catalog>=0.8.0

MAX_LEAD = pd.Timedelta(hours=360)
N_LEAD_DAYS = 15


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--valid-start", required=True, help="First valid day, e.g. 2024-12-01")
    p.add_argument("--valid-end", required=True, help="Last valid day, e.g. 2025-02-28")
    p.add_argument("--land-mask", required=True,
                   help="NetCDF with a boolean/0-1 variable 'land' on the ERA5 0.25 deg grid")
    p.add_argument("--lat-min", type=float, default=30.0)
    p.add_argument("--lat-max", type=float, default=80.0)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ds = dynamical_catalog.open("ecmwf-aifs-single-forecast", chunks=None)
    pr = ds["precipitation_surface"]  # mean rate over the previous 6 h, mm/s equivalent

    lat = pr.latitude.values
    lat_sel = slice(args.lat_max, args.lat_min) if lat[0] > lat[-1] else slice(args.lat_min, args.lat_max)
    pr = pr.sel(latitude=lat_sel)

    # Land mask onto the AIFS grid (lon -180..180); fails loudly if the grids differ
    land = xr.open_dataset(args.land_mask)["land"].astype(bool)
    land = land.assign_coords(longitude=((land.longitude + 180) % 360) - 180).sortby("longitude")
    land = land.reindex(latitude=pr.latitude, longitude=pr.longitude, method="nearest", tolerance=1e-3)
    if land.isnull().any():
        raise ValueError("Land mask does not cover the AIFS grid; check coordinates")
    prow, pcol = np.nonzero(land.values)
    plat, plon = pr.latitude.values[prow], pr.longitude.values[pcol]
    print(f"{len(plat)} land points")

    # Every init whose 15-day forecast reaches the valid period
    v0 = pd.Timestamp(args.valid_start)
    v1 = pd.Timestamp(args.valid_end)
    avail = pd.DatetimeIndex(pr.init_time.values)
    inits = avail[(avail >= v0 - MAX_LEAD) & (avail <= v1 + pd.Timedelta(days=1))]
    print(f"{len(inits)} init times from {inits[0]} to {inits[-1]}")

    def daily_from_init(init):
        da = pr.sel(init_time=init).load()                            # (lead, lat, lon)
        lead_h = da.lead_time.values / np.timedelta64(1, "h")
        keep = lead_h > 0
        mm6 = da.values[keep][:, prow, pcol] * 21600.0               # mm per 6 h, land points
        ends = init + pd.to_timedelta(lead_h[keep], unit="h")
        days = (ends - pd.Timedelta(seconds=1)).floor("D")          # step (T-6h, T] -> its UTC day

        vals = np.full((N_LEAD_DAYS, len(plat)), np.nan, dtype=np.float32)
        vdate = np.full(N_LEAD_DAYS, np.datetime64("NaT"), dtype="datetime64[ns]")
        lend = np.full(N_LEAD_DAYS, np.nan, dtype=np.float32)
        k = 0
        for d in pd.unique(days):
            sel = days == d
            if sel.sum() != 4 or k >= N_LEAD_DAYS:
                continue                                              # incomplete day
            vals[k] = mm6[sel].sum(axis=0)                            # NaN if any step missing
            vdate[k] = np.datetime64(d)
            lend[k] = (d + pd.Timedelta(days=1) - init) / pd.Timedelta(hours=1)
            k += 1
        return vals, vdate, lend

    months = pd.Series(inits).groupby(inits.to_period("M"))
    for month, month_inits in months:
        out = os.path.join(args.out_dir, f"aifs_daily_land_tp_init{month}.nc")
        if os.path.exists(out):
            print(f"{month}: exists, skipping")
            continue
        month_inits = list(month_inits)
        with ThreadPoolExecutor(args.workers) as pool:
            res = list(tqdm(pool.map(daily_from_init, month_inits),
                            total=len(month_inits), desc=str(month)))
        vals, vdate, lend = (np.stack(x) for x in zip(*res))
        xr.Dataset(
            {"tp_daily": (["init_time", "lead_day", "point"], vals)},
            coords={"init_time": month_inits,
                    "lead_day": np.arange(N_LEAD_DAYS),
                    "valid_date": (["init_time", "lead_day"], vdate),
                    "lead_end_h": (["init_time", "lead_day"], lend),
                    "point_lat": ("point", plat.astype(np.float32)),
                    "point_lon": ("point", plon.astype(np.float32))},
            attrs={"source": "ECMWF AIFS Single forecast via dynamical.org",
                   "day_definition": "UTC day, four 6 h steps ending 06, 12, 18, 24 UTC",
                   "units": "mm day-1"},
        ).to_netcdf(out, encoding={"tp_daily": {"zlib": True, "complevel": 4}})
        print(f"saved {out}")


if __name__ == "__main__":
    main()