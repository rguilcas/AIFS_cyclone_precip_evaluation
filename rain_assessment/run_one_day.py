"""
One-day pipeline: ERA5 heavy-rain events linked to cyclones, verified against
AIFS Single, IFS control and the IFS ensemble.

For one UTC day it:
  1. detects heavy-rain objects over land in ERA5 (local P95 and > 10 mm),
  2. attributes them to CNECT cyclones (storm-radius footprints, polar equal-area),
  3. merges objects of the same storm into one event and draws a 2000 km box around it,
  4. extracts ERA5, AIFS Single, IFS control and all 51 IFS members on exactly the box
     cells, for forecasts started 1..14 days before,
  5. writes one NetCDF per event with variables
        observation            (latitude, longitude)
        event_mask             (latitude, longitude)            1 = event cell
        forecast_aifs          (init_time, latitude, longitude)
        forecast_ifs_control   (init_time, latitude, longitude)
        forecast_ifs_ens       (init_time, ensemble_member, latitude, longitude)
     all in mm per UTC day (00-24 UTC).

Usage in a notebook:
    from run_one_day import run_one_day
    paths = run_one_day("2025-03-10", precip, climato, mask_land)

precip:    ERA5 daily totals (time, latitude, longitude), mm per 00-24 UTC day, covering
           at least 10-90N so every box fits
climato:   local precipitation quantiles (quantile, latitude, longitude)
mask_land: boolean land mask
"""

import os

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import xarray as xr
from shapely.geometry import MultiPolygon, Polygon
from tqdm.auto import tqdm

import dynamical_catalog
from polar_events import (POLAR, TO_POLAR, attribute, cut, cyclone_footprints, event_boxes,
                          merge_by_cyclone, rain_objects)

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
OUT_DIR = "/nird/datapeak/NS12107K/rguilcas/aifs-cyclone-rain-predictions"
CYC_DIR = "/nird/datapeak/NS12107K/rguilcas/data/cyclones_cnect"
LEAD_DAYS = np.arange(1, 15)          # forecasts started 1..14 days before the event day (00 UTC)
HEAVY_QUANTILE = 0.95
HEAVY_MIN_MM = 10.0
MIN_OBJECT_KM2 = 20000
BOX_MIN_KM, BOX_MARGIN_KM = 2000, 500
IFS_CONTROL = 0                       # ECMWF convention: member 0 is the control
N_WORKERS = 16
# Date AIFS Single v1.0 became operational: CHECK against ECMWF's announcement.
# Archived forecasts before it come from the earlier (experimental) AIFS version.
AIFS_V1_START = pd.Timestamp("2025-02-25")

# ----------------------------------------------------------------------------
# Forecast archives (opened once; nothing is loaded here)
# ----------------------------------------------------------------------------
AIFS = dynamical_catalog.open("ecmwf-aifs-single-forecast", chunks=None)["precipitation_surface"]
IFS = dynamical_catalog.open("ecmwf-ifs-ens-forecast-15-day-0-25-degree",
                             chunks=None)["precipitation_surface"]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def dims_only(a):
    """Drop every coordinate that is not a dimension (spatial_ref, expver, time, ...)."""
    return a.drop_vars([c for c in a.coords if c not in a.dims])


def standard_grid(a, ndigits=4):
    """float64, rounded coords, lon in -180..180, lat descending. Apply to FULL fields only."""
    lat = np.round(a.latitude.values.astype("float64"), ndigits)
    lon = np.round((((a.longitude.values.astype("float64") + 180) % 360) - 180), ndigits)
    return a.assign_coords(latitude=lat, longitude=lon).sortby("latitude", ascending=False).sortby("longitude")


def snap(a, ndigits=4):
    """Round coords after cut(); keeps cut()'s continuous longitudes and ordering."""
    return a.assign_coords(latitude=np.round(a.latitude.values.astype("float64"), ndigits),
                           longitude=np.round(a.longitude.values.astype("float64"), ndigits))


def sel_like(field, ob):
    """Exactly ob's cells from a -180..180 field, in ob's order (dateline boxes stay
    continuous). With a lazy Zarr field, only the chunks under the box are read."""
    lon = np.round(((ob.longitude.values + 180) % 360) - 180, 4)
    out = field.sel(latitude=ob.latitude.values, longitude=lon, method="nearest", tolerance=1e-3)
    return out.assign_coords(latitude=ob.latitude.values, longitude=ob.longitude.values)


def daily_total(pr, day, init):
    """Sum a rate (mm/s, mean over the previous step) over [day 00, day+1 00] UTC, using
    each step's own length, so 3-hourly and 6-hourly steps both work. Loads only that day."""
    start, end = day - init, day + pd.Timedelta(days=1) - init
    lt = pd.to_timedelta(pr.lead_time.values)
    step_s = np.diff(np.concatenate([[lt[0] - (lt[1] - lt[0])], lt])) / pd.Timedelta(seconds=1)
    pr = pr.assign_coords(step_s=("lead_time", step_s))
    sel = pr.sel(lead_time=slice(start + pd.Timedelta(seconds=1), end))
    covered = float(sel.step_s.sum())
    if abs(covered - 86400) > 1:
        raise ValueError(f"init {init}: steps cover {covered / 3600:.1f} h, not 24 h")
    return (sel.load() * sel.step_s).sum("lead_time", min_count=1)


def forecast_box(src, init, ob, day):
    """Daily total of one forecast start on the box cells; NaN if the start is not archived."""
    if init not in src.indexes["init_time"]:
        return None
    return dims_only(daily_total(sel_like(src.sel(init_time=init), ob), day, init)).expand_dims(init_time=[init])


def event_mask(field, geom):
    """True where a cell centre lies inside the event polygon (polar CRS)."""
    LA, LO = xr.broadcast(dims_only(field.latitude), dims_only(field.longitude))
    X, Y = TO_POLAR.transform(LO.values, LA.values)
    return xr.DataArray(shapely.contains_xy(geom, X, Y), coords=LA.coords, dims=LA.dims)


def _polygonal(g):
    """Polygonal part of one geometry, or None if nothing polygonal is left.

    Degenerate footprints (zero or NaN radius, reprojection near the pole or dateline,
    make_valid on self-intersecting shapes) can come out as points, lines or
    GeometryCollections. Only their area matters for the overlap, so keep the polygons.
    """
    if g is None or g.is_empty:
        return None
    if not g.is_valid:
        g = shapely.make_valid(g)
    if isinstance(g, (Polygon, MultiPolygon)):
        return g if not g.is_empty else None
    polys = []
    for part in shapely.get_parts(g):
        if isinstance(part, Polygon) and not part.is_empty:
            polys.append(part)
        elif isinstance(part, MultiPolygon):
            polys.extend(p for p in part.geoms if not p.is_empty)
    if not polys:
        return None
    return polys[0] if len(polys) == 1 else MultiPolygon(polys)


def polygons_only(gdf, what="geometries", crs=POLAR):
    """Reproject to `crs`, keep only polygonal parts and drop rows left without area.
    Cleaning happens in the polar CRS so attribute()'s own to_crs(POLAR) is a no-op
    and cannot re-introduce non-polygonal shapes."""
    gdf = gdf.to_crs(crs)
    types_before = gdf.geom_type.value_counts().to_dict()
    geom = gdf.geometry.apply(_polygonal)
    keep = geom.notna().values
    out = gdf.loc[keep].copy()
    out = out.set_geometry(gpd.GeoSeries(list(geom[keep]), index=out.index, crs=crs))
    n_drop = int((~keep).sum())
    if n_drop or set(types_before) - {"Polygon", "MultiPolygon"}:
        print(f"  {what}: types {types_before} -> kept {len(out)} polygonal, dropped {n_drop}")
    return out


def load_cyclones(day):
    """Track points in [day 00 UTC, next day 00 UTC], across a month boundary if needed."""
    nxt = day + pd.Timedelta(days=1)
    files = {f"{CYC_DIR}/{t.year}/System13_2E5R_BBox10_{t.year}{t.month:02d}.csv" for t in (day, nxt)}
    cyc = pd.concat([pd.read_csv(f) for f in sorted(files) if os.path.exists(f)])
    cyc["datetime"] = pd.to_datetime(cyc[["year", "month", "day", "hour"]])
    cyc = cyc[(cyc["datetime"] >= day) & (cyc["datetime"] <= nxt)].drop_duplicates()
    # TODO: track filters (lifetime/length per sid, primary centres only)
    return cyc[cyc["lat"] > 20]


def stack_leads(pieces, ob, inits):
    """Concatenate per-lead boxes; missing starts become NaN so every file has all leads."""
    have = {pd.Timestamp(p.init_time.values[0]): p for p in pieces if p is not None}
    ref = next(iter(have.values()), None)
    out = []
    for init in inits:
        if init in have:
            out.append(have[init])
        elif ref is not None:
            out.append((ref * np.nan).assign_coords(init_time=[init]))
    return xr.concat(out, dim="init_time") if out else None


def netcdf_safe_attrs(ds):
    """Keep only attributes netCDF can store; turn anything else (dicts, None, bools) into text."""
    def clean(attrs):
        out = {}
        for key, v in attrs.items():
            if v is None:
                continue
            if isinstance(v, (bool, np.bool_)):
                out[key] = int(v)
            elif isinstance(v, (str, int, float, np.number, np.ndarray, list, tuple, bytes)):
                out[key] = v
            else:
                out[key] = str(v)                          # e.g. {'min': 0, 'max': 50} -> text
        return out

    ds = ds.copy()
    ds.attrs = clean(ds.attrs)
    for name in ds.variables:
        ds[name].attrs = clean(ds[name].attrs)
    return ds


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def run_one_day(day, precip, climato, mask_land, out_dir=OUT_DIR, overwrite=False):
    day = pd.Timestamp(day)
    os.makedirs(out_dir, exist_ok=True)

    # ---- 1. ERA5 heavy-rain objects --------------------------------------------------
    day_str = f"{day:%Y-%m-%d}"
    sel = precip.sel(time=day_str)                            # every time stamp on that date
    if "time" in sel.dims:
        if sel.sizes["time"] != 1:
            raise ValueError(f"{day_str}: expected 1 daily field, found {sel.sizes['time']}")
        sel = sel.isel(time=0)
    da = dims_only(standard_grid(sel))                        # full field: used for cutting boxes

    thr = dims_only(standard_grid(climato.sel(quantile=HEAVY_QUANTILE)))
    land = dims_only(standard_grid(mask_land))

    # detection domain = the climatology's grid; precip and land mask are cut to it
    da_d = da.sel(latitude=thr.latitude, longitude=thr.longitude)
    land = land.sel(latitude=thr.latitude, longitude=thr.longitude)
    da_d, thr, land = xr.align(da_d, thr, land, join="exact")
    heavy = (da_d > thr) & (da_d > HEAVY_MIN_MM) & land.astype(bool)

    objs = rain_objects(da_d, heavy, min_area_km2=MIN_OBJECT_KM2)
    if len(objs) == 0:
        return []

    # ---- 2. cyclone attribution and events -------------------------------------------
    cyc = load_cyclones(day)
    if len(cyc) == 0:
        return []
    fp_raw = cyclone_footprints(cyc, cyc["radius"], aggfunc={"radius": "max", "depth": "max"})
    # depth per storm from the raw footprints, so storms whose footprint gets dropped
    # below still keep their metadata
    depth_by_sid = fp_raw.set_index("sid")["depth"] if "depth" in fp_raw else pd.Series(dtype=float)

    # gpd.overlay refuses mixed geometry types: keep only polygons on both sides
    fp = polygons_only(fp_raw, what=f"{day_str} cyclone footprints")
    objs = polygons_only(objs, what=f"{day_str} rain objects")
    if len(fp) == 0 or len(objs) == 0:
        return []

    objs = attribute(objs, fp)
    merged = merge_by_cyclone(objs)
    if len(merged) == 0:
        return []
    events = event_boxes(merged, min_size_km=BOX_MIN_KM, margin_km=BOX_MARGIN_KM)

    # verification field: the full precip field (wider than the detection band)
    obs_boxes = [snap(cut(da, b)) for b in events["box"]]

    # ---- 3. forecasts on exactly the box cells (parallel downloads) --------------------
    from concurrent.futures import ThreadPoolExecutor

    inits = [day - pd.Timedelta(days=int(d)) for d in LEAD_DAYS]
    jobs = [(name, src, init, k, ob)
            for init in inits
            for k, ob in enumerate(obs_boxes)
            for name, src in (("aifs", AIFS), ("ifs", IFS))]

    def work(job):
        name, src, init, k, ob = job
        return name, k, forecast_box(src, init, ob, day)

    aifs_p = [[] for _ in obs_boxes]
    ifs_p = [[] for _ in obs_boxes]
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        for name, k, res in tqdm(pool.map(work, jobs), total=len(jobs),
                                 desc=f"{day:%Y-%m-%d}", leave=False):
            (aifs_p if name == "aifs" else ifs_p)[k].append(res)

    # ---- 4. one file per event -------------------------------------------------------
    paths = []
    for k, (_, ev) in enumerate(events.iterrows()):
        sid = int(ev["sid"])
        path = f"{out_dir}/cyclonic_event_{day:%Y-%m-%d}_sid{sid}.nc"
        if os.path.exists(path) and not overwrite:
            paths.append(path)
            continue

        ob = obs_boxes[k]
        aifs = stack_leads(aifs_p[k], ob, inits)
        ifs = stack_leads(ifs_p[k], ob, inits)
        if aifs is None or ifs is None:
            print(f"{day:%Y-%m-%d} sid {sid}: no forecasts archived, skipping")
            continue
        aifs, ifs, ob = xr.align(aifs, ifs, ob, join="exact")
        lead = ("init_time", [int(d) for d in LEAD_DAYS])
        aifs = aifs.assign_coords(lead_days=lead)
        ifs = ifs.assign_coords(lead_days=lead)

        m = (event_mask(ob, ev.geometry) & ob.notnull()).astype("int8")
        out = xr.Dataset({
            "observation": ob,
            "event_mask": m,
            "forecast_aifs": aifs,
            "forecast_ifs_control": ifs.sel(ensemble_member=IFS_CONTROL, drop=True),
            "forecast_ifs_ens": ifs,
        })
        out = netcdf_safe_attrs(out)
        out = out.assign_coords(aifs_v1=("init_time", [i >= AIFS_V1_START for i in inits]))

        for v, name in [("observation", "ERA5"), ("forecast_aifs", "AIFS Single"),
                        ("forecast_ifs_control", "IFS ENS control"),
                        ("forecast_ifs_ens", "IFS ENS members")]:
            out[v].attrs.update(units="mm day-1", long_name=f"{name} precipitation, 00-24 UTC")
        out["event_mask"].attrs.update(long_name="ERA5 heavy-rain event cells", flag_values="0 1")
        out["aifs_v1"].attrs.update(long_name="AIFS Single v1.0 in operation at init (check date)")
        out.attrs.update(
            event_day=f"{day:%Y-%m-%d}", sid=sid,
            cyclone_depth_max=float(depth_by_sid.get(ev["sid"], np.nan)),
            event_area_km2=float(ev["area_km2"]), n_objects=len(ev["labels"]),
            centre_lat=float(ev["centre_lat"]), centre_lon=float(ev["centre_lon"]),
            overlap_frac_min=float(ev["overlap_frac"]), rain_max=float(ev["rain_max"]),
            n_event_cells=int(m.sum()), box_polar_bounds_m=list(ev["box"].bounds),
            box_crs="EPSG:6931",
            heavy_definition=f"ERA5 > local P{int(HEAVY_QUANTILE * 100)} and > {HEAVY_MIN_MM} mm, land",
            attribution="object area overlap with CNECT storm-radius footprint >= 0.5",
        )
        enc = {v: {"zlib": True, "complevel": 4} for v in out.data_vars}
        out.to_netcdf(path, encoding=enc)
        paths.append(path)
    return paths