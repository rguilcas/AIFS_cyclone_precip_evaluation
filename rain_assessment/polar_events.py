"""
Heavy-rain objects, cyclone footprints and event boxes in a north-polar projection.

Everything geometric happens in EASE-Grid 2.0 North (EPSG:6931), a Lambert azimuthal
EQUAL-AREA projection centred on the North Pole:
  * no dateline: shapes crossing 180 deg are single, continuous polygons;
  * no pole problem: a cyclone circle containing the pole is an ordinary ring;
  * areas are exact, so overlap fractions are true area fractions;
  * boxes are defined in km.
(Stereographic would also remove the seams, but it is not equal-area: at 30N it
inflates areas by ~1.8x relative to the pole, which would bias overlap fractions.)
Northern Hemisphere only.

Typical use for one day:

    objs = rain_objects(da, mask)                            # labels -> polar polygons
    fp   = cyclone_footprints(cyclones, cyclones["radius"])  # geodesic circles, dissolved
    objs = attribute(objs, fp)                               # overlap_frac, sid, is_cyclonic
    ev   = merge_by_cyclone(objs)                            # one event per storm
    ev   = event_boxes(ev, min_size_km=2000, margin_km=500)
    sub  = cut(da, ev.box.iloc[0])                           # same call for ERA5 and AIFS
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
import xarray as xr
from pyproj import Geod, Transformer
from rasterio.features import shapes
from rasterio.transform import from_origin
from scipy.ndimage import label as nd_label
from shapely.geometry import box, shape

POLAR = "EPSG:6931"  # EASE-Grid 2.0 North (LAEA, equal-area), metres
GEOD = Geod(ellps="WGS84")
TO_POLAR = Transformer.from_crs("EPSG:4326", POLAR, always_xy=True)
TO_LL = Transformer.from_crs(POLAR, "EPSG:4326", always_xy=True)
STRUCT = np.ones((3, 3), dtype=bool)  # 8-connectivity


def _to_polar(geom):
    return shapely.transform(geom, lambda xy: np.column_stack(TO_POLAR.transform(xy[:, 0], xy[:, 1])))


def _grid(da):
    """(lat, lon) oriented north->south, and checks the grid spans the full circle."""
    da = da.transpose("latitude", "longitude")
    if da.latitude[0] < da.latitude[-1]:
        da = da.isel(latitude=slice(None, None, -1))
    lon = da.longitude.values
    dlon = float(abs(lon[1] - lon[0]))
    periodic = np.isclose((lon[-1] + dlon) - lon[0], 360.0)
    return da, periodic


# ----------------------------------------------------------------------------
# Labelling that is periodic in longitude
# ----------------------------------------------------------------------------
def periodic_label(mask, structure=STRUCT):
    """scipy.ndimage.label, plus merging of objects that touch across the
    first/last longitude column (the dateline or Greenwich, depending on the grid)."""
    lab, n = nd_label(mask, structure=structure)
    if n == 0:
        return lab, 0
    parent = np.arange(n + 1)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    left, right = lab[:, 0], lab[:, -1]
    nrow = lab.shape[0]
    offsets = (-1, 0, 1) if structure[0, 0] else (0,)
    for dr in offsets:
        for r in np.nonzero(left)[0]:
            rr = r + dr
            if 0 <= rr < nrow and right[rr]:
                a, b = find(left[r]), find(right[rr])
                if a != b:
                    parent[max(a, b)] = min(a, b)
    roots = np.array([find(i) for i in range(n + 1)])
    _, new = np.unique(roots, return_inverse=True)  # 0 stays 0
    return new[lab], int(new.max())


# ----------------------------------------------------------------------------
# Rain objects
# ----------------------------------------------------------------------------
def rain_objects(da, mask, min_area_km2=0.0, structure=STRUCT):
    """Heavy-rain objects as polar-projection polygons.

    da:   DataArray (latitude, longitude) of precipitation, any lon convention
    mask: boolean DataArray on the same grid (e.g. heavy & land)
    Returns a GeoDataFrame (crs EPSG:6931) indexed by 'label' with area_km2, n_cells,
    rain_max, rain_mass (sum of rain x cell area, mm km2) and the rain-weighted
    centre (cx, cy in polar metres; centre_lat, centre_lon).
    """
    da, periodic = _grid(da)
    mask = mask.transpose("latitude", "longitude").reindex_like(da)
    if mask.shape != da.shape:
        raise ValueError("mask and da are on different grids")
    lat, lon = da.latitude.values, da.longitude.values
    dlat, dlon = abs(lat[1] - lat[0]), abs(lon[1] - lon[0])

    m = mask.fillna(False).values.astype(bool)
    lab, n = periodic_label(m, structure) if periodic else nd_label(m, structure=structure)
    if n == 0:
        return gpd.GeoDataFrame({"label": []}, geometry=[], crs=POLAR).set_index("label")

    # polygons: trace in lon/lat, densify so straight lon/lat edges become proper arcs,
    # clip at the pole, project, union pieces of the same label (seam pieces join here)
    transform = from_origin(lon[0] - dlon / 2, lat[0] + dlat / 2, dlon, dlat)
    world = box(-1e4, -90, 1e4, 90)
    parts = {}
    for geom, val in shapes(lab.astype(np.int32), mask=lab > 0, transform=transform,
                            connectivity=8):
        g = shapely.segmentize(shape(geom).intersection(world), max(dlat, dlon))
        parts.setdefault(int(val), []).append(_to_polar(g))
    ids = sorted(parts)
    geoms = [shapely.union_all(parts[i], grid_size=1.0) for i in ids]

    # per-object statistics from the cells
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    cell_km2 = (np.deg2rad(dlon) * 6371.0 ** 2 * np.deg2rad(dlat) * np.cos(np.deg2rad(LA)))
    X, Y = TO_POLAR.transform(LO, LA)
    r = np.nan_to_num(da.values)
    sel = lab > 0
    L = lab[sel]
    mass_cell = r[sel] * cell_km2[sel]
    s = pd.DataFrame({"label": L, "mass": mass_cell, "mx": mass_cell * X[sel],
                      "my": mass_cell * Y[sel], "r": r[sel], "a": cell_km2[sel]})
    g = s.groupby("label")
    stats = pd.DataFrame({
        "n_cells": g.size(),
        "rain_max": g["r"].max(),
        "rain_mass": g["mass"].sum(),
        "cx": g["mx"].sum() / g["mass"].sum(),
        "cy": g["my"].sum() / g["mass"].sum(),
    }).reindex(ids)

    out = gpd.GeoDataFrame(stats, geometry=geoms, crs=POLAR)
    out.index.name = "label"
    out["area_km2"] = out.area / 1e6
    out["centre_lon"], out["centre_lat"] = TO_LL.transform(out["cx"].values, out["cy"].values)
    return out[out["area_km2"] > min_area_km2]


# ----------------------------------------------------------------------------
# Cyclone footprints
# ----------------------------------------------------------------------------
def cyclone_footprints(df, radius_km, by=("sid",), lon="lon", lat="lat", n_vertices=72,
                       aggfunc=None):
    """Geodesic circles of true radius around each track point, projected to the polar
    CRS (a circle around the pole is an ordinary ring), dissolved per `by`.
    df: track points for ONE period (e.g. one day). radius_km: scalar or per-row array."""
    df = pd.DataFrame(df).copy()
    lo = df[lon].to_numpy(dtype=float)
    la = df[lat].to_numpy(dtype=float)
    r = np.broadcast_to(np.asarray(radius_km, dtype=float), lo.shape) * 1000.0
    n = len(lo)
    az = np.linspace(0, 360, n_vertices, endpoint=False)
    plon, plat, _ = GEOD.fwd(np.repeat(lo, n_vertices), np.repeat(la, n_vertices),
                             np.tile(az, n), np.repeat(r, n_vertices))
    x, y = TO_POLAR.transform(plon, plat)
    ring = np.stack([x.reshape(n, n_vertices), y.reshape(n, n_vertices)], axis=-1)
    geoms = shapely.make_valid(shapely.polygons(np.concatenate([ring, ring[:, :1]], axis=1)))
    df["buffer_km"] = r / 1000.0
    gdf = gpd.GeoDataFrame(df.drop(columns="geometry", errors="ignore"), geometry=geoms, crs=POLAR)
    if by is None:
        return gdf
    fp = gdf.dissolve(by=list(by), aggfunc=aggfunc or "first").reset_index()
    fp.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c for c in fp.columns]
    return fp


# ----------------------------------------------------------------------------
# Attribution and merging
# ----------------------------------------------------------------------------
def attribute(objs, footprints, key="sid", threshold=0.5):
    """Area overlap of each object with each footprint (equal-area, so exact).
    Adds overlap_frac (best storm), sid (best storm), n_storms, is_cyclonic."""
    objs = objs.copy()
    objs["overlap_frac"], objs[key], objs["n_storms"] = 0.0, np.nan, 0
    if len(objs) == 0 or len(footprints) == 0:
        objs["is_cyclonic"] = False
        return objs
    left = objs.reset_index()[["label", "geometry"]]
    right = footprints[[key, "geometry"]].to_crs(POLAR)
    inter = gpd.overlay(left, right, how="intersection", keep_geom_type=True)
    if len(inter):
        inter["overlap_km2"] = inter.area / 1e6
        ov = (inter.groupby(["label", key])["overlap_km2"].sum()
                   .div(objs["area_km2"], level="label").rename("frac").reset_index())
        best = ov.sort_values("frac", ascending=False).drop_duplicates("label").set_index("label")
        objs.loc[best.index, "overlap_frac"] = best["frac"]
        objs.loc[best.index, key] = best[key]
        objs["n_storms"] = ov.groupby("label")[key].nunique().reindex(objs.index).fillna(0).astype(int)
    objs["is_cyclonic"] = objs["overlap_frac"] >= threshold
    return objs


def merge_by_cyclone(objs, key="sid", only_cyclonic=True):
    """One event per storm: union of its objects, with mass-weighted centre and summed stats.
    Non-cyclonic objects (if kept) stay as individual events."""
    o = objs[objs["is_cyclonic"]] if only_cyclonic else objs
    if len(o) == 0:
        return o.copy()
    o = o.copy()
    o["group"] = np.where(o["is_cyclonic"], o[key].astype(str), "obj_" + o.index.astype(str))
    o["_mx"], o["_my"] = o["cx"] * o["rain_mass"], o["cy"] * o["rain_mass"]
    ev = o.dissolve(by="group", aggfunc={
        key: "first", "rain_mass": "sum", "_mx": "sum", "_my": "sum", "n_cells": "sum",
        "rain_max": "max", "overlap_frac": "min", "is_cyclonic": "first",
    })
    ev["cx"], ev["cy"] = ev["_mx"] / ev["rain_mass"], ev["_my"] / ev["rain_mass"]
    ev["area_km2"] = ev.area / 1e6
    ev["centre_lon"], ev["centre_lat"] = TO_LL.transform(ev["cx"].values, ev["cy"].values)
    ev["labels"] = o.reset_index().groupby("group")["label"].apply(list)
    return ev.drop(columns=["_mx", "_my"])


# ----------------------------------------------------------------------------
# Event boxes and cutting fields
# ----------------------------------------------------------------------------
def event_boxes(events, min_size_km=2000.0, margin_km=500.0):
    """Square-ish boxes in the polar projection (km), centred on the rain-weighted centre,
    at least min_size_km per side and always containing the event plus margin_km."""
    boxes = []
    for (_, e) in events.iterrows():
        minx, miny, maxx, maxy = e.geometry.bounds
        hx = max(min_size_km / 2, max(e.cx - minx, maxx - e.cx) / 1000 + margin_km) * 1000
        hy = max(min_size_km / 2, max(e.cy - miny, maxy - e.cy) / 1000 + margin_km) * 1000
        boxes.append(box(e.cx - hx, e.cy - hy, e.cx + hx, e.cy + hy))
    out = events.copy()
    out["box"] = gpd.GeoSeries(boxes, index=events.index, crs=POLAR)
    return out


def cut(da, box_geom):
    """Cells of da whose centres fall inside a polar box. Longitudes are first made
    continuous around the box centre, so boxes crossing the dateline or Greenwich give
    one contiguous array. Cells outside the (rotated) box are NaN."""
    minx, miny, maxx, maxy = box_geom.bounds
    c = box_geom.centroid
    clon, _ = TO_LL.transform(c.x, c.y)
    lon_c = ((da.longitude - clon + 180) % 360) - 180 + clon
    d = da.assign_coords(longitude=lon_c).sortby("longitude")
    LA, LO = xr.broadcast(d.latitude, d.longitude)
    X, Y = TO_POLAR.transform(LO.values, LA.values)
    inside = (X >= minx) & (X <= maxx) & (Y >= miny) & (Y <= maxy)
    inside = xr.DataArray(inside, coords=LA.coords, dims=LA.dims)
    return d.where(inside, drop=True)