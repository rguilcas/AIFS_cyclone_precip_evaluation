"""
Turn a labelled lat/lon DataArray (0 = background, 1..n = objects, e.g. from
scipy.ndimage.label) into a GeoDataFrame with one polygon (or multipolygon) per object.

    from labels_to_gdf import labels_to_gdf
    gdf = labels_to_gdf(labels)                     # labels: DataArray (latitude, longitude)
    gdf = labels_to_gdf(labels, values=precip)      # + per-object precipitation stats
"""

import numpy as np
import geopandas as gpd
import xarray as xr
from rasterio.features import shapes
from rasterio.transform import from_origin
from shapely.geometry import shape
from shapely.ops import unary_union


def labels_to_gdf(labels, values=None, lat="latitude", lon="longitude", background=0,
                  connectivity=8, equal_area_crs="EPSG:6933"):
    labels = labels.transpose(lat, lon)
    if values is not None:
        values = values.transpose(lat, lon)
    # rasterio expects rows from north to south
    if labels[lat][0] < labels[lat][-1]:
        labels = labels.isel({lat: slice(None, None, -1)})
        if values is not None:
            values = values.isel({lat: slice(None, None, -1)})

    la, lo = labels[lat].values, labels[lon].values
    dlat, dlon = abs(la[1] - la[0]), abs(lo[1] - lo[0])
    transform = from_origin(lo[0] - dlon / 2, la[0] + dlat / 2, dlon, dlat)  # cell edges

    arr = np.nan_to_num(labels.values, nan=background).astype(np.int32)
    parts = {}
    for geom, val in shapes(arr, mask=arr != background, transform=transform,
                            connectivity=connectivity):
        parts.setdefault(int(val), []).append(shape(geom))

    ids = sorted(parts)
    gdf = gpd.GeoDataFrame({"label": ids},
                           geometry=[unary_union(parts[i]) for i in ids], crs="EPSG:4326")
    gdf["n_cells"] = [int((arr == i).sum()) for i in ids]
    gdf["area_km2"] = gdf.to_crs(equal_area_crs).area / 1e6
    c = gdf.to_crs(equal_area_crs).centroid.to_crs("EPSG:4326")
    gdf["centroid_lon"], gdf["centroid_lat"] = c.x, c.y

    if values is not None:
        v = values.values
        w = np.cos(np.deg2rad(la))[:, None] * np.ones_like(v)
        gdf["mean"] = [np.nansum((v * w)[arr == i]) / np.nansum(w[arr == i]) for i in ids]
        gdf["max"] = [np.nanmax(v[arr == i]) for i in ids]
        # rainfall-weighted centre (what SAL uses), as opposed to the geometric centroid
        LA, LO = np.meshgrid(la, lo, indexing="ij")
        ww = [np.nan_to_num(v * w)[arr == i] for i in ids]
        gdf["wcentre_lat"] = [np.sum(x * LA[arr == i]) / x.sum() for x, i in zip(ww, ids)]
        gdf["wcentre_lon"] = [np.sum(x * LO[arr == i]) / x.sum() for x, i in zip(ww, ids)]
    return gdf.set_index("label")


"""
Geodesic circles (true km radius) around cyclone centres, as a GeoDataFrame in EPSG:4326.

    from cyclone_buffers import cyclone_buffers
    gdf = cyclone_buffers(df_cycs, radius_km=df_cycs["radius"] * 1.5)   # or a fixed 1000
    hits = gpd.sjoin(objects_gdf, gdf, predicate="intersects")

Buffering in degrees (points.buffer(10)) would be wrong: a degree of longitude is
~110 km at 30N but ~40 km at 70N. Here each circle is built from geodesic points
at a fixed distance, so the radius is correct at every latitude.
"""

import numpy as np
import geopandas as gpd
import shapely
from pyproj import Geod
from shapely.geometry import box
from shapely.affinity import translate

GEOD = Geod(ellps="WGS84")


def _wrap_to_180(geom):
    """Split a polygon that extends past +-180 and move the outside part across."""
    minx, _, maxx, _ = geom.bounds
    if minx >= -180 and maxx <= 180:
        return geom
    parts = [geom.intersection(box(-180, -90, 180, 90))]
    if maxx > 180:
        parts.append(translate(geom.intersection(box(180, -90, 540, 90)), xoff=-360))
    if minx < -180:
        parts.append(translate(geom.intersection(box(-540, -90, -180, 90)), xoff=360))
    return shapely.union_all(parts)


def cyclone_buffers(df, radius_km, lon="lon", lat="lat", n_vertices=72, lon_range="-180"):
    """lon_range: '-180' for -180..180 output (default) or '360' for 0..360 output.
    Circles crossing the edge of that range are split into a MultiPolygon."""
    lo = ((np.asarray(df[lon], dtype=float) + 180) % 360) - 180
    la = np.asarray(df[lat], dtype=float)
    r = np.broadcast_to(np.asarray(radius_km, dtype=float), lo.shape) * 1000.0

    az = np.linspace(0, 360, n_vertices, endpoint=False)
    n = len(lo)
    plon, plat, _ = GEOD.fwd(np.repeat(lo, n_vertices), np.repeat(la, n_vertices),
                             np.tile(az, n), np.repeat(r, n_vertices))
    plon, plat = plon.reshape(n, n_vertices), plat.reshape(n, n_vertices)
    plon = lo[:, None] + ((plon - lo[:, None] + 180) % 360) - 180  # continuous around centre

    rings = np.stack([plon, plat], axis=-1)
    geoms = shapely.polygons(np.concatenate([rings, rings[:, :1]], axis=1))

    # Circles containing the pole cannot be a simple ring in lon/lat: use a polar cap
    dist_to_pole = np.array(GEOD.inv(lo, la, lo, np.full(n, 90.0))[2])
    pole = dist_to_pole < r
    for i in np.nonzero(pole)[0]:
        geoms[i] = box(-180 - 360, plat[i].min(), 180 + 360, 90)  # clipped by the wrap below

    if lon_range == "360":
        shift = lambda g: translate(_wrap_to_180(translate(g, xoff=-180)), xoff=180)
    else:
        shift = _wrap_to_180
    geoms = [shift(g) for g in geoms]

    out = gpd.GeoDataFrame(df.copy(), geometry=geoms, crs="EPSG:4326")
    out["buffer_km"] = r / 1000.0
    out["contains_pole"] = pole
    return out


"""
Frequency maps of heavy-precipitation events from a GeoDataFrame of event polygons.

    from event_density import event_frequency, plot_event_frequency
    freq = event_frequency(all_events, n_days=n_days_analysed)
    plot_event_frequency(freq)

Each polygon is burnt back onto the 0.25 deg grid and counted, so a grid cell's value
is the number of days it was inside a heavy-precipitation event. Polygons from
labels_to_gdf follow grid-cell edges, so this recovers the original cells exactly.
"""
"""
Frequency maps of heavy-precipitation events from a GeoDataFrame of event polygons.

    from event_density import event_frequency, plot_event_frequency
    freq = event_frequency(all_events, n_days=n_days_analysed)
    plot_event_frequency(freq)

Each polygon is burnt back onto the 0.25 deg grid and counted, so a grid cell's value
is the number of days it was inside a heavy-precipitation event. Polygons from
labels_to_gdf follow grid-cell edges, so this recovers the original cells exactly.
"""

import numpy as np
import xarray as xr
from rasterio.features import rasterize, MergeAlg
from rasterio.transform import from_origin
from shapely.affinity import translate
from shapely.geometry import box
import shapely


def _to_180(geom):
    """Bring a polygon into -180..180, splitting it if it crosses 180 (works for inputs
    in 0..360 or -180..180)."""
    minx, _, maxx, _ = geom.bounds
    if minx >= -180 and maxx <= 180:
        return geom
    parts = [geom.intersection(box(-180, -90, 180, 90)),
             translate(geom.intersection(box(180, -90, 540, 90)), xoff=-360),
             translate(geom.intersection(box(-540, -90, -180, 90)), xoff=360)]
    return shapely.union_all([p for p in parts if not p.is_empty])


def _count(geoms, lat_min, lat_max, lon_min, res):
    nlat = int(round((lat_max - lat_min) / res)) + 1
    nlon = int(round(360 / res))
    transform = from_origin(lon_min - res / 2, lat_max + res / 2, res, res)
    if len(geoms) == 0:
        return np.zeros((nlat, nlon), dtype=np.int32)
    return rasterize(((g, 1) for g in geoms), out_shape=(nlat, nlon), transform=transform,
                     merge_alg=MergeAlg.add, fill=0, dtype="int32")


def event_frequency(events, n_days, flag="cyclone_related", lat_min=30.0, lat_max=80.0,
                    res=0.25):
    """Returns a Dataset with, per grid cell:
       n_events / n_cyclone     number of event days (all / cyclone-related)
       freq / freq_cyclone      event days per 100 analysed days
       cyclone_fraction         share of event days that are cyclone-related
    n_days: number of days ANALYSED (including days without events), for normalising.
    Polygons may use 0..360 or -180..180 longitudes; the output grid is -180..180."""
    lon_min = -180.0
    geoms = np.array([_to_180(g) for g in events.geometry], dtype=object)
    is_cyc = events[flag].astype(bool).values
    n_all = _count(list(geoms), lat_min, lat_max, lon_min, res)
    n_cyc = _count(list(geoms[is_cyc]), lat_min, lat_max, lon_min, res)

    lat = lat_max - res * np.arange(n_all.shape[0])
    lon = lon_min + res * np.arange(n_all.shape[1])
    ds = xr.Dataset(
        {"n_events": (["latitude", "longitude"], n_all),
         "n_cyclone": (["latitude", "longitude"], n_cyc)},
        coords={"latitude": lat, "longitude": lon},
    )
    ds["freq"] = 100 * ds.n_events / n_days
    ds["freq_cyclone"] = 100 * ds.n_cyclone / n_days
    ds["cyclone_fraction"] = (ds.n_cyclone / ds.n_events).where(ds.n_events > 0)
    ds["freq"].attrs["units"] = ds["freq_cyclone"].attrs["units"] = "% of days"
    ds.attrs["n_days"] = n_days
    return ds


def plot_event_frequency(freq, min_events=5, vmax=None):
    """Three polar panels: all events, cyclone-related events, cyclone fraction.
    The fraction is shown only where a cell had at least `min_events` events."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt

    proj = ccrs.NorthPolarStereo()
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), subplot_kw={"projection": proj})
    vmax = vmax or float(freq.freq.quantile(0.995))
    panels = [
        (freq.freq.where(freq.n_events > 0), "All heavy-precip events",
         dict(cmap="viridis", vmin=0, vmax=vmax), "% of days"),
        (freq.freq_cyclone.where(freq.n_cyclone > 0), "Cyclone-related events",
         dict(cmap="viridis", vmin=0, vmax=vmax), "% of days"),
        (freq.cyclone_fraction.where(freq.n_events >= min_events),
         f"Cyclone fraction (≥{min_events} events)",
         dict(cmap="RdBu_r", vmin=0, vmax=1), "fraction"),
    ]
    for ax, (da, title, kw, label) in zip(axes, panels):
        ax.set_extent([-180, 180, float(freq.latitude.min()), 90], ccrs.PlateCarree())
        im = da.plot(ax=ax, transform=ccrs.PlateCarree(), add_colorbar=False, **kw)
        ax.coastlines(linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.2)
        ax.gridlines(linewidth=0.3)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, orientation="horizontal", shrink=0.8, pad=0.04, label=label)
    fig.suptitle(f"{freq.attrs.get('n_days', '?')} days analysed")
    return fig

import numpy as np
from shapely.geometry import box

KM_PER_DEG = 111.2

def event_box(obj, min_size_km=1000, margin_km=100):
    """Lat/lon box around one rain object: at least min_size_km wide and tall,
    and always containing the object plus margin_km on every side."""
    lat0, lon0 = obj["centroid_lat"], obj["centroid_lon"]
    coslat = np.cos(np.deg2rad(lat0))
    minx, miny, maxx, maxy = obj.geometry.bounds

    half_lat = max(min_size_km / 2 / KM_PER_DEG,
                   max(lat0 - miny, maxy - lat0) + margin_km / KM_PER_DEG)
    half_lon = max(min_size_km / 2 / (KM_PER_DEG * coslat),
                   max(lon0 - minx, maxx - lon0) + margin_km / (KM_PER_DEG * coslat))
    return box(lon0 - half_lon, lat0 - half_lat, lon0 + half_lon, lat0 + half_lat)

def cut(dataarray, box):
    minx, miny, maxx, maxy = box.bounds
    lat = slice(maxy, miny) if dataarray.latitude[0] > dataarray.latitude[-1] else slice(miny, maxy)
    return dataarray.sel(latitude=lat, longitude=slice(minx, maxx))