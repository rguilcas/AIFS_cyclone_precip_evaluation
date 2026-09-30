"""
Verification metrics for one precipitation event, forecast vs ERA5, as a function of lead.

    from event_metrics import event_metrics, plot_event_metrics
    df = event_metrics(fcst, obs, event_day="2024-10-01")
    plot_event_metrics(df)

fcst: DataArray (init_time, latitude, longitude), daily total in mm, for the event day
obs:  DataArray (latitude, longitude), ERA5 daily total in mm, same grid and lon convention
      (NaN outside the verification area, e.g. ocean, is fine: only points valid in both are used)

All area statistics use cos(latitude) weights.
"""

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import label, uniform_filter

KM_PER_DEG = 111.2













def _fss(f2, o2, m2, thr_f, thr_o, size):
    m = m2.astype(float)
    denom = uniform_filter(m, size=size, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        pf = uniform_filter(((f2 >= thr_f) & m2).astype(float), size=size, mode="constant") / denom
        po = uniform_filter(((o2 >= thr_o) & m2).astype(float), size=size, mode="constant") / denom
    pf, po = pf[m2], po[m2]
    ref = np.mean(pf ** 2) + np.mean(po ** 2)
    return 1 - np.mean((pf - po) ** 2) / ref if ref > 0 else np.nan

def _wmean(x, w):
    return np.sum(w * x) / np.sum(w)


def _contingency(f, o, w, thr):
    fy, oy = f >= thr, o >= thr
    hits = np.sum(w * (fy & oy))
    misses = np.sum(w * (~fy & oy))
    fas = np.sum(w * (fy & ~oy))
    total = np.sum(w)
    hits_random = (hits + misses) * (hits + fas) / total
    nan = np.nan
    return {
        f"POD_{thr}": hits / (hits + misses) if hits + misses > 0 else nan,
        f"FAR_{thr}": fas / (hits + fas) if hits + fas > 0 else nan,
        f"CSI_{thr}": hits / (hits + misses + fas) if hits + misses + fas > 0 else nan,
        f"ETS_{thr}": ((hits - hits_random) / (hits + misses + fas - hits_random)
                       if hits + misses + fas - hits_random > 0 else nan),
        f"freqbias_{thr}": (hits + fas) / (hits + misses) if hits + misses > 0 else nan,
    }



def _centroid(x, lat, lon, w, thr):
    sel = x >= thr
    if not sel.any():
        return np.nan, np.nan
    ww = w[sel] * x[sel]
    return np.sum(ww * lat[sel]) / ww.sum(), np.sum(ww * lon[sel]) / ww.sum()


def _gc_km(lat1, lon1, lat2, lon2):
    if np.isnan([lat1, lon1, lat2, lon2]).any():
        return np.nan
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dl = np.deg2rad(lon2 - lon1)
    c = np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(dl)
    return np.arccos(np.clip(c, -1, 1)) * 6371.0


def _sal_objects(r2, m2, w2, f_thr=1 / 15, wet=0.1):
    """Objects as in Wernli et al. (2008): connected areas with R >= f * R95, where R95 is
    the 95th percentile of wet (> 0.1 mm) values in the domain. 8-connectivity."""
    wet_vals = r2[m2 & (r2 > wet)]
    if wet_vals.size == 0:
        return np.zeros_like(r2, dtype=int), 0
    thr = f_thr * np.percentile(wet_vals, 95)
    return label(m2 & (r2 >= thr), structure=np.ones((3, 3)))


def _sal_field(r2, m2, w2, x2, y2):
    """Domain mean, centre of mass, and the object quantities r and V for one field."""
    rw = np.where(m2, r2 * w2, 0.0)
    total = rw.sum()
    mean = total / np.where(m2, w2, 0).sum()
    if total <= 0:
        return mean, (np.nan, np.nan), np.nan, np.nan
    com = (np.sum(rw * x2) / total, np.sum(rw * y2) / total)
    lab, n = _sal_objects(r2, m2, w2)
    if n == 0:
        return mean, com, np.nan, np.nan
    Rn = np.array([rw[lab == k].sum() for k in range(1, n + 1)])
    xn = np.array([np.sum(rw[lab == k] * x2[lab == k]) / Rn[k - 1] for k in range(1, n + 1)])
    yn = np.array([np.sum(rw[lab == k] * y2[lab == k]) / Rn[k - 1] for k in range(1, n + 1)])
    Rmax = np.array([r2[lab == k].max() for k in range(1, n + 1)])
    r = np.sum(Rn * np.hypot(xn - com[0], yn - com[1])) / Rn.sum()
    V = np.sum(Rn * (Rn / Rmax)) / Rn.sum()
    return mean, com, r, V


def sal(f2, o2, m2, lat2, lon2, w2):
    """Structure, Amplitude, Location (Wernli et al. 2008, MWR 136, 4470).

    A, S in [-2, 2], L in [0, 2]; 0 is perfect. Positive A: too much precipitation;
    positive S: objects too large/flat; negative S: too small/peaked.
    Distances use a local equirectangular projection (km); d is the largest distance
    across the verification domain (diagonal of its bounding box).
    """
    lat0 = np.deg2rad(lat2[m2].mean())
    x2 = np.deg2rad(lon2) * np.cos(lat0) * 6371.0
    y2 = np.deg2rad(lat2) * 6371.0
    d = np.hypot(x2[m2].max() - x2[m2].min(), y2[m2].max() - y2[m2].min())

    Df, cf, rf, Vf = _sal_field(f2, m2, w2, x2, y2)
    Do, co, ro, Vo = _sal_field(o2, m2, w2, x2, y2)
    A = (Df - Do) / (0.5 * (Df + Do)) if Df + Do > 0 else np.nan
    S = (Vf - Vo) / (0.5 * (Vf + Vo))
    L1 = np.hypot(cf[0] - co[0], cf[1] - co[1]) / d
    L2 = 2 * abs(rf - ro) / d
    return {"SAL_S": S, "SAL_A": A, "SAL_L": L1 + L2, "SAL_L1": L1, "SAL_L2": L2}


def event_metrics(fcst, obs, event_day=None, thresholds=(10, 20, 40),
                  fss_threshold=20, fss_scales_km=(25, 50, 100, 200, 400),
                  init_dim="init_time"):
    obs, fcst = xr.align(obs, fcst, join="exact")  # grids must match exactly
    mask2 = (obs.notnull() & fcst.notnull().all(init_dim)).values
    lat2, lon2 = np.meshgrid(obs.latitude.values, obs.longitude.values, indexing="ij")
    w2 = np.cos(np.deg2rad(lat2))
    o2 = np.nan_to_num(obs.values)
    o, w, lat, lon = o2[mask2], w2[mask2], lat2[mask2], lon2[mask2]

    dlat = abs(float(obs.latitude[1] - obs.latitude[0]))
    dlon = abs(float(obs.longitude[1] - obs.longitude[0]))
    coslat = np.cos(np.deg2rad(float(obs.latitude.mean())))
    sizes = {s: (max(1, int(round(s / (dlat * KM_PER_DEG))) | 1),
                 max(1, int(round(s / (dlon * KM_PER_DEG * coslat))) | 1)) for s in fss_scales_km}

    oc = _centroid(o, lat, lon, w, fss_threshold)
    rows = []
    for init in fcst[init_dim].values:
        f2 = np.nan_to_num(fcst.sel({init_dim: init}).values)
        f = f2[mask2]
        wf, wo = w * f, w * o
        r = {init_dim: pd.Timestamp(init)}
        if event_day is not None:
            r["lead_days"] = (pd.Timestamp(event_day) + pd.Timedelta(days=1)
                              - pd.Timestamp(init)) / pd.Timedelta(days=1)  # to end of day
        r["obs_mean"] = _wmean(o, w)
        r["fcst_mean"] = _wmean(f, w)
        r["amount_ratio"] = wf.sum() / wo.sum() if wo.sum() > 0 else np.nan
        r["MAE"] = _wmean(np.abs(f - o), w)
        r["RMSE"] = np.sqrt(_wmean((f - o) ** 2, w))
        fa, oa = f - r["fcst_mean"], o - r["obs_mean"]
        r["pattern_corr"] = _wmean(fa * oa, w) / np.sqrt(_wmean(fa ** 2, w) * _wmean(oa ** 2, w))
        r["max_ratio"] = f.max() / o.max() if o.max() > 0 else np.nan
        for thr in thresholds:
            r.update(_contingency(f, o, w, thr))
        # main: placement only (95th percentile of each field separately)
        tf, to = np.percentile(f, 95), np.percentile(o, 95)
        fss_p = {s: _fss(f2, o2, mask2, tf, to, size) for s, size in sizes.items()}
        for s, v in fss_p.items():
            r[f"FSS_p95_{s}km"] = v
        useful = 0.5 + 0.05 / 2
        ok = [s for s in sorted(fss_p) if fss_p[s] >= useful]
        r["skilful_scale_km"] = ok[0] if ok else np.inf

        # secondary: fixed threshold, amount and placement together
        for s, size in sizes.items():
            r[f"FSS_{fss_threshold}mm_{s}km"] = _fss(f2, o2, mask2, fss_threshold, fss_threshold, size)
        fc = _centroid(f, lat, lon, w, fss_threshold)
        r[f"centroid_dist_km_{fss_threshold}mm"] = _gc_km(*oc, *fc)
        r.update(sal(f2, o2, mask2, lat2, lon2, w2))
        rows.append(r)
    return pd.DataFrame(rows).set_index(init_dim)


def plot_event_metrics(df, thresholds=(10, 20, 40)):
    import matplotlib.pyplot as plt
    x = df["lead_days"] if "lead_days" in df else np.arange(len(df))
    fig, ax = plt.subplots(2, 2, figsize=(11, 7), sharex=True)

    ax[0, 0].plot(x, df["amount_ratio"], "o-", label="area-total ratio")
    ax[0, 0].plot(x, df["max_ratio"], "s--", label="max ratio")
    ax[0, 0].axhline(1, color="k", lw=0.8)
    ax[0, 0].set_title("Amplitude (1 = perfect)")
    ax[0, 0].legend()

    ax[0, 1].plot(x, df["pattern_corr"], "o-")
    ax[0, 1].set_title("Pattern correlation")

    for thr in thresholds:
        ax[1, 0].plot(x, df[f"ETS_{thr}"], "o-", label=f"≥{thr} mm")
    ax[1, 0].set_title("Equitable threat score")
    ax[1, 0].legend()

    for c in [c for c in df.columns if c.startswith("FSS_")]:
        ax[1, 1].plot(x, df[c], "o-", label=c.split("_")[-1])
    ax[1, 1].set_title("FSS by neighbourhood scale")
    ax[1, 1].legend()

    for a in ax[1]:
        a.set_xlabel("lead time to end of event day (days)")
        a.invert_xaxis()
    fig.tight_layout()
    return fig


def plot_sal(df, ax=None):
    """SAL diagram: S (x) vs A (y), coloured by L, labelled with lead time in days."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5.5))
    sc = ax.scatter(df["SAL_S"], df["SAL_A"], c=df["SAL_L"], cmap="viridis_r",
                    vmin=0, vmax=max(0.5, float(np.nanmax(df["SAL_L"]))), s=70,
                    edgecolor="k", zorder=3)
    if "lead_days" in df:
        for s, a, lead in zip(df["SAL_S"], df["SAL_A"], df["lead_days"]):
            ax.annotate(f"{lead:g}", (s, a), textcoords="offset points", xytext=(5, 4), fontsize=8)
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_xlabel("S  (− too small/peaked · + too large/flat)")
    ax.set_ylabel("A  (− too dry · + too wet)")
    ax.set_title("SAL (labels: lead time in days)")
    plt.colorbar(sc, ax=ax, label="L")
    return ax