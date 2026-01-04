# simulazione_dash.py
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Dict
import re
import itertools
from functools import lru_cache
import uuid

import numpy as np
import pandas as pd

from dash import Dash, dcc, html, dash_table, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from qualificateML import build_tab_qual_ml
from teamsimulation import build_team_simulation_tab  # <-- NUOVO


# ===================== CONFIG CARTELLA =====================
base_dir = Path(__file__).resolve().parent
sim_dir = base_dir / "Simulazione"
target_dir = sim_dir if sim_dir.exists() and sim_dir.is_dir() else base_dir

# ===================== OPERATORI TARGET (canonici) =====================
TARGET_OPERATORS_CANON = [
    "Fabio Galli",
    "Roberta Messina",
    "Sabrina Mastinu",
    "Paolo Pidatella",
    "Fabrizio Meola",
    "Lonetti Maria Teresa",
    "Luigi Berti",
    "Agata Arena",
]

# (per matching file: gestiamo inversi e alias comuni)
TARGET_ALIASES = {
    "Fabio Galli": ["Galli Fabio", "Fabio Galli"],
    "Roberta Messina": ["Messina Roberta", "Roberta Messina"],
    "Sabrina Mastinu": ["Mastinu Sabrina", "Sabrina Mastinu"],
    "Paolo Pidatella": ["Pidatella Paolo", "Paolo Pidatella"],
    "Fabrizio Meola": ["Meola Fabrizio", "Fabrizio Meola"],
    "Lonetti Maria Teresa": ["Lonetti Maria Teresa", "Maria Teresa Lonetti"],
    "Luigi Berti": ["Berti Luigi", "Luigi Berti"],
    "Agata Arena": ["Arena Agata", "Agata Arena", "Arena"],
}

# ===================== COLONNE (richieste) =====================
METRIC_COLS = [
    "Lavorazione  generale",
    "Lavorazione contatti",
    "Lavorazione varie",
    "In chiamata",
    "Conversazione",
    "Nr. chiamate effettuate",
    "Nr. chiamate con risposta",
    "Processati",
    "Positivi",
    "Positivi confermati",
    "Positivi per ora di lavoro contatti",
    "Positivi per ora di lavoro generale",
    "Positivi per ora di conversazione",
    "Processati per positivo",
]

TIME_ACTIVITY_COLS = [
    "Lavorazione  generale",
    "Lavorazione contatti",
    "Lavorazione varie",
    "In chiamata",
    "Conversazione",
    "In attesa di chiamata",  # opzionale
]

DERIVED_COLS_DAILY = [
    "Giorni",
    "Ore totali (giorno medio)",
    "% Attività",
    "Rapporto InChiamata/Generale",
    "RedL%",
    "% risposta",
    "Processati per positivo",
]

# ===================== THEME =====================
BG = "#0b0f14"
CARD = "#0f1621"
GRID = "rgba(255,255,255,0.10)"
TXT = "#cfd8dc"
MUTED = "#9fb0b8"
ACC_GREEN = "#39ff14"
ACC_ORANGE = "#ff7a00"
ACC_FUCHSIA = "#ff00ff"

HEAT_FLUO = [
    [0.0, "#ff7a00"],
    [0.5, "#0b0f14"],
    [1.0, "#39ff14"],
]


# ===================== UTILS: normalizzazione =====================
def normalize_text(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    return " ".join(s.split())


def tokens_of(name: str) -> List[str]:
    return [t for t in normalize_text(name).split(" ") if t]


def norm_colname(s: str) -> str:
    return " ".join(str(s).strip().lower().split())


def build_token_regex(target: str) -> str:
    toks = tokens_of(target)
    parts = [rf"(?=.*\b{re.escape(t)}\b)" for t in toks]
    return "".join(parts) + r".*"


def norm_filename(name: str) -> str:
    n = normalize_text(name)
    n = n.replace("csv", "").strip()
    return n


def canonical_operator_name(raw_name: str) -> str:
    raw = str(raw_name) if raw_name is not None else ""
    raw_toks = set(tokens_of(raw))
    if not raw_toks:
        return raw

    raw_norm = normalize_text(raw)
    for canon, aliases in TARGET_ALIASES.items():
        for a in aliases:
            if normalize_text(a) == raw_norm:
                return canon

    for canon, aliases in TARGET_ALIASES.items():
        for a in [canon] + list(aliases):
            if set(tokens_of(a)) == raw_toks:
                return canon

    for canon in TARGET_OPERATORS_CANON:
        if set(tokens_of(canon)) == raw_toks:
            return canon

    return raw


def filter_to_target_ops(df_in: pd.DataFrame) -> pd.DataFrame:
    if df_in.empty:
        return df_in
    s = df_in["Operatore"].astype(str).map(normalize_text)
    mask = pd.Series(False, index=df_in.index)
    for t in TARGET_OPERATORS_CANON:
        mask = mask | s.str.contains(build_token_regex(t), regex=True, na=False)
    return df_in.loc[mask].copy()


# ===================== LETTURA CSV =====================
def read_csv_best(path: Path) -> Optional[pd.DataFrame]:
    best_df = None
    best_ncol = -1
    for enc in ["utf-8-sig", "utf-8", "latin1"]:
        for sep in [";", ",", "\t", "|"]:
            try:
                df = pd.read_csv(path, sep=sep, encoding=enc, engine="python")
                if df.shape[1] > best_ncol:
                    best_df = df
                    best_ncol = df.shape[1]
            except Exception:
                continue
    return best_df


@lru_cache(maxsize=4)
def load_all_csv_cached(folder: str) -> Tuple[pd.DataFrame, List[str]]:
    p = Path(folder)
    csvs = sorted(
        [f for f in p.iterdir() if f.is_file() and f.suffix.lower() == ".csv"],
        key=lambda x: x.name.lower(),
    )
    tables = []
    used_files: List[str] = []
    for f in csvs:
        df = read_csv_best(f)
        if df is None or df.empty:
            continue
        df = df.copy()
        df["__OperatoreC"] = df.iloc[:, 2] if df.shape[1] >= 3 else pd.NA  # colonna C
        df["__source_file"] = f.name
        tables.append(df)
        used_files.append(f.name)
    if not tables:
        return pd.DataFrame(), used_files
    out = pd.concat(tables, ignore_index=True, sort=False)
    return out, used_files


# ===================== PARSING TEMPO/NUMERI =====================
_TIME_RE = re.compile(r"^\s*\d{1,2}:\d{2}(:\d{2})?\s*$")


def is_time_like_series(s: pd.Series, sample: int = 150) -> bool:
    vals = s.dropna().astype(str).head(sample).tolist()
    for v in vals:
        v = v.strip()
        if ":" in v and _TIME_RE.match(v):
            return True
    return False


def parse_duration_to_hours(v) -> float:
    if pd.isna(v):
        return np.nan
    if isinstance(v, (int, float, np.number)):
        return float(v)
    s = str(v).strip()
    if s == "":
        return np.nan
    if ":" in s:
        try:
            td = pd.to_timedelta(s)
            return td.total_seconds() / 3600.0
        except Exception:
            parts = s.split(":")
            try:
                parts = [float(p) for p in parts]
                if len(parts) == 3:
                    h, m, sec = parts
                elif len(parts) == 2:
                    h, m = parts
                    sec = 0.0
                else:
                    return np.nan
                return (h * 3600 + m * 60 + sec) / 3600.0
            except Exception:
                return np.nan
    return np.nan


def coerce_numeric(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    s = s.str.replace(r"[^\d\.,\-]", "", regex=True)

    both = s.str.contains(r"\.", na=False) & s.str.contains(",", na=False)
    s.loc[both] = s.loc[both].str.replace(".", "", regex=False).str.replace(",", ".", regex=False)

    only_comma = s.str.contains(",", na=False) & ~s.str.contains(r"\.", na=False)
    s.loc[only_comma] = s.loc[only_comma].str.replace(",", ".", regex=False)

    return pd.to_numeric(s, errors="coerce")


def coerce_time_to_hours_auto(series: pd.Series) -> pd.Series:
    if is_time_like_series(series):
        return series.map(parse_duration_to_hours)

    num = coerce_numeric(series)
    arr = num.to_numpy(dtype=float)
    med = float(np.nanmedian(arr)) if np.isfinite(arr).any() else np.nan
    if not np.isfinite(med):
        return num
    if med <= 24.0:
        return num
    if med <= 24.0 * 60.0:
        return num / 60.0
    return num / 3600.0


def coerce_metric(series: pd.Series) -> pd.Series:
    if is_time_like_series(series):
        return series.map(parse_duration_to_hours)
    return coerce_numeric(series)


# ===================== DATE: colonna data =====================
def pick_date_column(df: pd.DataFrame) -> Optional[str]:
    if df.empty:
        return None
    for cand in ["Data", "data", "DATA", "Date", "date"]:
        if cand in df.columns:
            return cand

    skip = {"__OperatoreC", "__source_file", "_op_norm", "Operatore"}
    best_col = None
    best_ratio = 0.0
    for c in df.columns:
        if c in skip:
            continue
        if df[c].dtype.kind in "biufc":
            continue
        parsed = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
        ratio = float(parsed.notna().mean())
        if ratio > best_ratio:
            best_ratio = ratio
            best_col = c
    return best_col if best_ratio >= 0.60 else None


def add_parsed_date(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = df.copy()
    dt = pd.to_datetime(out[date_col], errors="coerce", dayfirst=True)
    out["__date__"] = dt.dt.date
    return out


# ===================== MATH: distanze/similarità =====================
def standardize_frame(df_in: pd.DataFrame) -> pd.DataFrame:
    X = df_in.astype(float)
    mu = X.mean(axis=0)
    sd = X.std(axis=0).replace(0, 1.0)
    return (X - mu) / sd


def cosine_similarity_matrix(Z: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Zn = Z / norms
    return Zn @ Zn.T


def euclidean_distance_matrix(Z: np.ndarray) -> np.ndarray:
    return np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(axis=2))


def pairwise_summary_entities(features: pd.DataFrame, entity_name: str, top_k: int = 5) -> pd.DataFrame:
    idx = list(features.index)
    Z = standardize_frame(features).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()

    cos = cosine_similarity_matrix(Z)
    dists = euclidean_distance_matrix(Z)

    cols = list(features.columns)
    rows = []
    for i, j in itertools.combinations(range(len(idx)), 2):
        diffs = np.abs(Z[i] - Z[j])
        top_idx = np.argsort(diffs)[::-1][:top_k]
        top_feats = [f"{cols[k]} ({diffs[k]:.2f})" for k in top_idx]
        rows.append(
            {
                f"{entity_name} A": idx[i],
                f"{entity_name} B": idx[j],
                "Distanza (euclid, z-score)": float(dists[i, j]),
                "Similarità (coseno, z-score)": float(cos[i, j]),
                "Top divergenze": " | ".join(top_feats),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["Distanza (euclid, z-score)", "Similarità (coseno, z-score)"],
        ascending=[True, False],
    )


# ===================== PROFILI: MEDIA GIORNALIERA CORRETTA =====================
def build_daily_mean_profiles(
    df_rows: pd.DataFrame,
    col_map_norm: dict,
    time_cols_present: List[str],
    count_cols_present: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cols = time_cols_present + count_cols_present

    daily = (
        df_rows.groupby(["Operatore", "__date__"], dropna=False)[cols]
        .sum(numeric_only=True, min_count=1)
        .reset_index()
    )
    daily = daily.dropna(subset=["__date__"])
    if daily.empty:
        return pd.DataFrame(), pd.DataFrame()

    g = daily.groupby("Operatore", dropna=False)
    profile_sum = g[cols].sum(numeric_only=True, min_count=1)
    n_days = g["__date__"].nunique(dropna=True).astype(float).replace(0, np.nan)

    profile_mean = profile_sum.div(n_days, axis=0)
    profile_mean["Giorni"] = n_days

    ore_medie = profile_mean[time_cols_present].sum(axis=1) if time_cols_present else pd.Series(np.nan, index=profile_mean.index)
    profile_mean["Ore totali (giorno medio)"] = ore_medie.replace(0, np.nan)

    for c in time_cols_present:
        denom = profile_mean["Ore totali (giorno medio)"]
        profile_mean[f"% {c}"] = (profile_mean[c] / denom) * 100.0

    def get_col(name: str) -> Optional[str]:
        return col_map_norm.get(norm_colname(name))

    c_incall = get_col("In chiamata")
    if c_incall is not None and f"% {c_incall}" in profile_mean.columns:
        profile_mean["% Attività"] = profile_mean[f"% {c_incall}"]
    else:
        profile_mean["% Attività"] = np.nan

    c_gen = get_col("Lavorazione  generale")
    if c_incall is not None and c_gen is not None and c_incall in profile_mean.columns and c_gen in profile_mean.columns:
        profile_mean["Rapporto InChiamata/Generale"] = profile_mean[c_incall] / profile_mean[c_gen].replace(0, np.nan)
    else:
        profile_mean["Rapporto InChiamata/Generale"] = np.nan

    c_eff = get_col("Nr. chiamate effettuate")
    c_risp = get_col("Nr. chiamate con risposta")
    if c_eff is not None and c_risp is not None and c_eff in profile_sum.columns and c_risp in profile_sum.columns:
        profile_mean["% risposta"] = (profile_sum[c_risp] / profile_sum[c_eff].replace(0, np.nan)) * 100.0
    else:
        profile_mean["% risposta"] = np.nan

    c_proc = get_col("Processati")
    c_pos = get_col("Positivi")
    c_pos_conf = get_col("Positivi confermati")
    c_pos_use = c_pos if (c_pos is not None and c_pos in profile_sum.columns) else c_pos_conf

    if c_proc is not None and c_pos_use is not None and c_proc in profile_sum.columns and c_pos_use in profile_sum.columns:
        profile_mean["RedL%"] = (profile_sum[c_pos_use] / profile_sum[c_proc].replace(0, np.nan)) * 100.0
        profile_mean["Processati per positivo"] = (profile_sum[c_proc] / profile_sum[c_pos_use].replace(0, np.nan))
    else:
        profile_mean["RedL%"] = np.nan
        profile_mean["Processati per positivo"] = np.nan

    return profile_mean, profile_sum


# ===================== DASH TABLE =====================
def make_table(
    df_in: pd.DataFrame,
    page_size: int = 18,
    table_id: Optional[str] = None,
    max_width_px: int = 260,
    extra_style_data_conditional: Optional[List[dict]] = None,
) -> dash_table.DataTable:
    df2 = df_in.copy().reset_index(drop=True)

    if table_id is None:
        table_id = f"tbl-{uuid.uuid4().hex}"

    tooltip_header = {c: {"value": str(c), "type": "text"} for c in df2.columns}

    sdc = [
        {"if": {"row_index": "odd"}, "backgroundColor": "#0d1420"},
        {"if": {"state": "active"}, "backgroundColor": "#142235", "border": f"1px solid {ACC_ORANGE}"},
    ]
    if extra_style_data_conditional:
        sdc = sdc + extra_style_data_conditional

    return dash_table.DataTable(
        id=table_id,
        columns=[{"name": c, "id": c} for c in df2.columns],
        data=df2.to_dict("records"),
        page_size=page_size,
        sort_action="native",
        filter_action="native",
        style_table={
            "overflowX": "auto",
            "border": f"1px solid {GRID}",
            "borderRadius": "12px",
            "backgroundColor": CARD,
            "width": "100%",
        },
        style_cell={
            "backgroundColor": CARD,
            "color": TXT,
            "fontFamily": "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Courier New', monospace",
            "fontSize": "11px",
            "padding": "4px 6px",
            "whiteSpace": "nowrap",
            "overflow": "hidden",
            "textOverflow": "ellipsis",
            "border": f"1px solid {GRID}",
            "maxWidth": f"{max_width_px}px",
        },
        style_header={
            "backgroundColor": "#121c2a",
            "color": TXT,
            "fontWeight": "700",
            "fontSize": "10px",
            "whiteSpace": "normal",
            "height": "auto",
            "border": f"1px solid {GRID}",
        },
        style_filter={
            "backgroundColor": CARD,
            "color": TXT,
            "border": f"1px solid {GRID}",
            "fontSize": "10px",
        },
        style_data_conditional=sdc,
        tooltip_header=tooltip_header,
        tooltip_delay=0,
        tooltip_duration=None,
    )


# ===================== COLOR SCALE per tabelle correlazione =====================
def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _hex_to_rgb(h: str) -> Tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _interp_hex(c1: str, c2: str, t: float) -> str:
    r1, g1, b1 = _hex_to_rgb(c1)
    r2, g2, b2 = _hex_to_rgb(c2)
    r = int(_lerp(r1, r2, t))
    g = int(_lerp(g1, g2, t))
    b = int(_lerp(b1, b2, t))
    return _rgb_to_hex((r, g, b))


def corr_color(val: float) -> str:
    if not np.isfinite(val):
        return CARD
    v = float(np.clip(val, -1, 1))
    if v >= 0:
        return _interp_hex("#0b0f14", ACC_GREEN, v)
    return _interp_hex("#0b0f14", ACC_ORANGE, abs(v))


def decorr_color(val: float, vmax: float = 2.0) -> str:
    if not np.isfinite(val):
        return CARD
    v = float(np.clip(val, 0, vmax))
    if v <= 1.0:
        return _interp_hex(ACC_GREEN, "#0b0f14", v / 1.0)
    t = (v - 1.0) / (vmax - 1.0 + 1e-9)
    return _interp_hex("#0b0f14", ACC_ORANGE, t)


def matrix_to_colored_table(df_mat: pd.DataFrame, first_col_name: str, mode: str) -> dash_table.DataTable:
    df_show = df_mat.copy()
    df_show.insert(0, first_col_name, df_show.index.astype(str))
    numeric_cols = [c for c in df_show.columns if c != first_col_name]

    sdc = []
    for col in numeric_cols:
        for i in range(len(df_show)):
            v = df_show.iloc[i][col]
            if mode == "corr":
                bg = corr_color(float(v)) if pd.notna(v) else CARD
            else:
                bg = decorr_color(float(v)) if pd.notna(v) else CARD
            sdc.append(
                {
                    "if": {"filter_query": f"{{{first_col_name}}} = '{df_show.iloc[i][first_col_name]}'", "column_id": col},
                    "backgroundColor": bg,
                    "color": TXT,
                }
            )

    df_fmt = df_show.copy()
    for c in numeric_cols:
        df_fmt[c] = pd.to_numeric(df_fmt[c], errors="coerce").round(4)

    return make_table(df_fmt, page_size=18, max_width_px=220, extra_style_data_conditional=sdc)


# ===================== PLOTLY HELPERS =====================
def base_layout(fig: go.Figure, title: str, height: int = 440) -> go.Figure:
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=18, r=18, t=48, b=18),
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        font=dict(color=TXT, size=11),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, tickfont=dict(size=10)),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, tickfont=dict(size=10)),
        legend=dict(font=dict(size=10)),
    )
    return fig


def fig_heatmap(matrix: np.ndarray, labels: List[str], title: str, zmin=None, zmax=None) -> go.Figure:
    if matrix.size == 0:
        fig = go.Figure()
        fig.add_annotation(text="Dati insufficienti", x=0.5, y=0.5, showarrow=False, font=dict(color=TXT))
        return base_layout(fig, title)
    z = np.array(matrix, dtype=float)
    if np.all(np.isnan(z)):
        z = np.zeros_like(z)
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            zmin=zmin,
            zmax=zmax,
            colorscale=HEAT_FLUO,
            hoverongaps=False,
            colorbar=dict(thickness=12),
        )
    )
    fig.update_xaxes(tickangle=45, tickfont=dict(size=9))
    fig.update_yaxes(tickfont=dict(size=9))
    return base_layout(fig, title, height=560)


def fig_operator_mix_trend(
    title: str,
    mix_labels: List[str],
    pos_vals: List[float],
    proc_vals: List[float],
    red_vals: List[float],
    pos_std: Optional[float],
    proc_std: Optional[float],
    red_std: Optional[float],
) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=mix_labels, y=pos_vals, mode="lines+markers", name="Positivi (scenario)", line=dict(width=2, color=ACC_ORANGE)))
    fig.add_trace(go.Scatter(x=mix_labels, y=proc_vals, mode="lines+markers", name="Processati (scenario)", line=dict(width=2, color=ACC_FUCHSIA)))
    fig.add_trace(go.Scatter(x=mix_labels, y=red_vals, mode="lines+markers", name="Red% (scenario)", yaxis="y2", line=dict(width=2, color=ACC_GREEN)))

    if pos_std is not None and np.isfinite(pos_std):
        fig.add_trace(
            go.Scatter(
                x=["Standard"],
                y=[pos_std],
                mode="markers+text",
                text=[f"{pos_std:.0f}"],
                textposition="top center",
                name="Positivi attuali (STANDARD)",
                marker=dict(size=10, color=ACC_GREEN),
            )
        )

    fig.update_xaxes(title_text="Mix (Qualificate/Freddo)")
    fig.update_yaxes(title_text="Positivi / Processati")
    fig.update_layout(
        yaxis2=dict(
            title="Red% (scenario)",
            overlaying="y",
            side="right",
            gridcolor="rgba(0,0,0,0)",
            tickfont=dict(size=10),
        )
    )
    return base_layout(fig, title, height=460)


# ===================== PROFILING (totali) + DET MIX =====================
def compute_profile_totals_table(df_sub: pd.DataFrame, col_map_norm: dict, operator_col: str = "Operatore") -> pd.DataFrame:
    """
    Profilo TOTALI (periodo del CSV):
      - Totale chiamate effettuate
      - Totale chiamate con risposta
      - Processati
      - Positivi
      - Ore lavoro (tot)  (da Lavorazione generale)
      - Metriche derivate: % risposta, Resa oraria, Red%, Processati per positivo, Chiamate per positivo
    """
    def get_col(name: str) -> Optional[str]:
        return col_map_norm.get(norm_colname(name))

    c_eff = get_col("Nr. chiamate effettuate")
    c_risp = get_col("Nr. chiamate con risposta")
    c_proc = get_col("Processati")
    c_pos = get_col("Positivi")
    c_pos_conf = get_col("Positivi confermati")
    c_lavgen = get_col("Lavorazione  generale")

    if c_eff is None or c_risp is None or c_proc is None or c_lavgen is None:
        return pd.DataFrame()

    tmp = df_sub.copy()
    tmp[c_eff] = coerce_numeric(tmp[c_eff])
    tmp[c_risp] = coerce_numeric(tmp[c_risp])
    tmp[c_proc] = coerce_numeric(tmp[c_proc])
    tmp[c_lavgen] = coerce_time_to_hours_auto(tmp[c_lavgen])

    if c_pos is None and c_pos_conf is None:
        tmp["__pos__"] = np.nan
    else:
        if c_pos is None:
            tmp["__pos__"] = coerce_numeric(tmp[c_pos_conf])
        elif c_pos_conf is None:
            tmp["__pos__"] = coerce_numeric(tmp[c_pos])
        else:
            merged = tmp[c_pos].where(tmp[c_pos].notna(), tmp[c_pos_conf])
            tmp["__pos__"] = coerce_numeric(merged)

    g = tmp.groupby(operator_col, dropna=False)
    calls = g[c_eff].sum(min_count=1)
    ans = g[c_risp].sum(min_count=1)
    proc = g[c_proc].sum(min_count=1)
    pos = g["__pos__"].sum(min_count=1)
    ore = g[c_lavgen].sum(min_count=1)

    resp_pct = (ans / calls.replace(0, np.nan)) * 100.0
    red_pct = (pos / proc.replace(0, np.nan)) * 100.0
    resa_h = (pos / ore.replace(0, np.nan))
    proc_per_pos = (proc / pos.replace(0, np.nan))
    calls_per_pos = (calls / pos.replace(0, np.nan))

    out = pd.DataFrame(
        {
            "Operatore": calls.index.astype(str),
            "Totale chiamate effettuate": calls.values,
            "Totale chiamate con risposta": ans.values,
            "Positivi": pos.values,
            "Processati": proc.values,
            "Ore lavoro (tot)": ore.values,
            "% risposta": resp_pct.values,
            "Resa oraria": resa_h.values,
            "Red%": red_pct.values,
            "Processati per positivo": proc_per_pos.values,
            "Chiamate per positivo": calls_per_pos.values,
        }
    )

    # TEAM (somma)
    calls_T = float(np.nansum(calls.values))
    ans_T = float(np.nansum(ans.values))
    proc_T = float(np.nansum(proc.values))
    pos_T = float(np.nansum(pos.values))
    ore_T = float(np.nansum(ore.values))

    team = {
        "Operatore": "Team",
        "Totale chiamate effettuate": calls_T,
        "Totale chiamate con risposta": ans_T,
        "Positivi": pos_T,
        "Processati": proc_T,
        "Ore lavoro (tot)": ore_T,
        "% risposta": (ans_T / calls_T * 100.0) if calls_T else np.nan,
        "Resa oraria": (pos_T / ore_T) if ore_T else np.nan,
        "Red%": (pos_T / proc_T * 100.0) if proc_T else np.nan,
        "Processati per positivo": (proc_T / pos_T) if pos_T else np.nan,
        "Chiamate per positivo": (calls_T / pos_T) if pos_T else np.nan,
    }
    out = pd.concat([out, pd.DataFrame([team])], ignore_index=True)

    # formatting
    out["Totale chiamate effettuate"] = pd.to_numeric(out["Totale chiamate effettuate"], errors="coerce").round(0).astype("Int64")
    out["Totale chiamate con risposta"] = pd.to_numeric(out["Totale chiamate con risposta"], errors="coerce").round(0).astype("Int64")
    out["Positivi"] = pd.to_numeric(out["Positivi"], errors="coerce").round(0).astype("Int64")
    out["Processati"] = pd.to_numeric(out["Processati"], errors="coerce").round(0).astype("Int64")
    out["Ore lavoro (tot)"] = pd.to_numeric(out["Ore lavoro (tot)"], errors="coerce").round(2)
    out["% risposta"] = pd.to_numeric(out["% risposta"], errors="coerce").round(2)
    out["Resa oraria"] = pd.to_numeric(out["Resa oraria"], errors="coerce").round(6)
    out["Red%"] = pd.to_numeric(out["Red%"], errors="coerce").round(4)
    out["Processati per positivo"] = pd.to_numeric(out["Processati per positivo"], errors="coerce").round(4)
    out["Chiamate per positivo"] = pd.to_numeric(out["Chiamate per positivo"], errors="coerce").round(4)

    out["_is_team"] = out["Operatore"].eq("Team").astype(int)
    out = out.sort_values(["_is_team", "Operatore"], ascending=[True, True]).drop(columns=["_is_team"])
    return out


def mix_profiles_det(pq: pd.DataFrame, pf: pd.DataFrame, w_qual: float) -> pd.DataFrame:
    """
    Mix deterministico sui TOTALI (per operatore):
      totals_mix = w*pq + (1-w)*pf sui campi base.
      Poi ricalcolo % risposta, Red%, resa, ecc.
    """
    base_cols = [
        "Totale chiamate effettuate",
        "Totale chiamate con risposta",
        "Positivi",
        "Processati",
        "Ore lavoro (tot)",
    ]

    if pq is None or pq.empty:
        pq = pd.DataFrame(columns=["Operatore"] + base_cols)
    if pf is None or pf.empty:
        pf = pd.DataFrame(columns=["Operatore"] + base_cols)

    A = pq.set_index("Operatore").copy()
    B = pf.set_index("Operatore").copy()

    ops = sorted(set(A.index.astype(str)) | set(B.index.astype(str)))
    if "Team" in ops:
        ops = [o for o in ops if o != "Team"] + ["Team"]

    rows = []
    for op in ops:
        a = A.loc[op] if op in A.index else pd.Series(dtype=float)
        b = B.loc[op] if op in B.index else pd.Series(dtype=float)

        def _get(s, k):
            try:
                return float(pd.to_numeric(s.get(k, np.nan), errors="coerce"))
            except Exception:
                return np.nan

        calls = w_qual * (_get(a, base_cols[0])) + (1 - w_qual) * (_get(b, base_cols[0]))
        ans = w_qual * (_get(a, base_cols[1])) + (1 - w_qual) * (_get(b, base_cols[1]))
        pos = w_qual * (_get(a, base_cols[2])) + (1 - w_qual) * (_get(b, base_cols[2]))
        proc = w_qual * (_get(a, base_cols[3])) + (1 - w_qual) * (_get(b, base_cols[3]))
        ore = w_qual * (_get(a, base_cols[4])) + (1 - w_qual) * (_get(b, base_cols[4]))

        resp_pct = (ans / calls * 100.0) if np.isfinite(ans) and np.isfinite(calls) and calls else np.nan
        red_pct = (pos / proc * 100.0) if np.isfinite(pos) and np.isfinite(proc) and proc else np.nan
        resa_h = (pos / ore) if np.isfinite(pos) and np.isfinite(ore) and ore else np.nan
        ppp = (proc / pos) if np.isfinite(proc) and np.isfinite(pos) and pos else np.nan
        cpp = (calls / pos) if np.isfinite(calls) and np.isfinite(pos) and pos else np.nan

        rows.append(
            {
                "Operatore": op,
                "Totale chiamate effettuate": calls,
                "Totale chiamate con risposta": ans,
                "Positivi": pos,
                "Processati": proc,
                "Ore lavoro (tot)": ore,
                "% risposta": resp_pct,
                "Resa oraria": resa_h,
                "Red%": red_pct,
                "Processati per positivo": ppp,
                "Chiamate per positivo": cpp,
            }
        )

    out = pd.DataFrame(rows)

    # formatting coerente
    out["Totale chiamate effettuate"] = pd.to_numeric(out["Totale chiamate effettuate"], errors="coerce").round(0).astype("Int64")
    out["Totale chiamate con risposta"] = pd.to_numeric(out["Totale chiamate con risposta"], errors="coerce").round(0).astype("Int64")
    out["Positivi"] = pd.to_numeric(out["Positivi"], errors="coerce").round(2)
    out["Processati"] = pd.to_numeric(out["Processati"], errors="coerce").round(2)
    out["Ore lavoro (tot)"] = pd.to_numeric(out["Ore lavoro (tot)"], errors="coerce").round(2)
    out["% risposta"] = pd.to_numeric(out["% risposta"], errors="coerce").round(2)
    out["Resa oraria"] = pd.to_numeric(out["Resa oraria"], errors="coerce").round(6)
    out["Red%"] = pd.to_numeric(out["Red%"], errors="coerce").round(4)
    out["Processati per positivo"] = pd.to_numeric(out["Processati per positivo"], errors="coerce").round(4)
    out["Chiamate per positivo"] = pd.to_numeric(out["Chiamate per positivo"], errors="coerce").round(4)

    out["_is_team"] = out["Operatore"].astype(str).eq("Team").astype(int)
    out = out.sort_values(["_is_team", "Operatore"], ascending=[True, True]).drop(columns=["_is_team"])
    return out


# ===================== ML: daily dataset (per mix ML) =====================
def build_daily_dataset_for_mix_ml(df_use: pd.DataFrame) -> pd.DataFrame:
    """
    Costruisce dataset giornaliero per ML con feature:
      - calls, answers, ore_lavgen, w_qual (1=qual, 0=fredd)
      - operatore (one-hot)
    Target:
      - positivi, processati
    """
    if df_use is None or df_use.empty:
        return pd.DataFrame()

    date_col = pick_date_column(df_use)
    if not date_col:
        return pd.DataFrame()

    tmp = add_parsed_date(df_use, date_col).dropna(subset=["__date__", "Operatore"]).copy()
    if tmp.empty:
        return pd.DataFrame()

    col_map = {norm_colname(c): c for c in tmp.columns}

    def get_col(name: str) -> Optional[str]:
        return col_map.get(norm_colname(name))

    c_calls = get_col("Nr. chiamate effettuate")
    c_ans = get_col("Nr. chiamate con risposta")
    c_proc = get_col("Processati")
    c_pos = get_col("Positivi")
    c_pos_conf = get_col("Positivi confermati")
    c_ore = get_col("Lavorazione  generale")

    if c_calls is None or c_ans is None or c_proc is None or c_ore is None or (c_pos is None and c_pos_conf is None):
        return pd.DataFrame()

    tmp[c_calls] = coerce_numeric(tmp[c_calls])
    tmp[c_ans] = coerce_numeric(tmp[c_ans])
    tmp[c_proc] = coerce_numeric(tmp[c_proc])
    tmp[c_ore] = coerce_time_to_hours_auto(tmp[c_ore])

    if c_pos is None:
        tmp["__pos__"] = coerce_numeric(tmp[c_pos_conf])
    elif c_pos_conf is None:
        tmp["__pos__"] = coerce_numeric(tmp[c_pos])
    else:
        merged = tmp[c_pos].where(tmp[c_pos].notna(), tmp[c_pos_conf])
        tmp["__pos__"] = coerce_numeric(merged)

    # w_qual: 1 se file è Qualificate (non "fredd"), 0 se QualificateFreddo
    src_norm = tmp["__source_file"].astype(str).map(norm_filename)
    tmp["w_qual"] = np.where(src_norm.str.contains("fredd", na=False), 0.0, 1.0)

    g = tmp.groupby(["Operatore", "__date__", "w_qual"], dropna=False)
    daily = pd.DataFrame(
        {
            "calls": g[c_calls].sum(min_count=1),
            "answers": g[c_ans].sum(min_count=1),
            "processati": g[c_proc].sum(min_count=1),
            "positivi": g["__pos__"].sum(min_count=1),
            "ore": g[c_ore].sum(min_count=1),
        }
    ).reset_index()

    daily["resp_pct"] = (daily["answers"] / daily["calls"].replace(0, np.nan)) * 100.0
    daily = daily.replace([np.inf, -np.inf], np.nan).dropna(subset=["Operatore", "resp_pct"])
    return daily


def fit_mix_ml_models(daily: pd.DataFrame, model_kind: str) -> Dict[str, object]:
    """
    Allena 2 regressori (positivi, processati) con feature:
      resp_pct, calls, answers, ore, w_qual + one-hot Operatore
    model_kind: "ridge" | "hgb"
    """
    if daily is None or daily.empty:
        return {}

    needed = {"Operatore", "resp_pct", "calls", "answers", "ore", "w_qual", "positivi", "processati"}
    if not needed.issubset(set(daily.columns)):
        return {}

    X_num = daily[["resp_pct", "calls", "answers", "ore", "w_qual"]].copy()
    ops = pd.get_dummies(daily["Operatore"].astype(str), prefix="op", drop_first=False)
    X = pd.concat([X_num, ops], axis=1).fillna(0.0)
    y_pos = pd.to_numeric(daily["positivi"], errors="coerce").fillna(0.0).to_numpy(float)
    y_proc = pd.to_numeric(daily["processati"], errors="coerce").fillna(0.0).to_numpy(float)

    try:
        if model_kind == "hgb":
            from sklearn.ensemble import HistGradientBoostingRegressor

            m_pos = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.08, max_iter=300, random_state=7)
            m_proc = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.08, max_iter=300, random_state=7)
        else:
            from sklearn.linear_model import Ridge

            m_pos = Ridge(alpha=10.0, fit_intercept=True, random_state=7)
            m_proc = Ridge(alpha=10.0, fit_intercept=True, random_state=7)
    except Exception:
        return {}

    m_pos.fit(X.to_numpy(float), y_pos)
    m_proc.fit(X.to_numpy(float), y_proc)
    return {"X_cols": list(X.columns), "m_pos": m_pos, "m_proc": m_proc}


def predict_mix_ml(models: Dict[str, object], features_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    if not models:
        return np.full(len(features_df), np.nan), np.full(len(features_df), np.nan)
    cols = models["X_cols"]
    X = features_df.reindex(columns=cols, fill_value=0.0).to_numpy(float)
    pos = models["m_pos"].predict(X)
    proc = models["m_proc"].predict(X)
    return pos, proc


# ===================== LOAD + PRECOMPUTE =====================
DATA_ERR = None

df_all = pd.DataFrame()
used_files: List[str] = []

file_qual = None
file_fredd = None

try:
    df_raw, used_files = load_all_csv_cached(str(target_dir))
    if df_raw.empty:
        raise RuntimeError(f"Nessun CSV leggibile in: {target_dir}")

    df_all = df_raw.copy()
    unnamed_cols = [c for c in df_all.columns if str(c).lower().startswith("unnamed:")]
    if unnamed_cols:
        df_all = df_all.drop(columns=unnamed_cols)

    df_all["_op_norm"] = df_all["__OperatoreC"].map(normalize_text)
    df_all["Operatore"] = df_all["__OperatoreC"].astype(str).map(canonical_operator_name)

    files_norm = {f: norm_filename(f) for f in used_files}
    qual_files = [f for f, nf in files_norm.items() if "qualificate" in nf and "fredd" not in nf]
    fredd_files = [f for f, nf in files_norm.items() if "qualificate" in nf and "fredd" in nf]
    file_qual = sorted(qual_files)[0] if qual_files else None
    file_fredd = sorted(fredd_files)[0] if fredd_files else None

except Exception as e:
    DATA_ERR = str(e)


# ===================== DASH APP =====================
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server

HEADER = dbc.Card(
    dbc.CardBody(
        [
            html.H2("Simulazione Qualificate — Dashboard", style={"color": ACC_ORANGE, "fontWeight": 900, "marginBottom": "4px"}),
            html.Div("Correlazioni • clustering • ML • simulazioni mix", style={"color": MUTED, "marginBottom": "6px"}),
            html.Div(f"Cartella: {target_dir}", style={"color": MUTED, "fontFamily": "monospace", "fontSize": "12px"}),
            html.Div(
                f"CSV: {len(used_files)} — {', '.join(used_files) if used_files else '(nessuno)'}",
                style={"color": MUTED, "fontFamily": "monospace", "fontSize": "12px"},
            ),
            (dbc.Alert(DATA_ERR, color="danger") if DATA_ERR else html.Div()),
        ]
    ),
    style={"backgroundColor": BG, "border": f"1px solid {GRID}", "borderRadius": "18px"},
)

# ===================== TAB MEDIE =====================
TAB_MEDIE = dbc.Card(
    dbc.CardBody(
        [
            dbc.Alert(
                "Medie Operatore = MEDIA DEI TOTALI GIORNALIERI: somma per giorno → totale / giorni.",
                color="secondary",
                style={"backgroundColor": CARD, "border": f"1px solid {GRID}", "color": MUTED, "fontSize": "12px"},
            ),
            html.Div(id="medie-info"),
            html.Div(id="medie-operatori-block"),
            html.Hr(style={"borderColor": GRID}),
            html.H4("Correlazione tra operatori (feature su medie giornaliere) — colori", style={"color": TXT, "fontWeight": 800}),
            html.Div(id="medie-corr-operatori"),
            html.Div(style={"height": "12px"}),
            html.H4("Decorrelazione tra operatori (1 - correlazione) — colori", style={"color": TXT, "fontWeight": 800}),
            html.Div(id="medie-decorr-operatori"),
            html.Hr(style={"borderColor": GRID}),
            html.H4("Cluster operatori (medie giornaliere)", style={"color": TXT, "fontWeight": 800}),
            dcc.Slider(id="medie-k", min=2, max=8, step=1, value=3),
            html.Div(style={"height": "10px"}),
            html.Div(id="medie-cluster-table"),
            html.Div(style={"height": "10px"}),
            dcc.Graph(id="medie-cos-heat", config={"displayModeBar": False}),
            html.Div(style={"height": "10px"}),
            dcc.Graph(id="medie-dist-heat", config={"displayModeBar": False}),
            html.Div(style={"height": "10px"}),
            html.H4("Similarità/Differenze tra operatori (top divergenze)", style={"color": TXT, "fontWeight": 800}),
            html.Div(id="medie-pairwise-table"),
        ]
    ),
    style={"backgroundColor": BG, "border": f"1px solid {GRID}", "borderRadius": "18px"},
)

# ===================== TAB QUALIFICATE ML (ESTERNO) =====================
TAB_QUAL_ML = build_tab_qual_ml(
    app,
    shared={
        "DATA_ERR": lambda: DATA_ERR,
        "df_all": lambda: df_all,
        "file_qual": lambda: file_qual,
        "file_fredd": lambda: file_fredd,
        "BG": BG,
        "CARD": CARD,
        "GRID": GRID,
        "TXT": TXT,
        "MUTED": MUTED,
        "ACC_GREEN": ACC_GREEN,
        "ACC_ORANGE": ACC_ORANGE,
        "norm_colname": norm_colname,
        "METRIC_COLS": METRIC_COLS,
        "coerce_metric": coerce_metric,
        "coerce_numeric": coerce_numeric,
        "make_table": make_table,
        "matrix_to_colored_table": matrix_to_colored_table,
        "standardize_frame": standardize_frame,
        "cosine_similarity_matrix": cosine_similarity_matrix,
        "euclidean_distance_matrix": euclidean_distance_matrix,
        "pairwise_summary_entities": pairwise_summary_entities,
        "fig_heatmap": fig_heatmap,
        "base_layout": base_layout,
        # IMPORTANTE: non serve più passare fit_models_two_choices qui
    },
)

# ===================== TAB SIMULAZIONE MIX ML =====================
TAB_SIM_MIX_ML = dbc.Card(
    dbc.CardBody(
        [
            dbc.Alert(
                "Simulazione Mix: puoi usare il calcolo deterministico (mix diretto di tutte le grandezze) oppure 2 modelli ML addestrati sul giornaliero (Ridge / HistGradientBoosting).",
                color="secondary",
                style={"backgroundColor": CARD, "border": f"1px solid {GRID}", "color": MUTED, "fontSize": "12px"},
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div("Motore simulazione", style={"color": MUTED, "fontSize": "12px"}),
                            dcc.Dropdown(
                                id="mix-engine",
                                options=[
                                    {"label": "Deterministico (mix diretto)", "value": "det"},
                                    {"label": "ML — Ridge", "value": "ml_ridge"},
                                    {"label": "ML — HistGradientBoosting", "value": "ml_hgb"},
                                ],
                                value="det",
                                clearable=False,
                                style={"fontSize": "12px"},
                            ),
                        ],
                        width=5,
                    ),
                ],
                className="g-2",
            ),
            html.Div(style={"height": "12px"}),
            html.H4("Profiling — Qualificate (totali periodo)", style={"color": TXT, "fontWeight": 800}),
            html.Div(id="mix-prof-qual"),
            html.Div(style={"height": "14px"}),
            html.H4("Profiling — QualificateFreddo (totali periodo)", style={"color": TXT, "fontWeight": 800}),
            html.Div(id="mix-prof-fredd"),
            html.Div(style={"height": "14px"}),
            html.H4("Profiling — Totale (STANDARD attuale)", style={"color": TXT, "fontWeight": 800}),
            html.Div(id="mix-prof-tot"),
            html.Hr(style={"borderColor": GRID}),
            html.H3("Trend mix per operatore (90/10, 80/20, 70/30, 50/50) — confronto con STANDARD", style={"color": TXT, "fontWeight": 900}),
            html.Div(id="mix-mix-block"),
        ]
    ),
    style={"backgroundColor": BG, "border": f"1px solid {GRID}", "borderRadius": "18px"},
)

# ===================== TAB TEAM SIMULATION (NUOVO, CARD NEL FILE teamsimulation.py) =====================
TAB_TEAM_SIM = build_team_simulation_tab(
    app,
    shared={
        "DATA_ERR": lambda: DATA_ERR,
        "df_all": lambda: df_all,
        "file_qual": lambda: file_qual,
        "file_fredd": lambda: file_fredd,
        "BG": BG,
        "CARD": CARD,
        "GRID": GRID,
        "TXT": TXT,
        "MUTED": MUTED,
        "ACC_GREEN": ACC_GREEN,
        "ACC_ORANGE": ACC_ORANGE,
        "ACC_FUCHSIA": ACC_FUCHSIA,
        "norm_colname": norm_colname,
        "coerce_numeric": coerce_numeric,
        "coerce_time_to_hours_auto": coerce_time_to_hours_auto,
        "make_table": make_table,
        "base_layout": base_layout,
    },
)

# ===================== LAYOUT =====================
app.layout = dbc.Container(
    fluid=True,
    style={"backgroundColor": BG, "minHeight": "100vh", "padding": "14px"},
    children=[
        HEADER,
        html.Div(style={"height": "12px"}),
        dbc.Tabs(
            [
                dbc.Tab(TAB_MEDIE, label="Medie Operatore"),
                dbc.Tab(TAB_QUAL_ML, label="Qualificate ML"),
                dbc.Tab(TAB_SIM_MIX_ML, label="Simulazione Mix ML"),
                dbc.Tab(TAB_TEAM_SIM, label="Team Simulation"),  # <-- a destra
            ],
            style={"fontSize": "13px"},
        ),
        html.Div(style={"height": "10px"}),
    ],
)

# ===================== CALLBACKS: MEDIE =====================
@app.callback(
    Output("medie-info", "children"),
    Output("medie-operatori-block", "children"),
    Output("medie-corr-operatori", "children"),
    Output("medie-decorr-operatori", "children"),
    Input("medie-k", "value"),
)
def update_medie_block(_k: int):
    if DATA_ERR:
        a = dbc.Alert(DATA_ERR, color="danger")
        return a, a, a, a

    df_use_local = filter_to_target_ops(df_all).copy()
    if df_use_local.empty:
        msg = "Nessun dato per gli operatori target."
        a = dbc.Alert(msg, color="warning")
        return a, a, a, a

    date_col = pick_date_column(df_use_local)
    if not date_col:
        msg = "Non trovo una colonna DATA affidabile nel CSV."
        a = dbc.Alert(msg, color="danger")
        return a, a, a, a

    df_use_local = add_parsed_date(df_use_local, date_col).dropna(subset=["__date__"]).copy()
    if df_use_local.empty:
        msg = "Parsing data: nessuna riga con data valida."
        a = dbc.Alert(msg, color="danger")
        return dbc.Alert(f"Data: {date_col}", color="secondary"), a, a, a

    col_map = {norm_colname(c): c for c in df_use_local.columns}

    time_cols_present: List[str] = []
    for nm in TIME_ACTIVITY_COLS:
        c = col_map.get(norm_colname(nm))
        if c is not None:
            df_use_local[c] = coerce_time_to_hours_auto(df_use_local[c])
            time_cols_present.append(c)

    count_names = [
        "Nr. chiamate effettuate",
        "Nr. chiamate con risposta",
        "Processati",
        "Positivi",
        "Positivi confermati",
    ]
    count_cols_present: List[str] = []
    for nm in count_names:
        c = col_map.get(norm_colname(nm))
        if c is not None:
            df_use_local[c] = coerce_numeric(df_use_local[c])
            count_cols_present.append(c)

    op_mean, op_sum = build_daily_mean_profiles(df_use_local, col_map, time_cols_present, count_cols_present)
    if op_mean.empty:
        msg = "Non risultano giornate valide dopo l'aggregazione."
        a = dbc.Alert(msg, color="danger")
        return dbc.Alert(f"Data: {date_col}", color="secondary"), a, a, a

    team_mean = op_mean.select_dtypes(include=[np.number]).mean(numeric_only=True).to_frame().T
    team_mean.index = ["Team"]
    team_sum = op_sum.select_dtypes(include=[np.number]).sum(numeric_only=True).to_frame().T
    team_sum.index = ["Team"]
    op_mean2 = pd.concat([op_mean, team_mean], axis=0)
    op_sum2 = pd.concat([op_sum, team_sum], axis=0)

    base_show = time_cols_present + count_cols_present
    show_cols = base_show + [c for c in DERIVED_COLS_DAILY if c in op_mean2.columns]

    comp = pd.concat(
        [
            op_mean2[show_cols].round(6).add_suffix(" (mean/day)"),
            op_sum2[base_show].round(6).add_suffix(" (sum)"),
        ],
        axis=1,
    ).reset_index().rename(columns={"index": "Operatore"})

    info = html.Div(
        [
            html.Div(f"Data usata per il raggruppamento giornaliero: {date_col}", style={"color": MUTED, "fontFamily": "monospace", "fontSize": "12px"}),
            html.Div(
                "Tempo convertito in ORE (HH:MM oppure numerico con autodetect ore/minuti/secondi).",
                style={"color": MUTED, "fontSize": "12px"},
            ),
        ]
    )
    tbl = make_table(comp, page_size=18, max_width_px=240)

    feats = op_mean2.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
    feats = feats.loc[:, feats.nunique(dropna=True) > 1]
    if feats.shape[0] < 2 or feats.shape[1] < 2:
        msg = "Non ci sono abbastanza operatori/feature per correlazione."
        a = dbc.Alert(msg, color="warning")
        return info, tbl, a, a

    corr = feats.T.corr(method="pearson")
    corr_tbl = matrix_to_colored_table(corr, first_col_name="Operatore", mode="corr")

    decorr = (1.0 - corr).clip(lower=0.0, upper=2.0)
    decorr_tbl = matrix_to_colored_table(decorr, first_col_name="Operatore", mode="decorr")

    return info, tbl, corr_tbl, decorr_tbl


@app.callback(
    Output("medie-cluster-table", "children"),
    Output("medie-cos-heat", "figure"),
    Output("medie-dist-heat", "figure"),
    Output("medie-pairwise-table", "children"),
    Input("medie-k", "value"),
)
def update_medie_clustering(k: int):
    if DATA_ERR:
        fig = go.Figure()
        fig.add_annotation(text=DATA_ERR, x=0.5, y=0.5, showarrow=False, font=dict(color=TXT))
        a = dbc.Alert(DATA_ERR, color="danger")
        return a, base_layout(fig, "Coseno"), base_layout(fig, "Distanza"), a

    try:
        from sklearn.cluster import AgglomerativeClustering
    except Exception:
        msg = "Manca scikit-learn. Installa: pip install scikit-learn"
        fig = go.Figure()
        fig.add_annotation(text=msg, x=0.5, y=0.5, showarrow=False, font=dict(color=TXT))
        a = dbc.Alert(msg, color="danger")
        return a, base_layout(fig, "Coseno"), base_layout(fig, "Distanza"), a

    df_use_local = filter_to_target_ops(df_all).copy()
    date_col = pick_date_column(df_use_local)
    if not date_col:
        msg = "Non trovo una colonna DATA affidabile nel CSV."
        fig = go.Figure()
        fig.add_annotation(text=msg, x=0.5, y=0.5, showarrow=False, font=dict(color=TXT))
        a = dbc.Alert(msg, color="danger")
        return a, base_layout(fig, "Coseno"), base_layout(fig, "Distanza"), a

    df_use_local = add_parsed_date(df_use_local, date_col).dropna(subset=["__date__"]).copy()
    col_map = {norm_colname(c): c for c in df_use_local.columns}

    time_cols_present: List[str] = []
    for nm in TIME_ACTIVITY_COLS:
        c = col_map.get(norm_colname(nm))
        if c is not None:
            df_use_local[c] = coerce_time_to_hours_auto(df_use_local[c])
            time_cols_present.append(c)

    count_names = [
        "Nr. chiamate effettuate",
        "Nr. chiamate con risposta",
        "Processati",
        "Positivi",
        "Positivi confermati",
    ]
    count_cols_present: List[str] = []
    for nm in count_names:
        c = col_map.get(norm_colname(nm))
        if c is not None:
            df_use_local[c] = coerce_numeric(df_use_local[c])
            count_cols_present.append(c)

    op_mean, _op_sum = build_daily_mean_profiles(df_use_local, col_map, time_cols_present, count_cols_present)
    if op_mean.empty:
        msg = "Dati insufficienti per clustering (nessuna giornata valida)."
        fig = go.Figure()
        fig.add_annotation(text=msg, x=0.5, y=0.5, showarrow=False, font=dict(color=TXT))
        a = dbc.Alert(msg, color="warning")
        return a, base_layout(fig, "Coseno"), base_layout(fig, "Distanza"), a

    team_mean = op_mean.select_dtypes(include=[np.number]).mean(numeric_only=True).to_frame().T
    team_mean.index = ["Team"]
    op_mean2 = pd.concat([op_mean, team_mean], axis=0)

    feats = op_mean2.select_dtypes(include=[np.number]).copy()
    for c in feats.columns:
        feats[c] = feats[c].fillna(feats[c].median(skipna=True))
    feats = feats.loc[:, feats.nunique(dropna=True) > 1]

    if feats.shape[0] < 2 or feats.shape[1] < 1:
        msg = "Dati insufficienti per clustering."
        fig = go.Figure()
        fig.add_annotation(text=msg, x=0.5, y=0.5, showarrow=False, font=dict(color=TXT))
        a = dbc.Alert(msg, color="warning")
        return a, base_layout(fig, "Coseno"), base_layout(fig, "Distanza"), a

    Z = standardize_frame(feats).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    k = int(max(2, min(int(k or 2), feats.shape[0])))
    labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Z)

    clusters_df = pd.DataFrame({"Operatore": feats.index, "Cluster": labels}).sort_values(["Cluster", "Operatore"])
    cluster_table = make_table(clusters_df, page_size=18, max_width_px=280)

    cos = cosine_similarity_matrix(Z)
    dist = euclidean_distance_matrix(Z)

    fig_cos = fig_heatmap(cos, list(feats.index), f"Similarità coseno (k={k})", zmin=-1, zmax=1)
    zmax = float(np.nanpercentile(dist, 95)) if np.isfinite(dist).any() else None
    fig_dist = fig_heatmap(dist, list(feats.index), f"Distanza euclidea (k={k})", zmin=0, zmax=zmax)

    pair_df = pairwise_summary_entities(feats, "Operatore", top_k=5)
    pair_table = make_table(pair_df, page_size=14, max_width_px=360)

    return cluster_table, fig_cos, fig_dist, pair_table


# ===================== CALLBACKS: SIMULAZIONE MIX ML =====================
@app.callback(
    Output("mix-prof-qual", "children"),
    Output("mix-prof-fredd", "children"),
    Output("mix-prof-tot", "children"),
    Output("mix-mix-block", "children"),
    Input("mix-engine", "value"),
)
def update_mix_sim(engine_choice: str):
    if DATA_ERR:
        a = dbc.Alert(DATA_ERR, color="danger")
        return a, a, a, a
    if (not file_qual) or (not file_fredd):
        msg = "File Qualificate/QualificateFreddo non trovati."
        a = dbc.Alert(msg, color="danger")
        return a, a, a, a

    df_q = df_all[df_all["__source_file"].isin([file_qual, file_fredd])].copy()
    df_q["Operatore"] = df_q["Operatore"].astype(str).map(canonical_operator_name)
    col_map_q = {norm_colname(c): c for c in df_q.columns}

    df_qual = df_q[df_q["__source_file"] == file_qual].copy()
    df_frd = df_q[df_q["__source_file"] == file_fredd].copy()
    df_tot = df_q.copy()

    prof_qual_df = compute_profile_totals_table(df_qual, col_map_q, operator_col="Operatore")
    prof_frd_df = compute_profile_totals_table(df_frd, col_map_q, operator_col="Operatore")
    prof_tot_df = compute_profile_totals_table(df_tot, col_map_q, operator_col="Operatore")

    # tabelle base
    prof_qual_tbl = make_table(prof_qual_df, page_size=18, max_width_px=260) if not prof_qual_df.empty else dbc.Alert("Profiling non disponibile.", color="warning")
    prof_frd_tbl = make_table(prof_frd_df, page_size=18, max_width_px=260) if not prof_frd_df.empty else dbc.Alert("Profiling non disponibile.", color="warning")
    prof_tot_tbl = make_table(prof_tot_df, page_size=18, max_width_px=260) if not prof_tot_df.empty else dbc.Alert("Profiling non disponibile.", color="warning")

    std = prof_tot_df.set_index("Operatore").copy() if not prof_tot_df.empty else pd.DataFrame()

    ops_order = [op for op in TARGET_OPERATORS_CANON if (not std.empty and op in std.index)]
    if not std.empty:
        for op in std.index.astype(str).tolist():
            if op not in ops_order and op != "Team":
                ops_order.append(op)

    mix_specs = [
        ("Standard", None),
        ("90/10", 0.90),
        ("80/20", 0.80),
        ("70/30", 0.70),
        ("50/50", 0.50),
    ]

    # ====== DET engine: mix diretto dei TOTALI ======
    def _det_series_for_op(op: str):
        pos_s, proc_s, red_s = [], [], []
        for lab, w in mix_specs:
            if lab == "Standard":
                pos = float(pd.to_numeric(std.loc[op, "Positivi"], errors="coerce")) if (op in std.index) else np.nan
                proc = float(pd.to_numeric(std.loc[op, "Processati"], errors="coerce")) if (op in std.index) else np.nan
                red = float(pd.to_numeric(std.loc[op, "Red%"], errors="coerce")) if (op in std.index) else np.nan
                pos_s.append(pos); proc_s.append(proc); red_s.append(red)
            else:
                mix_df = mix_profiles_det(prof_qual_df, prof_frd_df, w_qual=float(w)).set_index("Operatore")
                if op not in mix_df.index:
                    pos_s.append(np.nan); proc_s.append(np.nan); red_s.append(np.nan)
                else:
                    pos_s.append(float(pd.to_numeric(mix_df.loc[op, "Positivi"], errors="coerce")))
                    proc_s.append(float(pd.to_numeric(mix_df.loc[op, "Processati"], errors="coerce")))
                    red_s.append(float(pd.to_numeric(mix_df.loc[op, "Red%"], errors="coerce")))
        return pos_s, proc_s, red_s

    # ====== ML engine: train su giornaliero, predici totali ~= pred(day)*n_days ======
    def _ml_series_all():
        daily = build_daily_dataset_for_mix_ml(df_tot)
        if daily.empty:
            return None, "Impossibile costruire dataset giornaliero per ML (colonne/data mancanti)."

        model_kind = "hgb" if engine_choice == "ml_hgb" else "ridge"
        models = fit_mix_ml_models(daily, model_kind=model_kind)
        if not models:
            return None, "Impossibile addestrare modelli ML (manca scikit-learn o dati insufficienti)."

        # medie giornaliere per qual/fredd (feature per scenario)
        dq = build_daily_dataset_for_mix_ml(df_qual)
        df = build_daily_dataset_for_mix_ml(df_frd)
        if dq.empty or df.empty:
            return None, "Per ML servono giornalieri sia Qualificate che QualificateFreddo."

        # n giorni (per scala su totale)
        date_col = pick_date_column(df_tot)
        n_days_by_op = {}
        if date_col:
            tmp = add_parsed_date(df_tot, date_col).dropna(subset=["__date__", "Operatore"])
            n_days_by_op = tmp.groupby("Operatore")["__date__"].nunique().to_dict()
        if "Team" not in n_days_by_op and len(n_days_by_op) > 0:
            n_days_by_op["Team"] = int(max(n_days_by_op.values()))

        meanq = dq.groupby("Operatore")[["resp_pct", "calls", "answers", "ore"]].mean(numeric_only=True)
        meanf = df.groupby("Operatore")[["resp_pct", "calls", "answers", "ore"]].mean(numeric_only=True)

        def features_for_mix(op: str, w: float) -> pd.DataFrame:
            def _row(m, opx):
                if opx in m.index:
                    return m.loc[opx]
                return pd.Series({"resp_pct": np.nan, "calls": np.nan, "answers": np.nan, "ore": np.nan})

            aq = _row(meanq, op)
            af = _row(meanf, op)
            row = {
                "resp_pct": w * float(aq.get("resp_pct", np.nan)) + (1 - w) * float(af.get("resp_pct", np.nan)),
                "calls": w * float(aq.get("calls", np.nan)) + (1 - w) * float(af.get("calls", np.nan)),
                "answers": w * float(aq.get("answers", np.nan)) + (1 - w) * float(af.get("answers", np.nan)),
                "ore": w * float(aq.get("ore", np.nan)) + (1 - w) * float(af.get("ore", np.nan)),
                "w_qual": float(w),
            }
            d = pd.DataFrame([row])
            op_dum = pd.get_dummies(pd.Series([op]).astype(str), prefix="op", drop_first=False)
            d = pd.concat([d, op_dum], axis=1)
            return d

        def team_feature_for_mix(w: float) -> pd.DataFrame:
            ops = [o for o in ops_order if o in meanq.index or o in meanf.index]
            if not ops:
                return pd.DataFrame()
            rows = []
            for op in ops:
                rows.append(features_for_mix(op, w))
            X = pd.concat(rows, ignore_index=True).fillna(0.0)
            return X

        results = {}
        mix_labels = [m[0] for m in mix_specs]

        for op in ops_order:
            pos_s, proc_s, red_s = [], [], []
            for lab, w in mix_specs:
                if lab == "Standard":
                    pos = float(pd.to_numeric(std.loc[op, "Positivi"], errors="coerce")) if (op in std.index) else np.nan
                    proc = float(pd.to_numeric(std.loc[op, "Processati"], errors="coerce")) if (op in std.index) else np.nan
                    red = float(pd.to_numeric(std.loc[op, "Red%"], errors="coerce")) if (op in std.index) else np.nan
                    pos_s.append(pos); proc_s.append(proc); red_s.append(red)
                else:
                    Xf = features_for_mix(op, float(w)).fillna(0.0)
                    p_day, pr_day = predict_mix_ml(models, Xf)
                    nd = float(n_days_by_op.get(op, np.nan))
                    if not np.isfinite(nd) or nd <= 0:
                        nd = float(n_days_by_op.get("Team", 1.0) or 1.0)
                    pos = float(p_day[0]) * nd
                    proc = float(pr_day[0]) * nd
                    red = (pos / proc * 100.0) if proc else np.nan
                    pos_s.append(pos); proc_s.append(proc); red_s.append(red)
            results[op] = (pos_s, proc_s, red_s)

        pos_sT, proc_sT, red_sT = [], [], []
        for lab, w in mix_specs:
            if lab == "Standard":
                pos = float(pd.to_numeric(std.loc["Team", "Positivi"], errors="coerce")) if ("Team" in std.index) else np.nan
                proc = float(pd.to_numeric(std.loc["Team", "Processati"], errors="coerce")) if ("Team" in std.index) else np.nan
                red = float(pd.to_numeric(std.loc["Team", "Red%"], errors="coerce")) if ("Team" in std.index) else np.nan
                pos_sT.append(pos); proc_sT.append(proc); red_sT.append(red)
            else:
                Xteam = team_feature_for_mix(float(w))
                if Xteam.empty:
                    pos_sT.append(np.nan); proc_sT.append(np.nan); red_sT.append(np.nan)
                else:
                    p_day, pr_day = predict_mix_ml(models, Xteam)
                    ndT = float(n_days_by_op.get("Team", 1.0) or 1.0)
                    pos = float(np.nansum(p_day)) * ndT
                    proc = float(np.nansum(pr_day)) * ndT
                    red = (pos / proc * 100.0) if proc else np.nan
                    pos_sT.append(pos); proc_sT.append(proc); red_sT.append(red)
        results["Team"] = (pos_sT, proc_sT, red_sT)

        return results, None

    accordion_items = []
    mix_labels = [m[0] for m in mix_specs]

    def _safe_float(v) -> Optional[float]:
        try:
            x = float(v)
            return x if np.isfinite(x) else None
        except Exception:
            return None

    if std.empty:
        return prof_qual_tbl, prof_frd_tbl, prof_tot_tbl, dbc.Alert("Profiling totale non disponibile.", color="danger")

    if engine_choice == "det":
        posT, procT, redT = _det_series_for_op("Team") if "Team" in std.index else ([np.nan]*len(mix_specs), [np.nan]*len(mix_specs), [np.nan]*len(mix_specs))
        fig_team = fig_operator_mix_trend(
            "Team — Trend mix (Deterministico)",
            mix_labels,
            posT,
            procT,
            redT,
            pos_std=_safe_float(std.loc["Team", "Positivi"]) if "Team" in std.index else None,
            proc_std=_safe_float(std.loc["Team", "Processati"]) if "Team" in std.index else None,
            red_std=_safe_float(std.loc["Team", "Red%"]) if "Team" in std.index else None,
        )
        accordion_items.append(dbc.AccordionItem(html.Div([dcc.Graph(figure=fig_team, config={"displayModeBar": False})]), title="Team (totale operatori)"))

        for op in ops_order:
            pos_s, proc_s, red_s = _det_series_for_op(op)
            fig_op = fig_operator_mix_trend(
                f"{op} — Trend mix (Deterministico)",
                mix_labels,
                pos_s,
                proc_s,
                red_s,
                pos_std=_safe_float(std.loc[op, "Positivi"]) if op in std.index else None,
                proc_std=_safe_float(std.loc[op, "Processati"]) if op in std.index else None,
                red_std=_safe_float(std.loc[op, "Red%"]) if op in std.index else None,
            )
            accordion_items.append(dbc.AccordionItem(html.Div([dcc.Graph(figure=fig_op, config={"displayModeBar": False})]), title=str(op)))

    else:
        results, err = _ml_series_all()
        if err:
            return prof_qual_tbl, prof_frd_tbl, prof_tot_tbl, dbc.Alert(err, color="danger")

        posT, procT, redT = results.get("Team", ([np.nan]*len(mix_specs), [np.nan]*len(mix_specs), [np.nan]*len(mix_specs)))
        fig_team = fig_operator_mix_trend(
            f"Team — Trend mix ({'ML HGB' if engine_choice=='ml_hgb' else 'ML Ridge'})",
            mix_labels,
            posT,
            procT,
            redT,
            pos_std=_safe_float(std.loc["Team", "Positivi"]) if "Team" in std.index else None,
            proc_std=_safe_float(std.loc["Team", "Processati"]) if "Team" in std.index else None,
            red_std=_safe_float(std.loc["Team", "Red%"]) if "Team" in std.index else None,
        )
        accordion_items.append(dbc.AccordionItem(html.Div([dcc.Graph(figure=fig_team, config={"displayModeBar": False})]), title="Team (totale operatori)"))

        for op in ops_order:
            pos_s, proc_s, red_s = results.get(op, ([np.nan]*len(mix_specs), [np.nan]*len(mix_specs), [np.nan]*len(mix_specs)))
            fig_op = fig_operator_mix_trend(
                f"{op} — Trend mix ({'ML HGB' if engine_choice=='ml_hgb' else 'ML Ridge'})",
                mix_labels,
                pos_s,
                proc_s,
                red_s,
                pos_std=_safe_float(std.loc[op, "Positivi"]) if op in std.index else None,
                proc_std=_safe_float(std.loc[op, "Processati"]) if op in std.index else None,
                red_std=_safe_float(std.loc[op, "Red%"]) if op in std.index else None,
            )
            accordion_items.append(dbc.AccordionItem(html.Div([dcc.Graph(figure=fig_op, config={"displayModeBar": False})]), title=str(op)))

    mix_block = dbc.Accordion(accordion_items, start_collapsed=True, always_open=False)
    return prof_qual_tbl, prof_frd_tbl, prof_tot_tbl, mix_block


# ===================== RUN =====================
if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)
