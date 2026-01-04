# qualificateML.py
from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

"""
QUALIFICATE ML (GIORNALIERO)

FIX definitivo per:
  TypeError: 'NoneType' object is not callable
causato da shadowing/sovrascrittura di fit_models_two_choices con None.

Regola:
- NON prendere funzioni ML da shared.
- NON usare mai come variabile locale il nome della funzione.
- La funzione ML interna ha un nome diverso: _fit_models_two_choices_internal
"""


# ===================== ML core: 2 modelli reali (Ridge lineare / Poly Ridge deg=2) =====================
def _fit_models_two_choices_internal(
    data: pd.DataFrame,
    ycol: str,
    model_choice: str,  # "ridge_linear" | "poly_ridge_deg2"
    min_points: int = 8,
    ridge_alpha: float = 10.0,
) -> Dict[str, Dict[str, float]]:
    """
    Fit per-operatore su relazione y ~ f(Answer%).
    Implementazione: Ridge su feature con dummy operatore + interazioni.
    Fallback senza sklearn: polyfit lineare per operatore / media.

    Output:
      { op: {"intercept":..., "slope":..., "quad":..., "n":..., "model":...}, ... }
    """
    if data is None or data.empty:
        return {}

    req = {"Operatore", "% risposta", ycol}
    if not req.issubset(set(data.columns)):
        return {}

    d = data[["Operatore", "% risposta", ycol]].copy()
    d["Operatore"] = d["Operatore"].astype(str)
    d["% risposta"] = pd.to_numeric(d["% risposta"], errors="coerce")
    d[ycol] = pd.to_numeric(d[ycol], errors="coerce")
    d = d.dropna(subset=["Operatore", "% risposta", ycol]).reset_index(drop=True)
    if d.empty:
        return {}

    # sklearn path
    try:
        from sklearn.linear_model import Ridge
    except Exception:
        # fallback: fit lineare per operatore con polyfit (o media se non possibile)
        out: Dict[str, Dict[str, float]] = {}
        for op, g in d.groupby("Operatore"):
            x = g["% risposta"].to_numpy(float)
            y = g[ycol].to_numpy(float)
            n = float(len(g))
            if len(g) >= 2 and np.nanstd(x) > 0:
                a, b = np.polyfit(x, y, 1)  # y = a*x + b
                out[op] = {
                    "intercept": float(b),
                    "slope": float(a),
                    "quad": 0.0,
                    "n": n,
                    "model": "fallback-lineare(np.polyfit)",
                }
            else:
                out[op] = {
                    "intercept": float(np.nanmean(y)),
                    "slope": 0.0,
                    "quad": 0.0,
                    "n": n,
                    "model": "fallback-media",
                }
        return out

    x = d["% risposta"].to_numpy(float)
    x2 = x**2

    dummies = pd.get_dummies(d["Operatore"], prefix="op", drop_first=False).reset_index(drop=True)

    if model_choice == "poly_ridge_deg2":
        X = pd.DataFrame({"x": x, "x2": x2})
        X = pd.concat([X, dummies], axis=1)
        for c in dummies.columns:
            X[f"{c}:x"] = dummies[c].to_numpy(float) * x
            X[f"{c}:x2"] = dummies[c].to_numpy(float) * x2
        model_name = "Poly Ridge (deg=2)"
    else:
        X = pd.DataFrame({"x": x})
        X = pd.concat([X, dummies], axis=1)
        for c in dummies.columns:
            X[f"{c}:x"] = dummies[c].to_numpy(float) * x
        model_name = "Ridge lineare"

    y = d[ycol].to_numpy(float)

    model = Ridge(alpha=float(ridge_alpha), fit_intercept=True)
    model.fit(X.to_numpy(float), y)

    coef = pd.Series(model.coef_, index=X.columns)
    intercept_global = float(model.intercept_)

    base_slope = float(coef.get("x", 0.0))
    base_quad = float(coef.get("x2", 0.0)) if model_choice == "poly_ridge_deg2" else 0.0

    out: Dict[str, Dict[str, float]] = {}
    for op in d["Operatore"].unique():
        op_col = f"op_{op}"
        int_op = intercept_global + float(coef.get(op_col, 0.0))
        slope_op = base_slope + float(coef.get(f"{op_col}:x", 0.0))
        quad_op = base_quad + (float(coef.get(f"{op_col}:x2", 0.0)) if model_choice == "poly_ridge_deg2" else 0.0)
        n = float((d["Operatore"] == op).sum())
        out[op] = {
            "intercept": float(int_op),
            "slope": float(slope_op),
            "quad": float(quad_op),
            "n": n,
            "model": model_name,
        }
    return out


# ===================== TAB BUILDER =====================
def build_tab_qual_ml(app, shared) -> dbc.Card:
    # ===================== shared getters =====================
    get_DATA_ERR = shared["DATA_ERR"]
    get_df_all = shared["df_all"]
    get_file_qual = shared["file_qual"]
    get_file_fredd = shared["file_fredd"]

    # ===================== theme =====================
    BG = shared["BG"]
    CARD = shared["CARD"]
    GRID = shared["GRID"]
    TXT = shared["TXT"]
    MUTED = shared["MUTED"]
    ACC_GREEN = shared["ACC_GREEN"]
    ACC_ORANGE = shared["ACC_ORANGE"]

    # ===================== shared helpers (required) =====================
    norm_colname = shared["norm_colname"]
    coerce_numeric = shared["coerce_numeric"]
    make_table = shared["make_table"]
    matrix_to_colored_table = shared["matrix_to_colored_table"]

    # clustering/plots
    standardize_frame = shared["standardize_frame"]
    cosine_similarity_matrix = shared["cosine_similarity_matrix"]
    euclidean_distance_matrix = shared["euclidean_distance_matrix"]
    pairwise_summary_entities = shared["pairwise_summary_entities"]
    fig_heatmap = shared["fig_heatmap"]
    base_layout = shared["base_layout"]

    # ===================== ML helpers =====================
    def _predict(params: Dict[str, float], x: float) -> float:
        b0 = float(params.get("intercept", np.nan))
        b1 = float(params.get("slope", 0.0))
        b2 = float(params.get("quad", 0.0))
        if not np.isfinite(x) or not np.isfinite(b0) or not np.isfinite(b1) or not np.isfinite(b2):
            return np.nan
        return float(b0 + b1 * x + b2 * (x**2))

    def _delta_per_1pt_answer(params: Dict[str, float], x: float) -> float:
        b1 = float(params.get("slope", np.nan))
        b2 = float(params.get("quad", 0.0))
        if not np.isfinite(x) or not np.isfinite(b1) or not np.isfinite(b2):
            return np.nan
        return float(b1 + 2.0 * b2 * x)

    def _prob_reach_answer(x_hist: np.ndarray, x_target: float, min_points: int = 8) -> float:
        x = np.array(x_hist, dtype=float)
        x = x[np.isfinite(x)]
        if len(x) < min_points or not np.isfinite(x_target):
            return np.nan
        return float((x >= x_target).mean() * 100.0)

    # ===================== DATE detection =====================
    def _find_best_date_col(df: pd.DataFrame) -> Optional[str]:
        best = None
        best_score = -1.0
        n = max(len(df), 1)

        for c in df.columns:
            s = df[c]
            if s.isna().all():
                continue
            if pd.api.types.is_numeric_dtype(s):
                continue

            dt = pd.to_datetime(s, errors="coerce", dayfirst=True, infer_datetime_format=True)
            ok_ratio = float(dt.notna().mean())
            if ok_ratio < 0.55:
                continue

            uniq_days = int(dt.dropna().dt.date.nunique())
            if uniq_days < 2:
                continue

            score = ok_ratio * 0.85 + min(uniq_days / n, 1.0) * 0.15
            if score > best_score:
                best_score = score
                best = c

        return best

    # ===================== UI =====================
    tab = dbc.Card(
        dbc.CardBody(
            [
                dbc.Alert(
                    "Qualificate ML (GIORNALIERO): Positivi = positivi/giorno (somma per data). "
                    "In fondo trovi la tabella scenari Answer% (+1/+2/+5) con frequenza storica di raggiungimento.",
                    color="secondary",
                    style={"backgroundColor": CARD, "border": f"1px solid {GRID}", "color": MUTED, "fontSize": "12px"},
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Div("Dataset", style={"color": MUTED, "fontSize": "12px"}),
                                dcc.Dropdown(
                                    id="qml-dataset",
                                    options=[
                                        {"label": "Combinato", "value": "Combinato"},
                                        {"label": "Qualificate", "value": "Qualificate"},
                                        {"label": "QualificateFreddo", "value": "QualificateFreddo"},
                                    ],
                                    value="Combinato",
                                    clearable=False,
                                    style={"fontSize": "12px"},
                                ),
                            ],
                            width=4,
                        ),
                        dbc.Col(
                            [
                                html.Div("Modello ML", style={"color": MUTED, "fontSize": "12px"}),
                                dcc.Dropdown(
                                    id="qml-model",
                                    options=[
                                        {"label": "Ridge lineare", "value": "ridge_linear"},
                                        {"label": "Poly Ridge (deg=2)", "value": "poly_ridge_deg2"},
                                    ],
                                    value="poly_ridge_deg2",
                                    clearable=False,
                                    style={"fontSize": "12px"},
                                ),
                            ],
                            width=4,
                        ),
                        dbc.Col(
                            [
                                html.Div("Operatori", style={"color": MUTED, "fontSize": "12px"}),
                                dcc.Dropdown(id="qml-ops", options=[], value=[], multi=True, style={"fontSize": "12px"}),
                            ],
                            width=4,
                        ),
                    ],
                    className="g-2",
                ),
                html.Div(style={"height": "12px"}),
                dbc.Tabs(
                    [
                        dbc.Tab(label="ML", tab_id="qml-tab-ml"),
                        dbc.Tab(label="Corr Feature", tab_id="qml-tab-corrfeat"),
                        dbc.Tab(label="Corr Operatori", tab_id="qml-tab-corrops"),
                        dbc.Tab(label="Cluster Operatori", tab_id="qml-tab-cluster"),
                    ],
                    id="qml-tabs",
                    active_tab="qml-tab-ml",
                ),
                html.Div(style={"height": "12px"}),
                html.Div(id="qml-content"),
            ]
        ),
        style={"backgroundColor": BG, "border": f"1px solid {GRID}", "borderRadius": "18px"},
    )

    # ===================== data selection =====================
    def _df_q() -> pd.DataFrame:
        df_all = get_df_all()
        fq = get_file_qual()
        ff = get_file_fredd()
        if df_all is None or df_all.empty or (not fq) or (not ff):
            return pd.DataFrame()
        if "__source_file" not in df_all.columns:
            return pd.DataFrame()
        return df_all[df_all["__source_file"].isin([fq, ff])].copy()

    def _select(df_q: pd.DataFrame, dataset_choice: str) -> pd.DataFrame:
        fq = get_file_qual()
        ff = get_file_fredd()
        if df_q.empty:
            return df_q
        if dataset_choice == "Qualificate":
            return df_q[df_q["__source_file"] == fq].copy()
        if dataset_choice == "QualificateFreddo":
            return df_q[df_q["__source_file"] == ff].copy()
        return df_q.copy()

    # ===================== DAILY dataset builder =====================
    def _build_daily_dataset(df_use: pd.DataFrame) -> pd.DataFrame:
        if df_use.empty:
            return pd.DataFrame()

        date_col = _find_best_date_col(df_use)
        if not date_col:
            return pd.DataFrame()

        col_map = {norm_colname(c): c for c in df_use.columns}

        def get_col(name: str) -> Optional[str]:
            return col_map.get(norm_colname(name))

        c_calls = get_col("Nr. chiamate effettuate")
        c_ans = get_col("Nr. chiamate con risposta")
        c_proc = get_col("Processati")
        c_pos = get_col("Positivi")
        c_pos_conf = get_col("Positivi confermati")

        if c_calls is None or c_ans is None or c_proc is None or (c_pos is None and c_pos_conf is None):
            return pd.DataFrame()
        if "Operatore" not in df_use.columns:
            return pd.DataFrame()

        tmp = df_use.copy()
        tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce", dayfirst=True, infer_datetime_format=True)
        tmp = tmp.dropna(subset=[date_col, "Operatore"])
        if tmp.empty:
            return pd.DataFrame()

        tmp["Operatore"] = tmp["Operatore"].astype(str)
        tmp["__date__"] = tmp[date_col].dt.date.astype(str)

        tmp[c_calls] = coerce_numeric(tmp[c_calls])
        tmp[c_ans] = coerce_numeric(tmp[c_ans])
        tmp[c_proc] = coerce_numeric(tmp[c_proc])

        if c_pos is None:
            tmp["__pos__"] = coerce_numeric(tmp[c_pos_conf])
        elif c_pos_conf is None:
            tmp["__pos__"] = coerce_numeric(tmp[c_pos])
        else:
            merged = tmp[c_pos].where(tmp[c_pos].notna(), tmp[c_pos_conf])
            tmp["__pos__"] = coerce_numeric(merged)

        g = tmp.groupby(["Operatore", "__date__"], dropna=False)
        daily = pd.DataFrame(
            {
                "Chiamate": g[c_calls].sum(min_count=1),
                "Risposte": g[c_ans].sum(min_count=1),
                "Processati": g[c_proc].sum(min_count=1),
                "Positivi": g["__pos__"].sum(min_count=1),
            }
        ).reset_index()

        # Team per-giorno
        gt = (
            daily.groupby("__date__", dropna=False)[["Chiamate", "Risposte", "Processati", "Positivi"]]
            .sum(min_count=1)
            .reset_index()
        )
        gt.insert(0, "Operatore", "Team")
        daily = pd.concat([daily, gt], ignore_index=True)

        daily["% risposta"] = (daily["Risposte"] / daily["Chiamate"].replace(0, np.nan)) * 100.0
        daily["Red%"] = (daily["Positivi"] / daily["Processati"].replace(0, np.nan)) * 100.0
        daily["Processati per positivo"] = daily["Processati"] / daily["Positivi"].replace(0, np.nan)

        daily = daily.replace([np.inf, -np.inf], np.nan)
        daily = daily.dropna(subset=["Operatore", "% risposta"])
        return daily

    def _operator_mean_for_corr(df_use: pd.DataFrame) -> pd.DataFrame:
        if df_use.empty or "Operatore" not in df_use.columns:
            return pd.DataFrame()
        numeric = df_use.select_dtypes(include=[np.number]).copy()
        if numeric.empty:
            return pd.DataFrame()
        mean_df = df_use.groupby("Operatore", dropna=False)[numeric.columns].mean(numeric_only=True)
        team = mean_df.mean(numeric_only=True).to_frame().T
        team.index = ["Team"]
        return pd.concat([mean_df, team], axis=0)

    # ===================== CALLBACKS =====================
    @app.callback(
        Output("qml-ops", "options"),
        Output("qml-ops", "value"),
        Input("qml-dataset", "value"),
    )
    def qml_ops_options(dataset_choice: str):
        if get_DATA_ERR():
            return [], []
        df_q = _df_q()
        if df_q.empty or "Operatore" not in df_q.columns:
            return [], []
        df_use = _select(df_q, dataset_choice)
        if df_use.empty or "Operatore" not in df_use.columns:
            return [], []
        ops = sorted(df_use["Operatore"].dropna().astype(str).unique().tolist())
        ops2 = ops if "Team" in ops else (ops + ["Team"])
        return [{"label": o, "value": o} for o in ops2], ops

    @app.callback(
        Output("qml-content", "children"),
        Input("qml-tabs", "active_tab"),
        Input("qml-dataset", "value"),
        Input("qml-model", "value"),
        Input("qml-ops", "value"),
    )
    def qml_render(active_tab: str, dataset_choice: str, model_choice: str, chosen_ops: List[str]):
        err = get_DATA_ERR()
        if err:
            return dbc.Alert(err, color="danger")

        df_q = _df_q()
        if df_q.empty:
            return dbc.Alert("File Qualificate/QualificateFreddo non trovati o dataset vuoto.", color="danger")

        df_use = _select(df_q, dataset_choice)

        if active_tab == "qml-tab-corrfeat":
            daily = _build_daily_dataset(df_use)
            if daily.empty:
                return dbc.Alert(
                    "Impossibile costruire dataset giornaliero. Serve una colonna Data valida e le colonne: "
                    "Nr. chiamate effettuate / Nr. chiamate con risposta / Processati / Positivi.",
                    color="danger",
                )
            feat = daily[["% risposta", "Positivi", "Processati", "Processati per positivo", "Red%"]].copy()
            corr = feat.corr(method="pearson")
            return html.Div(
                [
                    html.H4("Matrice correlazione feature (GIORNALIERO) — colori", style={"color": TXT, "fontWeight": 800}),
                    matrix_to_colored_table(corr, first_col_name="Feature", mode="corr"),
                ]
            )

        if active_tab == "qml-tab-corrops":
            mean_df = _operator_mean_for_corr(df_use)
            if mean_df.empty:
                return dbc.Alert("Non abbastanza dati numerici per correlazione operatori.", color="warning")

            feats = mean_df.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
            feats = feats.loc[:, feats.nunique(dropna=True) > 1]
            if feats.shape[0] < 2 or feats.shape[1] < 2:
                return dbc.Alert("Non abbastanza operatori/feature per correlazione.", color="warning")

            corr = feats.T.corr(method="pearson")
            decorr = (1.0 - corr).clip(lower=0.0, upper=2.0)

            return html.Div(
                [
                    html.H3("Correlazione tra operatori — colori", style={"color": TXT, "fontWeight": 900}),
                    matrix_to_colored_table(corr, first_col_name="Operatore", mode="corr"),
                    html.Div(style={"height": "14px"}),
                    html.H3("Decorrelazione (1 - corr) — colori", style={"color": TXT, "fontWeight": 900}),
                    matrix_to_colored_table(decorr, first_col_name="Operatore", mode="decorr"),
                ]
            )

        if active_tab == "qml-tab-cluster":
            try:
                from sklearn.cluster import AgglomerativeClustering
            except Exception:
                return dbc.Alert("Manca scikit-learn. Installa: pip install scikit-learn", color="danger")

            mean_df = _operator_mean_for_corr(df_use)
            if mean_df.empty:
                return dbc.Alert("Dati insufficienti per clustering.", color="warning")

            feats = mean_df.select_dtypes(include=[np.number]).copy()
            for c in feats.columns:
                feats[c] = feats[c].fillna(feats[c].median(skipna=True))
            feats = feats.loc[:, feats.nunique(dropna=True) > 1]
            if feats.shape[0] < 2 or feats.shape[1] < 1:
                return dbc.Alert("Dati insufficienti per clustering.", color="warning")

            Z = standardize_frame(feats).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
            k = int(max(2, min(4, feats.shape[0])))
            labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Z)

            clusters_df = pd.DataFrame({"Operatore": feats.index, "Cluster": labels}).sort_values(["Cluster", "Operatore"])
            cos = cosine_similarity_matrix(Z)
            dist = euclidean_distance_matrix(Z)
            pair_df = pairwise_summary_entities(feats, "Operatore", top_k=5)

            return html.Div(
                [
                    html.H4(f"Cluster operatori (dataset={dataset_choice})", style={"color": TXT, "fontWeight": 900}),
                    make_table(clusters_df, page_size=18, max_width_px=280),
                    html.Div(style={"height": "10px"}),
                    dcc.Graph(figure=fig_heatmap(cos, list(feats.index), "Similarità coseno", zmin=-1, zmax=1), config={"displayModeBar": False}),
                    html.Div(style={"height": "10px"}),
                    dcc.Graph(
                        figure=fig_heatmap(dist, list(feats.index), "Distanza euclidea", zmin=0, zmax=float(np.nanpercentile(dist, 95))),
                        config={"displayModeBar": False},
                    ),
                    html.Div(style={"height": "10px"}),
                    html.H4("Similarità/Differenze (top divergenze)", style={"color": TXT, "fontWeight": 800}),
                    make_table(pair_df, page_size=14, max_width_px=360),
                ]
            )

        # ---------- ML (GIORNALIERO) ----------
        chosen_ops = [str(x) for x in (chosen_ops or [])]
        if not chosen_ops:
            return dbc.Alert("Seleziona almeno un operatore.", color="warning")

        daily = _build_daily_dataset(df_use)
        if daily.empty:
            return dbc.Alert(
                "Impossibile costruire dataset GIORNALIERO. "
                "Serve una colonna Data valida + colonne: Nr. chiamate effettuate / Nr. chiamate con risposta / Processati / Positivi.",
                color="danger",
            )

        targets = ["Positivi", "Processati per positivo", "Red%"]

        models_by_target: Dict[str, Dict[str, Dict[str, float]]] = {}
        for tgt in targets:
            models_by_target[tgt] = _fit_models_two_choices_internal(
                data=daily,
                ycol=tgt,
                model_choice=("poly_ridge_deg2" if model_choice == "poly_ridge_deg2" else "ridge_linear"),
                min_points=8,
                ridge_alpha=10.0,
            )

        x_current_map = daily.groupby("Operatore")["% risposta"].mean().apply(lambda v: float(v) if pd.notna(v) else np.nan).to_dict()
        x_hist_map: Dict[str, np.ndarray] = {str(op): g["% risposta"].to_numpy(float) for op, g in daily.groupby("Operatore")}

        riepilogo_rows = []
        for op in chosen_ops:
            xcur = x_current_map.get(op, np.nan)

            mp_pos = models_by_target.get("Positivi", {}).get(op, {})
            mp_ppp = models_by_target.get("Processati per positivo", {}).get(op, {})
            mp_red = models_by_target.get("Red%", {}).get(op, {})

            pos_at = _predict(mp_pos, xcur) if mp_pos else np.nan
            ppp_at = _predict(mp_ppp, xcur) if mp_ppp else np.nan
            red_at = _predict(mp_red, xcur) if mp_red else np.nan

            dpos = _delta_per_1pt_answer(mp_pos, xcur) if mp_pos else np.nan
            dppp = _delta_per_1pt_answer(mp_ppp, xcur) if mp_ppp else np.nan
            dred = _delta_per_1pt_answer(mp_red, xcur) if mp_red else np.nan

            n_vals = []
            for mp in (mp_pos, mp_ppp, mp_red):
                if mp and np.isfinite(float(mp.get("n", np.nan))):
                    n_vals.append(float(mp.get("n", 0)))
            n_eff = int(max(n_vals) if n_vals else 0)

            mdl = (mp_pos.get("model") if mp_pos else None) or (mp_ppp.get("model") if mp_ppp else None) or (mp_red.get("model") if mp_red else None) or "?"

            riepilogo_rows.append(
                {
                    "Operatore": op,
                    "Answer% attuale (media gg)": xcur,
                    "Positivi/giorno (stima @ Answer%)": pos_at,
                    "Δ Positivi/giorno per +1pt Answer%": dpos,
                    "Processati/Positivo (stima @ Answer%)": ppp_at,
                    "Δ Proc/Pos per +1pt Answer%": dppp,
                    "Red% (stima @ Answer%)": red_at,
                    "Δ Red% per +1pt Answer%": dred,
                    "ML": mdl,
                    "n (giorni)": n_eff,
                }
            )

        riepilogo_df = pd.DataFrame(riepilogo_rows)
        if not riepilogo_df.empty:
            for c in [
                "Answer% attuale (media gg)",
                "Positivi/giorno (stima @ Answer%)",
                "Δ Positivi/giorno per +1pt Answer%",
                "Processati/Positivo (stima @ Answer%)",
                "Δ Proc/Pos per +1pt Answer%",
                "Red% (stima @ Answer%)",
                "Δ Red% per +1pt Answer%",
            ]:
                riepilogo_df[c] = pd.to_numeric(riepilogo_df[c], errors="coerce")

            riepilogo_df["Answer% attuale (media gg)"] = riepilogo_df["Answer% attuale (media gg)"].round(2)
            riepilogo_df["Positivi/giorno (stima @ Answer%)"] = riepilogo_df["Positivi/giorno (stima @ Answer%)"].round(4)
            riepilogo_df["Δ Positivi/giorno per +1pt Answer%"] = riepilogo_df["Δ Positivi/giorno per +1pt Answer%"].round(4)
            riepilogo_df["Processati/Positivo (stima @ Answer%)"] = riepilogo_df["Processati/Positivo (stima @ Answer%)"].round(4)
            riepilogo_df["Δ Proc/Pos per +1pt Answer%"] = riepilogo_df["Δ Proc/Pos per +1pt Answer%"].round(6)
            riepilogo_df["Red% (stima @ Answer%)"] = riepilogo_df["Red% (stima @ Answer%)"].round(4)
            riepilogo_df["Δ Red% per +1pt Answer%"] = riepilogo_df["Δ Red% per +1pt Answer%"].round(6)

        riepilogo_tbl = make_table(riepilogo_df, page_size=18, max_width_px=380) if not riepilogo_df.empty else dbc.Alert("Nessun modello disponibile.", color="warning")

        deltas = [1.0, 2.0, 5.0]
        scenario_rows = []
        for op in chosen_ops:
            xcur = x_current_map.get(op, np.nan)
            xhist = x_hist_map.get(op, np.array([], dtype=float))

            mp_pos = models_by_target.get("Positivi", {}).get(op, {})
            mp_ppp = models_by_target.get("Processati per positivo", {}).get(op, {})
            mp_red = models_by_target.get("Red%", {}).get(op, {})

            for dlt in deltas:
                xt = (float(xcur) + float(dlt)) if np.isfinite(xcur) else np.nan
                scenario_rows.append(
                    {
                        "Operatore": op,
                        "Scenario Answer%": f"+{int(dlt)}pt",
                        "Answer% target": xt,
                        "Prob. storica di raggiungerlo (%)": _prob_reach_answer(xhist, xt, min_points=8),
                        "Positivi/giorno (stima @ target)": _predict(mp_pos, xt) if mp_pos else np.nan,
                        "Processati/Positivo (stima @ target)": _predict(mp_ppp, xt) if mp_ppp else np.nan,
                        "Red% (stima @ target)": _predict(mp_red, xt) if mp_red else np.nan,
                    }
                )

        scenario_df = pd.DataFrame(scenario_rows)
        if not scenario_df.empty:
            scenario_df["Answer% target"] = pd.to_numeric(scenario_df["Answer% target"], errors="coerce").round(2)
            scenario_df["Prob. storica di raggiungerlo (%)"] = pd.to_numeric(scenario_df["Prob. storica di raggiungerlo (%)"], errors="coerce").round(1)
            scenario_df["Positivi/giorno (stima @ target)"] = pd.to_numeric(scenario_df["Positivi/giorno (stima @ target)"], errors="coerce").round(4)
            scenario_df["Processati/Positivo (stima @ target)"] = pd.to_numeric(scenario_df["Processati/Positivo (stima @ target)"], errors="coerce").round(4)
            scenario_df["Red% (stima @ target)"] = pd.to_numeric(scenario_df["Red% (stima @ target)"], errors="coerce").round(4)

        scenario_tbl = make_table(scenario_df, page_size=18, max_width_px=420) if not scenario_df.empty else dbc.Alert("Scenari non disponibili.", color="warning")

        acc_items = []
        for op in chosen_ops:
            xcur = x_current_map.get(op, np.nan)
            df_op = daily[daily["Operatore"].astype(str) == op].copy()

            graphs = []
            for tgt in targets:
                mp = models_by_target.get(tgt, {}).get(op, {})
                fig = go.Figure()
                title = f"{op} — {tgt} (GIORNALIERO) vs Answer% | ML={mp.get('model','?')} (n={int(mp.get('n',0))})"

                dd = df_op.dropna(subset=["% risposta", tgt]).copy()
                if dd.empty or len(dd) < 2 or not mp:
                    fig.add_annotation(text="Dati insufficienti", x=0.5, y=0.5, showarrow=False, font=dict(color=TXT))
                    graphs.append(dcc.Graph(figure=base_layout(fig, title, height=320), config={"displayModeBar": False}))
                    continue

                x = dd["% risposta"].to_numpy(float)
                y = dd[tgt].to_numpy(float)
                fig.add_trace(go.Scatter(x=x, y=y, mode="markers", marker=dict(size=6, color=ACC_GREEN), name="Dati (giorni)"))

                b0 = float(mp.get("intercept", np.nan))
                b1 = float(mp.get("slope", np.nan))
                b2 = float(mp.get("quad", 0.0))

                xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
                pad = max(1.0, 0.05 * (xmax - xmin))
                xg = np.linspace(max(0.0, xmin - pad), min(100.0, xmax + pad), 240)
                yg = b0 + b1 * xg + b2 * (xg**2)
                fig.add_trace(go.Scatter(x=xg, y=yg, mode="lines", line=dict(width=2, color=ACC_ORANGE), name="Modello"))

                if pd.notna(xcur):
                    ycur = b0 + b1 * float(xcur) + b2 * (float(xcur) ** 2)
                    fig.add_trace(go.Scatter(x=[float(xcur)], y=[float(ycur)], mode="markers", marker=dict(size=9, color=ACC_ORANGE), name="Pred (Answer% attuale)"))

                graphs.append(dcc.Graph(figure=base_layout(fig, title, height=320), config={"displayModeBar": False}))

            body = html.Div(
                [
                    html.Div(
                        f"Answer% attuale (media gg): {'' if pd.isna(xcur) else f'{xcur:.2f}%'}",
                        style={"color": MUTED, "fontFamily": "monospace", "fontSize": "12px"},
                    ),
                    html.Div(style={"height": "10px"}),
                    *graphs,
                ]
            )
            acc_items.append(dbc.AccordionItem(body, title=str(op)))

        accordion = dbc.Accordion(acc_items, start_collapsed=True, always_open=False) if acc_items else dbc.Alert("Nessun operatore.", color="warning")

        return html.Div(
            [
                html.H4("Riepilogo (GIORNALIERO) per operatore", style={"color": TXT, "fontWeight": 900}),
                riepilogo_tbl,
                html.Div(style={"height": "16px"}),
                html.H4("Scenari Answer% (+1/+2/+5) — stime giornaliere e probabilità (frequenza storica)", style={"color": TXT, "fontWeight": 900}),
                html.Div(
                    "Nota: la “probabilità” è la percentuale di GIORNI storici nel dataset selezionato con Answer% ≥ target (frequenza empirica).",
                    style={"color": MUTED, "fontSize": "12px", "marginBottom": "8px"},
                ),
                scenario_tbl,
                html.Div(style={"height": "16px"}),
                html.H4("Grafici giornalieri per operatore", style={"color": TXT, "fontWeight": 900}),
                accordion,
            ]
        )

    return tab
