# teamsimulation.py
from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

from dash import html, dcc, Input, Output, State, dash_table, ctx
import dash_bootstrap_components as dbc


def build_team_simulation_tab(app, shared) -> dbc.Card:
    # ===================== SHARED DATA =====================
    get_DATA_ERR = shared.get("DATA_ERR", lambda: None)
    get_df_all = shared.get("df_all", lambda: pd.DataFrame())
    get_file_qual = shared.get("file_qual", lambda: None)
    get_file_fredd = shared.get("file_fredd", lambda: None)

    norm_colname = shared.get("norm_colname", lambda s: " ".join(str(s).strip().lower().split()))
    coerce_numeric = shared.get("coerce_numeric", lambda s: pd.to_numeric(s, errors="coerce"))
    coerce_time_to_hours_auto = shared.get("coerce_time_to_hours_auto", lambda s: pd.to_numeric(s, errors="coerce"))

    # ===================== THEME =====================
    BG = shared["BG"]
    CARD = shared["CARD"]
    GRID = shared["GRID"]
    TXT = shared["TXT"]
    MUTED = shared["MUTED"]
    ACC_ORANGE = shared.get("ACC_ORANGE", "#ff7a00")

    # ===================== SETTINGS =====================
    WORK_DAYS_MONTH = 22
    MAX_CALLS_PER_LEAD_PER_DAY = 2.0  # max 2 chiamate/lead/giorno

    MIN_TRAIN_DAYS = 5

    # solver range
    CPH_MAX = 180.0   # calls/hour aggregate team
    CPH_STEP = 0.5

    FALLBACK_AVG_CALL_MIN = 2.5
    FALLBACK_ANSWER_PCT = 45.0
    FALLBACK_INCALL_PCT = 25.0

    # IMPORTANT: disabilita il cap InCall (più robusto per dimensionamento calls)
    DISABLE_INCALL_CAP = True

    # ===================== IDS =====================
    TEAM_TABLE_ID = "teamsim-team-table"
    OUT_TABLE_ID = "teamsim-output-table"
    BTN_ADD = "teamsim-add-row"
    BTN_DEL = "teamsim-del-row"
    MODE_ID = "teamsim-mode"
    MIX_ID = "teamsim-mix"
    RUN_ID = "teamsim-run"
    RUN_STATUS_ID = "teamsim-run-status"

    # ===================== TEAM TABLE =====================
    # Manuale: solo n_op, h_lav, resa_h_target
    manual_cols = {"n_op", "h_lav", "resa_h_target"}

    team_columns = [
        {"name": "n.Op (manuale)", "id": "n_op", "type": "numeric", "editable": True},
        {"name": "h.Lav (op/g) (manuale)", "id": "h_lav", "type": "numeric", "editable": True},
        {"name": "Resa/h attesa (manuale)", "id": "resa_h_target", "type": "numeric", "editable": True},

        {"name": "Ore Team (g) (calc)", "id": "ore_team_day", "type": "numeric", "editable": False},
        {"name": "Positivi target (g) (calc)", "id": "pos_target_day", "type": "numeric", "editable": False},

        {"name": "Answer% (auto storico) (calc)", "id": "answer_pct_auto", "type": "numeric", "editable": False},
        {"name": "InCall% (auto storico) (calc)", "id": "incall_pct_auto", "type": "numeric", "editable": False},

        {"name": "Calls Team necessarie (g) (calc)", "id": "calls_needed_day", "type": "numeric", "editable": False},
        {"name": "Calls / op / g (calc)", "id": "calls_per_op_day", "type": "numeric", "editable": False},

        {"name": "Answer% effettivo (calc)", "id": "answer_eff_pct", "type": "numeric", "editable": False},
        {"name": "Answers effettive (g) (calc)", "id": "answers_eff_day", "type": "numeric", "editable": False},

        {"name": "Resa/h stimata (calc)", "id": "resa_eff_h", "type": "numeric", "editable": False},
        {"name": "Red% stimato (calc)", "id": "red_eff_pct", "type": "numeric", "editable": False},

        {"name": "Red% Qual (calc)", "id": "red_qual_pct", "type": "numeric", "editable": False},
        {"name": "Red% Fred (calc)", "id": "red_fred_pct", "type": "numeric", "editable": False},

        {"name": "Proc Qual eff (g) (calc)", "id": "proc_qual_eff_day", "type": "numeric", "editable": False},
        {"name": "Proc Fred eff (g) (calc)", "id": "proc_fred_eff_day", "type": "numeric", "editable": False},
        {"name": "Pos Qual eff (g) (calc)", "id": "pos_qual_eff_day", "type": "numeric", "editable": False},
        {"name": "Pos Fred eff (g) (calc)", "id": "pos_fred_eff_day", "type": "numeric", "editable": False},

        {"name": "ProcNeed Qual (g) (calc)", "id": "proc_need_qual_day", "type": "numeric", "editable": False},
        {"name": "ProcNeed Fred (g) (calc)", "id": "proc_need_fred_day", "type": "numeric", "editable": False},
        {"name": "LeadNeed (g) (calc)", "id": "lead_need_day", "type": "numeric", "editable": False},
    ]

    out_columns = [
        {"name": "Lead Need tot (mese)", "id": "lead_need_tot", "type": "numeric", "editable": False},
        {"name": "Qualif tot (mese)", "id": "qualif_tot", "type": "numeric", "editable": False},
        {"name": "Fredde tot (mese)", "id": "fredde_tot", "type": "numeric", "editable": False},
        {"name": "Lead pool min/giorno (calls/2)", "id": "lead_pool_day_min", "type": "numeric", "editable": False},
    ]

    def blank_team_row() -> Dict[str, Any]:
        return {c["id"]: None for c in team_columns}

    TEAM_INIT = [blank_team_row()]
    OUT_INIT = [{"lead_need_tot": None, "qualif_tot": None, "fredde_tot": None, "lead_pool_day_min": None}]

    # ===================== HELPERS =====================
    def _to_float(v) -> float:
        try:
            x = float(v)
            return x if np.isfinite(x) else np.nan
        except Exception:
            return np.nan

    def _fmt_int(x: float) -> Optional[int]:
        if not np.isfinite(x):
            return None
        return int(round(float(x)))

    def _get_mix_ratios(mix_value: str) -> Tuple[float, float]:
        # mix_value = "80/20" means Fredde/Qualif
        if not isinstance(mix_value, str):
            return 0.8, 0.2
        mv = mix_value.strip().lower()
        if mv in {"100% fredde", "100% freddo", "100 fredde"}:
            return 1.0, 0.0
        if mv in {"100% qualificate", "100 qualificate"}:
            return 0.0, 1.0
        try:
            a, b = mix_value.split("/")
            wf = float(a) / 100.0
            wq = float(b) / 100.0
            s = wf + wq
            if s <= 0:
                return 0.8, 0.2
            return wf / s, wq / s
        except Exception:
            return 0.8, 0.2

    def _active_sources(wf: float, wq: float) -> Tuple[bool, bool]:
        # ritorna (use_fred, use_qual)
        use_fred = bool(np.isfinite(wf) and wf > 1e-12)
        use_qual = bool(np.isfinite(wq) and wq > 1e-12)
        return use_fred, use_qual

    # ===================== DATE COL DETECTION =====================
    def _find_best_date_col(df: pd.DataFrame) -> Optional[str]:
        for cand in ["Data", "data", "DATA", "Date", "date"]:
            if cand in df.columns:
                return cand
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
            ok = float(dt.notna().mean())
            if ok < 0.55:
                continue
            uniq = int(dt.dropna().dt.date.nunique())
            if uniq < 2:
                continue
            score = ok * 0.85 + min(uniq / n, 1.0) * 0.15
            if score > best_score:
                best_score = score
                best = c
        return best

    # ===================== DAILY DATASET (TEAM, src=qual/fred) =====================
    def _build_daily_team_dataset_all() -> pd.DataFrame:
        df_all = get_df_all()
        fq = get_file_qual()
        ff = get_file_fredd()

        if df_all is None or df_all.empty or (not fq) or (not ff) or "__source_file" not in df_all.columns:
            return pd.DataFrame()

        df_q = df_all[df_all["__source_file"].isin([fq, ff])].copy()
        if df_q.empty:
            return pd.DataFrame()

        date_col = _find_best_date_col(df_q)
        if not date_col:
            return pd.DataFrame()

        df_q[date_col] = pd.to_datetime(df_q[date_col], errors="coerce", dayfirst=True, infer_datetime_format=True)
        df_q = df_q.dropna(subset=[date_col]).copy()
        if df_q.empty:
            return pd.DataFrame()

        df_q["__date__"] = df_q[date_col].dt.date.astype(str)
        df_q["src"] = np.where(df_q["__source_file"].astype(str) == str(ff), "fred", "qual")

        cm = {norm_colname(c): c for c in df_q.columns}

        def col(name: str) -> Optional[str]:
            return cm.get(norm_colname(name))

        c_calls = col("Nr. chiamate effettuate")
        c_ans = col("Nr. chiamate con risposta")
        c_proc = col("Processati")
        c_pos = col("Positivi")
        c_posc = col("Positivi confermati")
        c_ore = col("Lavorazione  generale")
        c_incall = col("In chiamata")

        if c_calls is None or c_ans is None or c_proc is None or c_ore is None or (c_pos is None and c_posc is None):
            return pd.DataFrame()

        tmp = df_q.copy()
        tmp[c_calls] = coerce_numeric(tmp[c_calls])
        tmp[c_ans] = coerce_numeric(tmp[c_ans])
        tmp[c_proc] = coerce_numeric(tmp[c_proc])
        tmp[c_ore] = coerce_time_to_hours_auto(tmp[c_ore])

        has_incall = c_incall is not None
        if has_incall:
            tmp[c_incall] = coerce_time_to_hours_auto(tmp[c_incall])

        if c_pos is None:
            tmp["__pos__"] = coerce_numeric(tmp[c_posc])
        elif c_posc is None:
            tmp["__pos__"] = coerce_numeric(tmp[c_pos])
        else:
            merged = tmp[c_pos].where(tmp[c_pos].notna(), tmp[c_posc])
            tmp["__pos__"] = coerce_numeric(merged)

        g = tmp.groupby(["__date__", "src"], dropna=False)

        daily = pd.DataFrame(
            {
                "calls": g[c_calls].sum(min_count=1),
                "answers": g[c_ans].sum(min_count=1),
                "processati": g[c_proc].sum(min_count=1),
                "positivi": g["__pos__"].sum(min_count=1),
                "ore": g[c_ore].sum(min_count=1),
            }
        ).reset_index()

        if has_incall:
            daily["incall_ore"] = g[c_incall].sum(min_count=1).reset_index(drop=True)
        else:
            daily["incall_ore"] = np.nan

        daily["answer_pct"] = (daily["answers"] / daily["calls"].replace(0, np.nan)) * 100.0
        daily["incall_pct"] = (daily["incall_ore"] / daily["ore"].replace(0, np.nan)) * 100.0

        daily["calls_per_hour"] = daily["calls"] / daily["ore"].replace(0, np.nan)
        daily["answers_per_hour"] = daily["answers"] / daily["ore"].replace(0, np.nan)

        daily["proc_per_hour"] = daily["processati"] / daily["ore"].replace(0, np.nan)
        daily["pos_per_hour"] = daily["positivi"] / daily["ore"].replace(0, np.nan)

        daily["red_pct"] = (daily["positivi"] / daily["processati"].replace(0, np.nan)) * 100.0
        daily["resa_h"] = daily["positivi"] / daily["ore"].replace(0, np.nan)

        daily["avg_call_min"] = (daily["incall_ore"] * 60.0) / daily["answers"].replace(0, np.nan)

        daily = daily.replace([np.inf, -np.inf], np.nan)
        daily = daily.dropna(subset=["ore", "calls", "answers", "calls_per_hour", "answers_per_hour"])
        return daily

    # ===================== BASELINES =====================
    def _compute_baselines(daily_all: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for src in ["qual", "fred"]:
            d = daily_all[daily_all["src"] == src].copy()
            if d.empty:
                out[src] = {
                    "red_pct_median": np.nan,
                    "proc_ph_median": np.nan,
                    "pos_ph_median": np.nan,
                    "avg_call_min_median": np.nan,
                    "incall_pct_median": np.nan,
                    "answer_pct_median": np.nan,
                    "n_days": 0,
                }
                continue

            out[src] = {
                "red_pct_median": float(np.nanmedian(d["red_pct"].to_numpy(float))) if np.isfinite(d["red_pct"]).any() else np.nan,
                "proc_ph_median": float(np.nanmedian(d["proc_per_hour"].to_numpy(float))) if np.isfinite(d["proc_per_hour"]).any() else np.nan,
                "pos_ph_median": float(np.nanmedian(d["pos_per_hour"].to_numpy(float))) if np.isfinite(d["pos_per_hour"]).any() else np.nan,
                "avg_call_min_median": float(np.nanmedian(d["avg_call_min"].to_numpy(float))) if np.isfinite(d["avg_call_min"]).any() else np.nan,
                "incall_pct_median": float(np.nanmedian(d["incall_pct"].to_numpy(float))) if np.isfinite(d["incall_pct"]).any() else np.nan,
                "answer_pct_median": float(np.nanmedian(d["answer_pct"].to_numpy(float))) if np.isfinite(d["answer_pct"]).any() else np.nan,
                "n_days": int(len(d)),
            }

            if not np.isfinite(out[src]["avg_call_min_median"]):
                out[src]["avg_call_min_median"] = FALLBACK_AVG_CALL_MIN

        return out

    # ===================== FIT ML PER SOURCE (qual/fred) =====================
    def _fit_models_per_source(daily_all: pd.DataFrame, model_kind: str) -> Dict[str, Any]:
        if daily_all is None or daily_all.empty or "src" not in daily_all.columns:
            return {}

        base = _compute_baselines(daily_all)
        out: Dict[str, Any] = {"kind": model_kind, "models": {}, "base": base}

        for src in ["qual", "fred"]:
            d = daily_all[daily_all["src"] == src].copy()
            if len(d) < MIN_TRAIN_DAYS:
                continue

            X = d[["calls_per_hour", "answers_per_hour", "incall_pct"]].copy()
            y_proc = pd.to_numeric(d["proc_per_hour"], errors="coerce").to_numpy(float)
            y_pos = pd.to_numeric(d["pos_per_hour"], errors="coerce").to_numpy(float)

            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(X.median(numeric_only=True)).fillna(0.0)

            mask = np.isfinite(y_proc) & np.isfinite(y_pos)
            if int(mask.sum()) < MIN_TRAIN_DAYS:
                continue

            Xn = X.to_numpy(float)[mask]
            y_proc = y_proc[mask]
            y_pos = y_pos[mask]

            try:
                if model_kind == "hgb":
                    from sklearn.ensemble import HistGradientBoostingRegressor
                    m_proc = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.08, max_iter=300, random_state=7)
                    m_pos = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.08, max_iter=300, random_state=7)
                else:
                    from sklearn.linear_model import Ridge
                    m_proc = Ridge(alpha=10.0, fit_intercept=True)
                    m_pos = Ridge(alpha=10.0, fit_intercept=True)

                m_proc.fit(Xn, y_proc)
                m_pos.fit(Xn, y_pos)
                out["models"][src] = {"m_proc": m_proc, "m_pos": m_pos}
            except Exception:
                continue

        return out

    # ===================== PREDICT (ML se c'è, altrimenti baseline) =====================
    def _sanitize_Xrow(x: np.ndarray) -> np.ndarray:
        if x is None or not isinstance(x, np.ndarray) or x.shape != (1, 3):
            x = np.array([[0.0, 0.0, 0.0]], dtype=float)
        x = np.nan_to_num(x.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
        x[0, 0] = float(np.clip(x[0, 0], 0.0, 5000.0))
        x[0, 1] = float(np.clip(x[0, 1], 0.0, 5000.0))
        x[0, 2] = float(np.clip(x[0, 2], 0.0, 100.0))
        return x

    def _predict_proc_pos_ph(pack: Dict[str, Any], src: str, Xrow: np.ndarray) -> Tuple[float, float]:
        Xrow = _sanitize_Xrow(Xrow)

        ms = (pack.get("models") or {}).get(src)
        if ms:
            try:
                proc = float(ms["m_proc"].predict(Xrow)[0])
                pos = float(ms["m_pos"].predict(Xrow)[0])
                return max(proc, 0.0), max(pos, 0.0)
            except Exception:
                pass

        base = (pack.get("base") or {}).get(src, {})
        proc_b = float(base.get("proc_ph_median", np.nan))
        pos_b = float(base.get("pos_ph_median", np.nan))
        proc_b = proc_b if np.isfinite(proc_b) else 0.0
        pos_b = pos_b if np.isfinite(pos_b) else 0.0
        return proc_b, pos_b

    # ===================== ANSWERS (cap opzionale) =====================
    def _answers_eff(
        calls_day: float,
        answer_pct_target: float,
        incall_pct_target: float,
        ore_team: float,
        avg_call_min_mix: float,
    ) -> Tuple[float, float]:
        if not (np.isfinite(calls_day) and calls_day > 0 and np.isfinite(answer_pct_target)):
            return np.nan, np.nan

        ans_raw = calls_day * (answer_pct_target / 100.0)

        if DISABLE_INCALL_CAP:
            ans_eff = ans_raw
        else:
            if not (np.isfinite(avg_call_min_mix) and avg_call_min_mix > 0 and np.isfinite(incall_pct_target) and np.isfinite(ore_team) and ore_team > 0):
                ans_eff = ans_raw
            else:
                incall_ore = (incall_pct_target / 100.0) * ore_team
                cap = (incall_ore * 60.0) / avg_call_min_mix if incall_ore > 0 else 0.0
                cap = cap if np.isfinite(cap) else ans_raw
                ans_eff = min(ans_raw, cap)

        ans_eff = float(max(ans_eff, 0.0)) if np.isfinite(ans_eff) else np.nan
        ans_eff_pct = (ans_eff / calls_day) * 100.0 if np.isfinite(ans_eff) and calls_day > 0 else np.nan
        return ans_eff, ans_eff_pct

    # ===================== AUTO Answer/InCall from baselines (mix) =====================
    def _get_auto_answer_incall(pack: Dict[str, Any], wf: float, wq: float) -> Tuple[float, float]:
        base = pack.get("base", {}) if isinstance(pack, dict) else {}

        aq = float(base.get("qual", {}).get("answer_pct_median", np.nan))
        af = float(base.get("fred", {}).get("answer_pct_median", np.nan))
        iq = float(base.get("qual", {}).get("incall_pct_median", np.nan))
        inf = float(base.get("fred", {}).get("incall_pct_median", np.nan))

        if not np.isfinite(aq):
            aq = FALLBACK_ANSWER_PCT
        if not np.isfinite(af):
            af = FALLBACK_ANSWER_PCT
        if not np.isfinite(iq):
            iq = FALLBACK_INCALL_PCT
        if not np.isfinite(inf):
            inf = FALLBACK_INCALL_PCT

        use_fred, use_qual = _active_sources(wf, wq)

        if use_qual and not use_fred:
            ans = aq
            inc = iq
        elif use_fred and not use_qual:
            ans = af
            inc = inf
        else:
            ans = (wq * aq + wf * af)
            inc = (wq * iq + wf * inf)

        return float(np.clip(ans, 0.0, 100.0)), float(np.clip(inc, 0.0, 100.0))

    # ===================== SIMULATE GIVEN CALLS =====================
    def _simulate_given_calls(
        pack: Dict[str, Any],
        wf: float,
        wq: float,
        ore_team: float,
        calls_day: float,
        answer_pct_target: float,
        incall_pct_target: float,
    ) -> Dict[str, float]:
        ore_team = _to_float(ore_team)
        calls_day = _to_float(calls_day)
        answer_pct_target = _to_float(answer_pct_target)
        incall_pct_target = _to_float(incall_pct_target)

        if not (np.isfinite(ore_team) and ore_team > 0 and np.isfinite(calls_day) and calls_day >= 0):
            return {}

        wf = float(max(wf, 0.0)) if np.isfinite(wf) else 0.0
        wq = float(max(wq, 0.0)) if np.isfinite(wq) else 0.0
        s = wf + wq
        if s <= 0:
            wf, wq = 0.8, 0.2
        else:
            wf, wq = wf / s, wq / s

        use_fred, use_qual = _active_sources(wf, wq)

        answer_pct_target = float(np.clip(answer_pct_target if np.isfinite(answer_pct_target) else 0.0, 0.0, 100.0))
        incall_pct_target = float(np.clip(incall_pct_target if np.isfinite(incall_pct_target) else 0.0, 0.0, 100.0))

        base = pack.get("base", {})

        # avg_call_min: solo fonte attiva se 100%, altrimenti mix
        avg_q = float(base.get("qual", {}).get("avg_call_min_median", FALLBACK_AVG_CALL_MIN))
        avg_f = float(base.get("fred", {}).get("avg_call_min_median", FALLBACK_AVG_CALL_MIN))
        if not np.isfinite(avg_q):
            avg_q = FALLBACK_AVG_CALL_MIN
        if not np.isfinite(avg_f):
            avg_f = FALLBACK_AVG_CALL_MIN

        if use_qual and not use_fred:
            avg_call_min_mix = avg_q
        elif use_fred and not use_qual:
            avg_call_min_mix = avg_f
        else:
            avg_call_min_mix = (wq * avg_q + wf * avg_f)

        if not (np.isfinite(avg_call_min_mix) and avg_call_min_mix > 0):
            avg_call_min_mix = FALLBACK_AVG_CALL_MIN

        calls_per_hour = float(np.clip(calls_day / ore_team, 0.0, 5000.0))

        answers_eff, answer_eff_pct = _answers_eff(
            calls_day=calls_day,
            answer_pct_target=answer_pct_target,
            incall_pct_target=incall_pct_target,
            ore_team=ore_team,
            avg_call_min_mix=avg_call_min_mix,
        )
        if not np.isfinite(answers_eff):
            answers_eff = calls_day * (answer_pct_target / 100.0)
        answers_eff = float(max(answers_eff, 0.0))

        answers_per_hour = float(np.clip(answers_eff / ore_team, 0.0, 5000.0))
        Xrow = np.array([[calls_per_hour, answers_per_hour, incall_pct_target]], dtype=float)

        # calcolo SOLO per le fonti attive
        if use_qual:
            proc_ph_q, pos_ph_q = _predict_proc_pos_ph(pack, "qual", Xrow)
            proc_q_raw = proc_ph_q * ore_team
            pos_q_raw = pos_ph_q * ore_team
        else:
            proc_q_raw = 0.0
            pos_q_raw = 0.0

        if use_fred:
            proc_ph_f, pos_ph_f = _predict_proc_pos_ph(pack, "fred", Xrow)
            proc_f_raw = proc_ph_f * ore_team
            pos_f_raw = pos_ph_f * ore_team
        else:
            proc_f_raw = 0.0
            pos_f_raw = 0.0

        # red per fonte (solo se attiva)
        red_q = (pos_q_raw / proc_q_raw * 100.0) if (use_qual and proc_q_raw > 0) else np.nan
        red_f = (pos_f_raw / proc_f_raw * 100.0) if (use_fred and proc_f_raw > 0) else np.nan

        red_q_b = float(base.get("qual", {}).get("red_pct_median", np.nan))
        red_f_b = float(base.get("fred", {}).get("red_pct_median", np.nan))

        if use_qual and (not np.isfinite(red_q) or red_q <= 0):
            red_q = red_q_b if np.isfinite(red_q_b) else np.nan
        if use_fred and (not np.isfinite(red_f) or red_f <= 0):
            red_f = red_f_b if np.isfinite(red_f_b) else np.nan

        # mix (quota pratiche) – se 100% uno dei due, l'altro resta 0
        proc_q_eff = wq * proc_q_raw if use_qual else 0.0
        proc_f_eff = wf * proc_f_raw if use_fred else 0.0
        pos_q_eff = wq * pos_q_raw if use_qual else 0.0
        pos_f_eff = wf * pos_f_raw if use_fred else 0.0

        proc_tot = float(np.nansum([proc_q_eff, proc_f_eff]))
        pos_tot = float(np.nansum([pos_q_eff, pos_f_eff]))

        red_eff = (pos_tot / proc_tot * 100.0) if proc_tot > 0 else np.nan
        resa_eff = (pos_tot / ore_team) if ore_team > 0 else np.nan

        return {
            "calls_day": float(calls_day),
            "answers_eff": float(answers_eff),
            "answer_eff_pct": float(answer_eff_pct) if np.isfinite(answer_eff_pct) else np.nan,
            "proc_tot": proc_tot,
            "pos_tot": pos_tot,
            "red_eff": float(red_eff) if np.isfinite(red_eff) else np.nan,
            "resa_eff": float(resa_eff) if np.isfinite(resa_eff) else np.nan,
            "red_q": float(red_q) if np.isfinite(red_q) else np.nan,
            "red_f": float(red_f) if np.isfinite(red_f) else np.nan,
            "proc_q_eff": float(proc_q_eff),
            "proc_f_eff": float(proc_f_eff),
            "pos_q_eff": float(pos_q_eff),
            "pos_f_eff": float(pos_f_eff),
        }

    # ===================== FIND CALLS NEEDED (targets) =====================
    # vincolo: solo resa/positivi (Red% non è input)
    def _find_calls_needed(
        pack: Dict[str, Any],
        wf: float,
        wq: float,
        ore_team: float,
        pos_target_day: float,
        resa_target_h: float,
        answer_pct_target: float,
        incall_pct_target: float,
    ) -> Tuple[float, Dict[str, float], str]:
        ore_team = _to_float(ore_team)
        if not (np.isfinite(ore_team) and ore_team > 0):
            return np.nan, {}, "ore team non valide"

        pos_target_day = _to_float(pos_target_day)
        resa_target_h = _to_float(resa_target_h)

        pos_target_day = float(max(pos_target_day, 0.0)) if np.isfinite(pos_target_day) else 0.0
        resa_target_h = float(max(resa_target_h, 0.0)) if np.isfinite(resa_target_h) else 0.0

        for cph in np.arange(0.0, CPH_MAX + 1e-9, CPH_STEP):
            calls_day = float(cph * ore_team)

            sim = _simulate_given_calls(pack, wf, wq, ore_team, calls_day, answer_pct_target, incall_pct_target)
            if not sim:
                continue

            pos_day = sim.get("pos_tot", np.nan)
            resa_eff = sim.get("resa_eff", np.nan)

            if not (np.isfinite(pos_day) and np.isfinite(resa_eff)):
                continue

            ok_pos = pos_day >= pos_target_day - 1e-9
            ok_resa = resa_eff >= resa_target_h - 1e-9

            if ok_pos and ok_resa:
                return calls_day, sim, ""

        calls_day = float(CPH_MAX * ore_team)
        sim = _simulate_given_calls(pack, wf, wq, ore_team, calls_day, answer_pct_target, incall_pct_target)
        return calls_day, (sim or {}), "Target non raggiungibile nel range: uso max calls/hour"

    # ===================== RUN CALC ROW =====================
    def _run_calc_row(row: Dict[str, Any], mix_value: str, mode: str) -> Tuple[Dict[str, Any], str]:
        out = dict(row)

        n_op = _to_float(out.get("n_op"))
        h_lav = _to_float(out.get("h_lav"))
        resa_h_t = _to_float(out.get("resa_h_target"))

        if not (np.isfinite(n_op) and n_op > 0):
            return out, "Compila n.Op (poi Enter o click fuori)"
        if not (np.isfinite(h_lav) and h_lav > 0):
            return out, "Compila h.Lav (poi Enter o click fuori)"
        if not (np.isfinite(resa_h_t) and resa_h_t >= 0):
            return out, "Compila Resa/h attesa"

        ore_team = float(n_op * h_lav)
        pos_target = float(resa_h_t * ore_team)

        out["ore_team_day"] = round(ore_team, 4)
        out["pos_target_day"] = round(pos_target, 4)

        daily_all = _build_daily_team_dataset_all()
        if daily_all.empty:
            return out, "Dataset daily vuoto: controlla colonne nei CSV"

        wf, wq = _get_mix_ratios(mix_value)
        use_fred, use_qual = _active_sources(wf, wq)

        mk = "hgb" if mode == "ml_hgb" else "ridge"
        pack = _fit_models_per_source(daily_all, model_kind=mk)

        ans_auto, inc_auto = _get_auto_answer_incall(pack, wf=wf, wq=wq)
        out["answer_pct_auto"] = round(ans_auto, 2)
        out["incall_pct_auto"] = round(inc_auto, 2)

        calls_needed, _sim_det, warn = _find_calls_needed(
            pack=pack,
            wf=wf,
            wq=wq,
            ore_team=ore_team,
            pos_target_day=pos_target,
            resa_target_h=resa_h_t,
            answer_pct_target=ans_auto,
            incall_pct_target=inc_auto,
        )

        calls_used = float(calls_needed) if np.isfinite(calls_needed) else np.nan
        sim = _simulate_given_calls(pack, wf, wq, ore_team, calls_used, ans_auto, inc_auto) if np.isfinite(calls_used) else {}
        if not sim:
            return out, "Simulazione fallita"

        out["calls_needed_day"] = round(float(calls_needed), 2) if np.isfinite(calls_needed) else None
        cpo = (calls_used / n_op) if (np.isfinite(calls_used) and np.isfinite(n_op) and n_op > 0) else np.nan
        out["calls_per_op_day"] = round(float(cpo), 2) if np.isfinite(cpo) else None

        out["answer_eff_pct"] = round(float(sim.get("answer_eff_pct", np.nan)), 2) if np.isfinite(sim.get("answer_eff_pct", np.nan)) else None
        out["answers_eff_day"] = round(float(sim.get("answers_eff", np.nan)), 2) if np.isfinite(sim.get("answers_eff", np.nan)) else None

        out["resa_eff_h"] = round(float(sim.get("resa_eff", np.nan)), 6) if np.isfinite(sim.get("resa_eff", np.nan)) else None
        out["red_eff_pct"] = round(float(sim.get("red_eff", np.nan)), 4) if np.isfinite(sim.get("red_eff", np.nan)) else None

        red_q = float(sim.get("red_q", np.nan))
        red_f = float(sim.get("red_f", np.nan))

        # se fonte non attiva, mostra 0/None coerenti
        out["red_qual_pct"] = (round(red_q, 4) if (use_qual and np.isfinite(red_q)) else 0)
        out["red_fred_pct"] = (round(red_f, 4) if (use_fred and np.isfinite(red_f)) else 0)

        out["proc_qual_eff_day"] = (round(float(sim.get("proc_q_eff", 0.0)), 4) if use_qual else 0)
        out["proc_fred_eff_day"] = (round(float(sim.get("proc_f_eff", 0.0)), 4) if use_fred else 0)
        out["pos_qual_eff_day"] = (round(float(sim.get("pos_q_eff", 0.0)), 4) if use_qual else 0)
        out["pos_fred_eff_day"] = (round(float(sim.get("pos_f_eff", 0.0)), 4) if use_fred else 0)

        # lead consumption: SOLO per fonti attive
        pos_target_q = pos_target * wq if use_qual else 0.0
        pos_target_f = pos_target * wf if use_fred else 0.0

        proc_need_q = (pos_target_q / (red_q / 100.0)) if (use_qual and np.isfinite(red_q) and red_q > 0) else 0.0
        proc_need_f = (pos_target_f / (red_f / 100.0)) if (use_fred and np.isfinite(red_f) and red_f > 0) else 0.0

        lead_need_day = float(np.nansum([proc_need_q, proc_need_f]))

        out["proc_need_qual_day"] = round(float(proc_need_q), 4) if use_qual else 0
        out["proc_need_fred_day"] = round(float(proc_need_f), 4) if use_fred else 0
        out["lead_need_day"] = round(float(lead_need_day), 4) if np.isfinite(lead_need_day) else None

        return out, warn or ""

    # ===================== MONTHLY OUTPUT =====================
    def _monthly_output(rows: List[Dict[str, Any]]) -> Dict[str, Optional[int]]:
        qual_month = 0.0
        fred_month = 0.0
        lead_pool_day_min = 0.0
        ok = False

        for r in rows or []:
            pq = _to_float(r.get("proc_need_qual_day"))
            pf = _to_float(r.get("proc_need_fred_day"))
            ld = _to_float(r.get("lead_need_day"))
            cu = _to_float(r.get("calls_needed_day"))

            if np.isfinite(pq) and pq > 0:
                qual_month += pq * WORK_DAYS_MONTH
                ok = True
            if np.isfinite(pf) and pf > 0:
                fred_month += pf * WORK_DAYS_MONTH
                ok = True

            if (not ok) and np.isfinite(ld):
                ok = True

            if np.isfinite(cu):
                lead_pool_day_min = max(lead_pool_day_min, (cu / MAX_CALLS_PER_LEAD_PER_DAY))

        if not ok:
            return {
                "lead_need_tot": None,
                "qualif_tot": None,
                "fredde_tot": None,
                "lead_pool_day_min": _fmt_int(lead_pool_day_min) if np.isfinite(lead_pool_day_min) else None,
            }

        lead_tot = qual_month + fred_month
        return {
            "lead_need_tot": _fmt_int(lead_tot) if np.isfinite(lead_tot) else None,
            "qualif_tot": _fmt_int(qual_month) if np.isfinite(qual_month) else None,
            "fredde_tot": _fmt_int(fred_month) if np.isfinite(fred_month) else None,
            "lead_pool_day_min": _fmt_int(lead_pool_day_min) if np.isfinite(lead_pool_day_min) else None,
        }

    # ===================== UI =====================
    style_table_common = {
        "overflowX": "auto",
        "border": f"1px solid {GRID}",
        "borderRadius": "12px",
        "backgroundColor": CARD,
        "width": "100%",
    }
    style_header_common = {
        "backgroundColor": "#121c2a",
        "color": TXT,
        "fontWeight": "800",
        "fontSize": "11px",
        "border": f"1px solid {GRID}",
    }

    manual_style = [
        {"if": {"column_id": c}, "backgroundColor": ACC_ORANGE, "color": "#0b0f14", "fontWeight": "800"}
        for c in manual_cols
    ]

    team_table = dash_table.DataTable(
        id=TEAM_TABLE_ID,
        columns=team_columns,
        data=TEAM_INIT,
        editable=True,
        row_deletable=False,
        sort_action="none",
        filter_action="none",
        page_action="none",
        style_table=style_table_common,
        style_header=style_header_common,
        style_cell={
            "backgroundColor": CARD,
            "color": TXT,
            "fontFamily": "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Courier New', monospace",
            "fontSize": "12px",
            "padding": "6px 8px",
            "whiteSpace": "nowrap",
            "border": f"1px solid {GRID}",
            "minWidth": "190px",
            "maxWidth": "350px",
        },
        style_data_conditional=manual_style
        + [
            {"if": {"row_index": "odd"}, "backgroundColor": "#0d1420"},
            {"if": {"state": "active"}, "border": f"1px solid {ACC_ORANGE}"},
        ],
    )

    out_table = dash_table.DataTable(
        id=OUT_TABLE_ID,
        columns=out_columns,
        data=OUT_INIT,
        editable=False,
        row_deletable=False,
        sort_action="none",
        filter_action="none",
        page_action="none",
        style_table=style_table_common,
        style_header=style_header_common,
        style_cell={
            "backgroundColor": "#0d1420",
            "color": TXT,
            "fontFamily": "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Courier New', monospace",
            "fontSize": "12px",
            "padding": "8px 10px",
            "whiteSpace": "nowrap",
            "border": f"1px solid {GRID}",
            "minWidth": "240px",
            "maxWidth": "380px",
        },
    )

    header_row = dbc.Row(
        [
            dbc.Col(html.H4("Team", style={"color": TXT, "fontWeight": 900, "marginBottom": "0px"}), width="auto"),
            dbc.Col(
                html.Div(
                    [
                        dbc.Button("+", id=BTN_ADD, n_clicks=0, size="sm", color="secondary", outline=True,
                                   style={"width": "30px", "padding": "0px"}),
                        dbc.Button("-", id=BTN_DEL, n_clicks=0, size="sm", color="secondary", outline=True,
                                   style={"width": "30px", "padding": "0px", "marginLeft": "6px"}),
                    ],
                    style={"display": "flex", "justifyContent": "flex-end", "alignItems": "center"},
                ),
                width=True,
            ),
        ],
        align="center",
        className="g-2",
        style={"marginBottom": "10px"},
    )

    # CSS sicuro per dropdown: usa /assets
    # Crea il file: assets/dropdown.css con questo contenuto:
    #
    # .dd-black .Select-control,
    # .dd-black .Select-value-label,
    # .dd-black .Select-placeholder,
    # .dd-black .Select-input > input { color:#000 !important; }
    # .dd-black .Select-menu-outer,
    # .dd-black .Select-menu,
    # .dd-black .Select-option { color:#000 !important; }
    # .dd-black .VirtualizedSelectOption,
    # .dd-black .VirtualizedSelectFocusedOption { color:#000 !important; }

    controls_row = dbc.Row(
        [
            dbc.Col(
                [
                    html.Div("Modalità ML", style={"color": MUTED, "fontSize": "12px"}),
                    dcc.Dropdown(
                        id=MODE_ID,
                        className="dd-black",
                        options=[
                            {"label": "ML — Ridge (qual/fred separati)", "value": "ml_ridge"},
                            {"label": "ML — HistGradientBoosting (qual/fred separati)", "value": "ml_hgb"},
                        ],
                        value="ml_ridge",
                        clearable=False,
                        style={"fontSize": "12px"},
                    ),
                ],
                width=4,
            ),
            dbc.Col(
                [
                    html.Div("Mix Fredde/Qualificate", style={"color": MUTED, "fontSize": "12px"}),
                    dcc.Dropdown(
                        id=MIX_ID,
                        className="dd-black",
                        options=[
                            {"label": "90/10 (Fredde/Qualif)", "value": "90/10"},
                            {"label": "80/20 (Fredde/Qualif)", "value": "80/20"},
                            {"label": "70/30 (Fredde/Qualif)", "value": "70/30"},
                            {"label": "60/40 (Fredde/Qualif)", "value": "60/40"},
                            {"label": "50/50 (Fredde/Qualif)", "value": "50/50"},
                            {"label": "100% Fredde", "value": "100/0"},
                            {"label": "100% Qualificate", "value": "0/100"},
                        ],
                        value="80/20",
                        clearable=False,
                        style={"fontSize": "12px"},
                    ),
                ],
                width=4,
            ),
            dbc.Col(
                [
                    html.Div("Run (calcola)", style={"color": MUTED, "fontSize": "12px"}),
                    dbc.Button("Run", id=RUN_ID, n_clicks=0, color="warning", outline=True, style={"width": "100%"}),
                ],
                width=2,
            ),
            dbc.Col(
                [
                    html.Div("Stato", style={"color": MUTED, "fontSize": "12px"}),
                    html.Div(id=RUN_STATUS_ID, style={"color": TXT, "fontSize": "12px", "fontFamily": "monospace"}),
                ],
                width=2,
            ),
        ],
        className="g-2",
        style={"marginBottom": "12px"},
    )

    card = dbc.Card(
        dbc.CardBody(
            [
                (dbc.Alert(get_DATA_ERR(), color="danger") if get_DATA_ERR() else html.Div()),
                header_row,
                html.Div(
                    "Nota: Red% non è più un input: viene calcolato come output atteso. "
                    "Calls Team vengono calcolate come output (calls necessarie per raggiungere i target). "
                    "Se selezioni 100% Qualificate si calcola SOLO su Qualificate; se 100% Fredde si calcola SOLO su Fredde; "
                    "negli altri casi si usa il mix pesato.",
                    style={"color": MUTED, "fontSize": "12px", "marginBottom": "10px"},
                ),
                controls_row,
                team_table,
                html.Hr(style={"borderColor": GRID, "marginTop": "16px", "marginBottom": "16px"}),
                html.H4("Output", style={"color": TXT, "fontWeight": 900, "marginBottom": "10px"}),
                out_table,
            ]
        ),
        style={"backgroundColor": BG, "border": f"1px solid {GRID}", "borderRadius": "18px"},
    )

    # ===================== CALLBACK =====================
    @app.callback(
        Output(TEAM_TABLE_ID, "data"),
        Output(OUT_TABLE_ID, "data"),
        Output(RUN_STATUS_ID, "children"),
        Input(BTN_ADD, "n_clicks"),
        Input(BTN_DEL, "n_clicks"),
        Input(RUN_ID, "n_clicks"),
        State(TEAM_TABLE_ID, "data"),
        State(MODE_ID, "value"),
        State(MIX_ID, "value"),
        prevent_initial_call=False,
    )
    def _update(n_add, n_del, n_run, data: List[Dict[str, Any]], mode: str, mix_value: str):
        if not data:
            data = [blank_team_row()]

        trig = ctx.triggered_id or ""

        if trig == BTN_ADD:
            data = list(data) + [blank_team_row()]
            return data, OUT_INIT, "Aggiunta riga"

        if trig == BTN_DEL:
            if len(data) > 1:
                data = list(data)[:-1]
                return data, OUT_INIT, "Rimossa riga"
            return data, OUT_INIT, "Non posso rimuovere l’ultima riga"

        if trig != RUN_ID:
            return data, OUT_INIT, "Edita celle arancioni, poi Run"

        computed_rows: List[Dict[str, Any]] = []
        warn = ""
        for r in data:
            rr, w = _run_calc_row(r, mix_value=mix_value, mode=mode)
            computed_rows.append(rr)
            if w and not warn:
                warn = w

        out_row = _monthly_output(computed_rows)
        out = [out_row]

        wf, wq = _get_mix_ratios(mix_value)
        mode_label = {"ml_ridge": "ML Ridge", "ml_hgb": "ML HGB"}.get(mode, str(mode))
        status = f"OK | {mode_label} | Mix F/Q={mix_value} (wf={wf:.2f}, wq={wq:.2f}) | Run={int(n_run or 0)}"
        if warn:
            status += f" | Warn: {warn}"

        return computed_rows, out, status

    return card
