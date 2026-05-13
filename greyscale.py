import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from scipy.stats import t as student_t
from scipy.stats import f as fisher_f
from scipy.stats import norm
from scipy.stats import gamma as gamma_dist

import yfinance as yf


# ============================================================
# ALL-IN-ONE PAPER FIGURE SCRIPT
# SV-DLM / RV-DLM / RVL-DLM
# FIXED HYPERPARAMETERS
# LATEX-READY FIGURE NAMES
# ============================================================
#
# This script:
#
#   1. Downloads Yahoo Finance OHLC data.
#   2. Computes Rogers--Satchell realized variance.
#   3. Runs fixed-hyperparameter SV-DLM, RV-DLM, and RVL-DLM.
#   4. Uses Student-t posterior bands for state trajectories:
#
#          theta_j,t | D_t ~ T_{n_t}(m_j,t, C_jj,t)
#
#      so 90% bands are:
#
#          m_j,t +/- t_{0.95,n_t} sqrt(C_jj,t)
#
#      This is essentially the normal approximation when n_t is large,
#      but gives wider intervals when n_t is small.
#
#   5. Saves all paper figures separately into:
#
#          RVDLMfigures/
#
#      with exactly the names used in the LaTeX file:
#
#          FigureA1.pdf, FigureA2.pdf
#          FigureB1.pdf, FigureB2.pdf, FigureB3.pdf
#          FigureC1.pdf, FigureC2.pdf, FigureC3.pdf
#          FigureD1.pdf, ..., FigureD6.pdf
#          FigureE1.pdf, ..., FigureE6.pdf
#          FigureF1.pdf, ..., FigureF4.pdf
#          FigureG1.pdf, FigureG2.pdf, FigureG3.pdf
#
# Fixed values:
#
#      delta      = 0.999
#      beta_SV    = 0.95
#      beta_RV    = 0.875
#      alpha      = 2.75
#
# ============================================================


# ============================================================
# Settings
# ============================================================

TICKERS = [
    "SPY",
    "XLB",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLU",
    "XLV",
    "XLY",
]

SELECTED_TICKERS = ["XLB", "XLU", "XLE"]

START_DATE = "2000-01-01"
END_DATE = "2026-01-01"

AUTO_ADJUST = True

# Post-2010 evaluation period
SCORE_START_DATE = "2010-01-01"

# For Figure A latent SD comparison
FIGURE_A_TICKER = "SPY"
FIGURE_A_START = "2021-01-01"
FIGURE_A_END = "2025-12-31"

# Fixed upper y-axis limit for FigureA1 and FigureA2.
# The SD scale in these plots is around 0.05, so 0.055 gives a clean cap.
# Change to 0.55 only if you literally want ten times more headroom.
FIGURE_A_YMAX = 0.055

# Zoomed coefficient figure window
FIGURE_ZOOM_START = "2023-01-01"
FIGURE_ZOOM_END = "2023-12-31"

# Volatility construction
INCLUDE_OVERNIGHT = False

# Response
Y_MODE = "log_price"      # "log_price" or "log_return"

# Fixed hyperparameters
DELTA = 0.999
SV_BETA = 0.95
RV_BETA = 0.875
ALPHA = 2.75

# Initial prior information
N0 = 1.0
EPS = 1e-12

# SV and RV both use lagged RV predictor x_{t-1}
SV_USES_LAGGED_RV = True

# If True:
#   RV/RVL score y_t after observing z_t:
#       log p(y_t | z_t, D_{t-1})
#

RV_PRICE_SCORE_AFTER_Z = True

# Output folder matching LaTeX:
# \graphicspath{{./RVDLMfigures/}}
FIGURE_DIR = Path("RVDLMfiguresGreyScale")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

SAVE_TABLES = True
SAVE_FILTERS = False

SHOW_PLOTS = False

# Figure style
PAPER_FIGSIZE = (6.5, 4.25)
PAPER_DPI = 250
PAPER_LINE_WIDTH = 1.8
PAPER_BAND_ALPHA = 0.25

BASE_FONT_SIZE = 14
AXIS_LABEL_SIZE = 16
TICK_LABEL_SIZE = 13

# Greyscale / print-safe figure controls
GREYSCALE_FIGURES = True

# In the current Matplotlib color-cycle version of Figure B,
# the formerly purple line is XLK.
FIGURE_B_SPECIAL_TICKER = "XLK"

# Use Student-t rather than normal state-trajectory bands.
USE_STUDENT_T_STATE_INTERVALS = True
STATE_INTERVAL_LEVEL = 0.90

plt.rcParams.update({
    "font.size": BASE_FONT_SIZE,
    "axes.labelsize": AXIS_LABEL_SIZE,
    "xtick.labelsize": TICK_LABEL_SIZE,
    "ytick.labelsize": TICK_LABEL_SIZE,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
})


# ============================================================
# General helpers
# ============================================================

def stabilize(C):
    C = 0.5 * (C + C.T)

    try:
        mn = float(np.linalg.eigvalsh(C).min())
    except np.linalg.LinAlgError:
        mn = -1.0

    if mn < EPS:
        C = C + np.eye(C.shape[0]) * (EPS - mn)

    return C


def num_slug(x):
    return str(x).replace("-", "m").replace(".", "p")


def is_scored_date(date):
    return pd.Timestamp(date) >= pd.Timestamp(SCORE_START_DATE)


def save_paper_pdf(fig, filename):
    """
    Save with fixed page size.

    No bbox_inches='tight' so all PDFs have consistent dimensions
    for LaTeX includegraphics/minipage layouts.
    """
    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=PAPER_DPI)
    print(f"Saved: {path}")


def paper_axis(ax, ylabel, zoomed=False, ylim=None):
    ax.set_ylabel(ylabel)

    if zoomed:
        ax.set_xlabel("Month/Year")
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%y"))
    else:
        ax.set_xlabel("Year")
        ax.xaxis.set_major_locator(mdates.YearLocator(base=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="both", top=False, right=False)
    ax.margins(x=0.02)


def state_interval_critical_value(level=STATE_INTERVAL_LEVEL, nt=None):
    """
    Critical value for equal-tail state intervals.

    If nt is available, use Student-t with nt degrees of freedom.
    Otherwise fall back to the normal approximation.
    """

    prob = 0.5 + level / 2.0

    if (
        USE_STUDENT_T_STATE_INTERVALS
        and nt is not None
        and np.isfinite(float(nt))
        and float(nt) > 0.0
    ):
        return float(student_t.ppf(prob, df=float(nt)))

    return float(norm.ppf(prob))


def add_state_summaries(row, mt, Ct, st=None, nt=None, level=STATE_INTERVAL_LEVEL):
    """
    Add posterior state summaries and equal-tail bands.

    Convention used here:

        theta_j,t | D_t ~ T_{n_t}(m_j,t, C_jj,t)

    so the equal-tail 90% Student-t band is:

        m_j,t +/- t_{0.95,n_t} sqrt(C_jj,t)

    This does NOT multiply by s_t, matching the existing code convention.
    """

    qcrit = state_interval_critical_value(level=level, nt=nt)

    diagC = np.clip(np.diag(Ct), EPS, np.inf)
    scales = np.sqrt(diagC)

    row["state_interval_level"] = float(level)
    row["state_interval_df"] = (
        float(nt)
        if nt is not None and np.isfinite(float(nt))
        else np.nan
    )
    row["state_interval_qcrit"] = float(qcrit)

    for i in range(len(mt)):
        k = i + 1

        row[f"theta{k}"] = float(mt[i])
        row[f"theta{k}_lo90"] = float(mt[i] - qcrit * scales[i])
        row[f"theta{k}_hi90"] = float(mt[i] + qcrit * scales[i])
        row[f"theta{k}_scale"] = float(scales[i])

    # Fill absent theta columns with NaN up to 4 states
    for k in range(len(mt) + 1, 5):
        row[f"theta{k}"] = np.nan
        row[f"theta{k}_lo90"] = np.nan
        row[f"theta{k}_hi90"] = np.nan
        row[f"theta{k}_scale"] = np.nan

    return row


def centered_ylim_around_one(
    df,
    cols=("theta2", "theta2_lo90", "theta2_hi90"),
    center=1.0,
    min_half_width=0.002,
    pad_frac=0.20,
):
    """
    Build a y-axis range centered around 1 for the AR(1) / lag-1 log-price coefficient.
    """

    vals = []

    for col in cols:
        if col in df.columns:
            arr = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size > 0:
                vals.append(arr)

    if len(vals) == 0:
        return center - min_half_width, center + min_half_width

    vals = np.concatenate(vals)

    if vals.size == 0:
        return center - min_half_width, center + min_half_width

    max_dev = np.nanmax(np.abs(vals - center))

    if not np.isfinite(max_dev):
        max_dev = min_half_width

    half_width = max(min_half_width, max_dev * (1.0 + pad_frac))

    return center - half_width, center + half_width


# ============================================================
# Data loader
# ============================================================

def load_data_for_ticker(ticker):
    download_start = (
        pd.to_datetime(START_DATE) - pd.Timedelta(days=14)
    ).strftime("%Y-%m-%d")

    raw = yf.download(
        ticker,
        start=download_start,
        end=END_DATE,
        interval="1d",
        auto_adjust=AUTO_ADJUST,
        progress=False,
    )

    if raw.empty:
        raise ValueError(f"No Yahoo Finance data downloaded for {ticker}.")

    if isinstance(raw.columns, pd.MultiIndex):
        if ticker in raw.columns.get_level_values(0):
            raw = raw.xs(ticker, axis=1, level=0)
        elif ticker in raw.columns.get_level_values(-1):
            raw = raw.xs(ticker, axis=1, level=-1)
        else:
            raw.columns = raw.columns.get_level_values(0)

    raw.columns = [str(c).title() for c in raw.columns]

    required = ["Open", "High", "Low", "Close"]
    missing = [c for c in required if c not in raw.columns]

    if missing:
        raise ValueError(f"{ticker}: missing columns {missing}")

    px = raw[required].copy()
    px = px.apply(pd.to_numeric, errors="coerce")
    px = px.dropna()

    px = px[
        (px["Open"] > 0)
        & (px["High"] > 0)
        & (px["Low"] > 0)
        & (px["Close"] > 0)
    ].copy()

    px.index = pd.to_datetime(px.index).tz_localize(None)

    log_O = np.log(px["Open"])
    log_H = np.log(px["High"])
    log_L = np.log(px["Low"])
    log_C = np.log(px["Close"])
    log_C_prev = log_C.shift(1)

    # Rogers--Satchell variance:
    #
    #   z_t = log(H/C)log(H/O) + log(L/C)log(L/O)
    #
    # Equivalent algebraically to:
    #   (h-o)(h-c) + (o-l)(c-l)
    rs_var = (
        (log_H - log_O) * (log_H - log_C)
        +
        (log_O - log_L) * (log_C - log_L)
    )

    overnight_var = (log_O - log_C_prev) ** 2

    if INCLUDE_OVERNIGHT:
        z = rs_var + overnight_var
        variance_label = "rs_plus_overnight"
    else:
        z = rs_var
        variance_label = "rs_only"

    z = z.clip(lower=EPS)

    if Y_MODE == "log_price":
        y = log_C
    elif Y_MODE == "log_return":
        y = log_C.diff()
    else:
        raise ValueError("Y_MODE must be either 'log_price' or 'log_return'.")

    df = pd.DataFrame(
        {
            "date": px.index,
            "y": y.values,
            "z": z.values,
            "rs_var": rs_var.values,
            "overnight_var": overnight_var.values,
        }
    )

    df = df[df["date"] >= pd.to_datetime(START_DATE)].copy()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["z"] = pd.to_numeric(df["z"], errors="coerce").clip(lower=EPS)

    df["x"] = np.sqrt(df["z"])
    df["y_lag"] = df["y"].shift(1)
    df["x_lag"] = df["x"].shift(1)

    df = df.dropna().set_index("date")

    df.attrs["ticker"] = ticker
    df.attrs["variance_label"] = variance_label

    return df


# ============================================================
# Log predictive densities
# ============================================================

def logpdf_z_mike(z, st, alpha, nt):
    z = max(float(z), EPS)
    st = max(float(st), EPS)
    nt = max(float(nt), EPS)
    alpha = max(float(alpha), EPS)

    return float(
        fisher_f.logpdf(
            z / st,
            dfn=alpha,
            dfd=nt,
        )
        - math.log(st)
    )


def logpdf_y_mike(et, qt, nt):
    qt = max(float(qt), EPS)
    nt = max(float(nt), EPS)
    rqt = math.sqrt(qt)

    return float(
        student_t.logpdf(
            float(et) / rqt,
            df=nt,
            loc=0.0,
            scale=1.0,
        )
        - math.log(rqt)
    )


def initial_scale_from_data(data):
    init_window = min(20, len(data))
    st = float(data["z"].iloc[:init_window].median())
    return max(st, EPS)


# ============================================================
# SV-DLM recursion: fixed beta
# ============================================================

def fit_svdlm_fixed(data, beta=SV_BETA, delta=DELTA):
    if SV_USES_LAGGED_RV:
        mt = np.array([0.0, 1.0, 0.0], dtype=float)
        Ct = np.diag([0.10, 0.01, 0.05]).astype(float)
        model_label = "SV"
    else:
        mt = np.array([0.0, 1.0], dtype=float)
        Ct = np.diag([0.10, 0.01]).astype(float)
        model_label = "SV_AR"

    st = initial_scale_from_data(data)
    nt = float(N0)

    rows = []

    for t_matlab, (date, r) in enumerate(data.iterrows(), start=2):
        y_t = float(r["y"])
        y_lag = float(r["y_lag"])
        x_lag = float(r["x_lag"])
        x_t = float(r["x"])

        if SV_USES_LAGGED_RV:
            Ft = np.array([1.0, y_lag, x_lag], dtype=float)
        else:
            Ft = np.array([1.0, y_lag], dtype=float)

        scored = is_scored_date(date)

        # Prior evolution
        nt = beta * nt
        at = mt.copy()
        Rt = Ct / delta

        # Forecast
        At_raw = Rt @ Ft
        qt = float(st + Ft @ At_raw)
        qt = max(qt, EPS)

        At = At_raw / qt
        ft = float(Ft @ at)
        et = y_t - ft

        if scored:
            log_py = logpdf_y_mike(et=et, qt=qt, nt=nt)
        else:
            log_py = 0.0

        # Posterior update on y
        mt = at + At * et
        Ct = Rt - qt * np.outer(At, At)
        Ct = stabilize(Ct)

        rt_y = (nt + et * et / qt) / (nt + 1.0)
        st = max(float(rt_y * st), EPS)
        nt = nt + 1.0

        row = {
            "date": date,
            "model": model_label,
            "t_matlab": t_matlab,
            "scored": scored,
            "beta": beta,
            "delta": delta,
            "alpha": np.nan,
            "y_t": y_t,
            "x_t": x_t,
            "x_lag": x_lag,
            "log_py": log_py,
            "log_pz": np.nan,
            "log_joint": log_py,
            "forecast_mean_y": ft,
            "forecast_scale2_y_qt": qt,
            "forecast_error_y": et,
            "st_post": st,
            "nt_post": nt,
        }

        row = add_state_summaries(
            row,
            mt,
            Ct,
            st=st,
            nt=nt,
            level=STATE_INTERVAL_LEVEL,
        )
        rows.append(row)

    return pd.DataFrame(rows).set_index("date")


# ============================================================
# RV-DLM / RVL-DLM recursion: fixed alpha, fixed beta
# ============================================================

def fit_rv_model_fixed(
    data,
    model,
    beta=RV_BETA,
    delta=DELTA,
    alpha=ALPHA,
):
    model = model.upper().strip()

    if model not in ["RV", "RVL"]:
        raise ValueError("model must be either 'RV' or 'RVL'.")

    use_current_x = model == "RVL"

    # Code ordering:
    #
    #   RV:  F_t = (1, y_{t-1}, x_{t-1})
    #   RVL: F_t = (1, y_{t-1}, x_{t-1}, x_t)
    #
    # Therefore:
    #   theta3 = lagged RV coefficient
    #   theta4 = contemporaneous RV coefficient
    if use_current_x:
        mt = np.array([0.0, 1.0, 0.0, 0.0], dtype=float)
        Ct = np.diag([0.10, 0.01, 0.05, 0.05]).astype(float)
    else:
        mt = np.array([0.0, 1.0, 0.0], dtype=float)
        Ct = np.diag([0.10, 0.01, 0.05]).astype(float)

    st = initial_scale_from_data(data)
    nt = float(N0)

    rows = []

    for t_matlab, (date, r) in enumerate(data.iterrows(), start=2):
        y_t = float(r["y"])
        z_t = max(float(r["z"]), EPS)
        x_t = float(r["x"])
        y_lag = float(r["y_lag"])
        x_lag = float(r["x_lag"])

        if use_current_x:
            Ft = np.array([1.0, y_lag, x_lag, x_t], dtype=float)
        else:
            Ft = np.array([1.0, y_lag, x_lag], dtype=float)

        scored = is_scored_date(date)

        # Prior evolution
        nt_prior = beta * nt
        at = mt.copy()
        Rt = Ct / delta

        At_raw = Rt @ Ft
        ft = float(Ft @ at)
        et = y_t - ft
        F_R_F = float(Ft @ At_raw)

        # Prior price scale before z update
        qt_prior = max(float(st + F_R_F), EPS)

        # Score z using prior p(z_t | D_{t-1})
        if scored:
            log_pz = logpdf_z_mike(z=z_t, st=st, alpha=alpha, nt=nt_prior)
        else:
            log_pz = 0.0

        # Update on z first
        rt_z = (nt_prior + alpha * z_t / st) / (nt_prior + alpha)
        st_tilde = max(float(rt_z * st), EPS)
        nt_tilde = nt_prior + alpha

        # Conditional price score:
        #   p(y_t | z_t, D_{t-1})
        if RV_PRICE_SCORE_AFTER_Z:
            qt_y = max(float(st_tilde + F_R_F), EPS)
            nt_y = nt_tilde
        else:
            qt_y = qt_prior
            nt_y = nt_prior

        if scored:
            log_py = logpdf_y_mike(et=et, qt=qt_y, nt=nt_y)
        else:
            log_py = 0.0

        # Update on y second
        At = At_raw / qt_y

        mt = at + At * et
        Ct = Rt - qt_y * np.outer(At, At)
        Ct = stabilize(Ct)

        rt_y = (nt_tilde + et * et / qt_y) / (nt_tilde + 1.0)
        st = max(float(rt_y * st_tilde), EPS)
        nt = nt_tilde + 1.0

        row = {
            "date": date,
            "model": model,
            "t_matlab": t_matlab,
            "scored": scored,
            "beta": beta,
            "delta": delta,
            "alpha": alpha,
            "y_t": y_t,
            "z_t": z_t,
            "x_t": x_t,
            "x_lag": x_lag,
            "log_py": log_py,
            "log_pz": log_pz,
            "log_joint": log_py + log_pz,
            "forecast_mean_y": ft,
            "forecast_scale2_y_qt": qt_y,
            "forecast_error_y": et,
            "st_post": st,
            "nt_post": nt,
        }

        row = add_state_summaries(
            row,
            mt,
            Ct,
            st=st,
            nt=nt,
            level=STATE_INTERVAL_LEVEL,
        )
        rows.append(row)

    return pd.DataFrame(rows).set_index("date")


# ============================================================
# Run fixed-hyperparameter analysis
# ============================================================

def run_all_fixed():
    data_by_ticker = {}
    fits = {}
    summary_rows = []

    for ticker in TICKERS:
        print("\n" + "=" * 80)
        print(f"Running fixed-hyperparameter filters for {ticker}")
        print("=" * 80)

        data = load_data_for_ticker(ticker)
        data_by_ticker[ticker] = data

        sv = fit_svdlm_fixed(data, beta=SV_BETA, delta=DELTA)

        rv = fit_rv_model_fixed(
            data,
            model="RV",
            beta=RV_BETA,
            delta=DELTA,
            alpha=ALPHA,
        )

        rvl = fit_rv_model_fixed(
            data,
            model="RVL",
            beta=RV_BETA,
            delta=DELTA,
            alpha=ALPHA,
        )

        fits[ticker] = {
            "SV": sv,
            "RV": rv,
            "RVL": rvl,
        }

        for model_name, fit in fits[ticker].items():
            scored = fit["scored"].astype(bool)

            summary_rows.append(
                {
                    "ticker": ticker,
                    "model": model_name,
                    "alpha": ALPHA if model_name in ["RV", "RVL"] else np.nan,
                    "beta": SV_BETA if model_name == "SV" else RV_BETA,
                    "delta": DELTA,
                    "log_py_sum": float(fit.loc[scored, "log_py"].sum()),
                    "log_pz_sum": (
                        np.nan
                        if model_name == "SV"
                        else float(fit.loc[scored, "log_pz"].sum())
                    ),
                    "log_joint_sum": float(fit.loc[scored, "log_joint"].sum()),
                    "n_scored": int(scored.sum()),
                    "first_scored_date": fit.loc[scored].index.min().date(),
                    "last_scored_date": fit.loc[scored].index.max().date(),
                }
            )

            print(
                f"{model_name}: "
                f"log_py={summary_rows[-1]['log_py_sum']:.3f}, "
                f"log_pz={summary_rows[-1]['log_pz_sum']}, "
                f"log_joint={summary_rows[-1]['log_joint_sum']:.3f}"
            )

    summary = pd.DataFrame(summary_rows)

    return data_by_ticker, fits, summary


data_by_ticker, fits, score_summary = run_all_fixed()


# ============================================================
# Build comparison objects
# ============================================================

def build_comparisons(fits):
    comparisons = {}

    for ticker in TICKERS:
        sv = fits[ticker]["SV"]
        rv = fits[ticker]["RV"]
        rvl = fits[ticker]["RVL"]

        idx = sv.index.intersection(rv.index).intersection(rvl.index)

        sv = sv.loc[idx].copy()
        rv = rv.loc[idx].copy()
        rvl = rvl.loc[idx].copy()

        cmp = pd.DataFrame(index=idx)
        cmp["scored"] = sv["scored"].astype(bool)

        cmp["d_py_RV_minus_SV"] = rv["log_py"] - sv["log_py"]
        cmp["d_py_RVL_minus_RV"] = rvl["log_py"] - rv["log_py"]
        cmp["d_py_RVL_minus_SV"] = rvl["log_py"] - sv["log_py"]

        cmp["cum_py_RV_minus_SV"] = cmp["d_py_RV_minus_SV"].cumsum()
        cmp["cum_py_RVL_minus_RV"] = cmp["d_py_RVL_minus_RV"].cumsum()
        cmp["cum_py_RVL_minus_SV"] = cmp["d_py_RVL_minus_SV"].cumsum()

        comparisons[ticker] = cmp.loc[cmp["scored"]].copy()

    return comparisons


comparisons = build_comparisons(fits)


# ============================================================
# Generic plotting helpers
# ============================================================

def plot_series_frame(
    x,
    y,
    filename,
    ylabel="Log BF",
    hline=0.0,
    zoomed=False,
    ylim=None,
    linewidth=PAPER_LINE_WIDTH,
):
    fig, ax = plt.subplots(figsize=PAPER_FIGSIZE)

    line_kwargs = {}
    if GREYSCALE_FIGURES:
        line_kwargs["color"] = "black"

    ax.plot(
        x,
        y,
        linewidth=linewidth,
        **line_kwargs,
    )

    if hline is not None:
        ax.axhline(hline, color="black", linestyle="--", linewidth=1.0)

    paper_axis(ax, ylabel=ylabel, zoomed=zoomed, ylim=ylim)

    fig.tight_layout()
    save_paper_pdf(fig, filename)

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def plot_band_frame(
    df,
    mean_col,
    lower_col,
    upper_col,
    filename,
    ylabel="Parameter",
    hline=0.0,
    zoomed=False,
    ylim=None,
):
    fig, ax = plt.subplots(figsize=PAPER_FIGSIZE)

    line_kwargs = {}
    band_kwargs = {}

    if GREYSCALE_FIGURES:
        line_kwargs["color"] = "black"
        band_kwargs["color"] = "0.75"

    ax.plot(
        df.index,
        df[mean_col].astype(float).values,
        linewidth=PAPER_LINE_WIDTH,
        **line_kwargs,
    )

    ax.fill_between(
        df.index,
        df[lower_col].astype(float).values,
        df[upper_col].astype(float).values,
        alpha=PAPER_BAND_ALPHA,
        linewidth=0,
        **band_kwargs,
    )

    if hline is not None:
        ax.axhline(hline, color="black", linestyle="--", linewidth=1.0)

    paper_axis(ax, ylabel=ylabel, zoomed=zoomed, ylim=ylim)

    fig.tight_layout()
    save_paper_pdf(fig, filename)

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


# ============================================================
# Figure A: SPY latent SD comparison
# FigureA1 = price-series discount volatility using daily log changes
# FigureA2 = RV-only dynamic gamma analysis of z_t
# ============================================================

def posterior_sd_summary_from_gamma_precision(n_value, s_value, interval=0.90):
    """
    phi | D ~ Gamma(n/2, n*s/2), rate parameterization.
    SD = sqrt(v) = 1 / sqrt(phi).
    """
    n_value = max(float(n_value), EPS)
    s_value = max(float(s_value), EPS)

    shape = n_value / 2.0
    rate = n_value * s_value / 2.0
    scale = 1.0 / rate

    tail = (1.0 - interval) / 2.0

    # Larger phi means smaller SD.
    phi_for_sd_lower = max(
        float(gamma_dist.ppf(1.0 - tail, a=shape, scale=scale)),
        EPS,
    )

    phi_for_sd_median = max(
        float(gamma_dist.ppf(0.5, a=shape, scale=scale)),
        EPS,
    )

    phi_for_sd_upper = max(
        float(gamma_dist.ppf(tail, a=shape, scale=scale)),
        EPS,
    )

    return {
        "sd_lower": float(1.0 / np.sqrt(phi_for_sd_lower)),
        "sd_median": float(1.0 / np.sqrt(phi_for_sd_median)),
        "sd_upper": float(1.0 / np.sqrt(phi_for_sd_upper)),
    }


def fit_price_only_sd_filter(data, beta=SV_BETA, n0=N0, interval=0.90):
    """
    FigureA1 helper:
    discount volatility analysis of daily log price changes only.
    """

    y = data["y"].astype(float)
    r = y.diff().dropna()

    init_window = min(20, len(r))
    s_prev = float(max(r.iloc[:init_window].var(), EPS))
    n_prev = float(n0)

    rows = []

    for date, r_t in r.items():
        r_t = float(r_t)

        n_pred = beta * n_prev

        n_post = n_pred + 1.0
        s_post = (n_pred * s_prev + r_t * r_t) / n_post
        s_post = max(float(s_post), EPS)

        sd = posterior_sd_summary_from_gamma_precision(
            n_value=n_post,
            s_value=s_post,
            interval=interval,
        )

        rows.append(
            {
                "date": date,
                "obs_sd": abs(r_t),
                "n_post": n_post,
                "s_post": s_post,
                **sd,
            }
        )

        n_prev = n_post
        s_prev = s_post

    return pd.DataFrame(rows).set_index("date")


def fit_rv_only_sd_filter(data, beta=RV_BETA, alpha=ALPHA, n0=N0, interval=0.90):
    """
    FigureA2 helper:
    dynamic gamma RV-only analysis of realized variance z_t.
    """

    z = data["z"].astype(float).clip(lower=EPS)

    init_window = min(20, len(z))
    s_prev = float(max(z.iloc[:init_window].median(), EPS))
    n_prev = float(n0)

    rows = []

    for date, z_t in z.items():
        z_t = max(float(z_t), EPS)

        n_pred = beta * n_prev

        n_post = n_pred + alpha
        s_post = (n_pred * s_prev + alpha * z_t) / n_post
        s_post = max(float(s_post), EPS)

        sd = posterior_sd_summary_from_gamma_precision(
            n_value=n_post,
            s_value=s_post,
            interval=interval,
        )

        rows.append(
            {
                "date": date,
                "obs_sd": np.sqrt(z_t),
                "n_post": n_post,
                "s_post": s_post,
                **sd,
            }
        )

        n_prev = n_post
        s_prev = s_post

    return pd.DataFrame(rows).set_index("date")


def plot_sd_band_frame(df, filename, ylim=None):
    """
    Figure A plotting helper.

    Greyscale version:
      1. raw observed volatility proxy as a light grey noisy line
      2. posterior 90% SD band in grey
      3. posterior median SD in black
    """

    fig, ax = plt.subplots(figsize=PAPER_FIGSIZE)

    if GREYSCALE_FIGURES:
        raw_color = "0.65"
        band_color = "0.80"
        median_color = "black"
    else:
        raw_color = "C0"
        band_color = "C0"
        median_color = "C0"

    # Raw underlying volatility proxy:
    #   FigureA1: |r_t| = sqrt(r_t^2)
    #   FigureA2: sqrt(RS_t)
    ax.plot(
        df.index,
        df["obs_sd"].astype(float).values,
        linewidth=0.7,
        alpha=0.35,
        color=raw_color,
        zorder=1,
    )

    # Posterior 90% SD band
    ax.fill_between(
        df.index,
        df["sd_lower"].astype(float).values,
        df["sd_upper"].astype(float).values,
        alpha=PAPER_BAND_ALPHA,
        linewidth=0,
        color=band_color,
        zorder=2,
    )

    # Posterior median SD
    ax.plot(
        df.index,
        df["sd_median"].astype(float).values,
        linewidth=PAPER_LINE_WIDTH,
        color=median_color,
        zorder=3,
    )

    paper_axis(ax, ylabel="SD", zoomed=False, ylim=ylim)

    fig.tight_layout()
    save_paper_pdf(fig, filename)

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def save_figure_A(data_by_ticker):
    data = data_by_ticker[FIGURE_A_TICKER]

    price_sd = fit_price_only_sd_filter(
        data,
        beta=SV_BETA,
        n0=N0,
        interval=0.90,
    )

    rv_sd = fit_rv_only_sd_filter(
        data,
        beta=RV_BETA,
        alpha=ALPHA,
        n0=N0,
        interval=0.90,
    )

    price_sd = price_sd.loc[
        (price_sd.index >= pd.Timestamp(FIGURE_A_START))
        & (price_sd.index <= pd.Timestamp(FIGURE_A_END))
    ].copy()

    rv_sd = rv_sd.loc[
        (rv_sd.index >= pd.Timestamp(FIGURE_A_START))
        & (rv_sd.index <= pd.Timestamp(FIGURE_A_END))
    ].copy()

    # Fixed shared y-axis limit for FigureA1 and FigureA2.
    # This clips large raw spikes above the display range and keeps
    # the two panels directly comparable.
    ylim = (0.0, FIGURE_A_YMAX)

    plot_sd_band_frame(price_sd, "FigureA1.pdf", ylim=ylim)
    plot_sd_band_frame(rv_sd, "FigureA2.pdf", ylim=ylim)


# ============================================================
# Figure B: all symbols cumulative log BF, separate frames
# FigureB1 = RV vs SV
# FigureB2 = RVL vs RV
# FigureB3 = RVL vs SV
#
# Greyscale update:
#   SPY is dashed black.
#   XLK is dotted black; this is the formerly purple line.
#   Other ETFs use solid greyscale lines.
# ============================================================

def figure_b_style_for_ticker(ticker, grey_by_ticker):
    """
    Print-safe styles for Figure B.

    SPY: dashed black.
    XLK: dotted black, replacing the old purple line.
    Others: solid greyscale spectrum.
    """

    if GREYSCALE_FIGURES:
        if ticker == "SPY":
            return {
                "color": "black",
                "linestyle": "--",
                "linewidth": 2.1,
                "alpha": 0.95,
            }

        if ticker == FIGURE_B_SPECIAL_TICKER:
            return {
                "color": "black",
                "linestyle": ":",
                "linewidth": 2.3,
                "alpha": 0.95,
            }

        return {
            "color": grey_by_ticker[ticker],
            "linestyle": "-",
            "linewidth": 1.4,
            "alpha": 0.90,
        }

    # Original colour version
    if ticker == "SPY":
        return {
            "color": "lightskyblue",
            "linestyle": "-",
            "linewidth": 2.0,
            "alpha": 0.95,
        }

    return {
        "linestyle": "-",
        "linewidth": 1.4,
        "alpha": 0.90,
    }


def save_figure_B(comparisons):
    specs = [
        ("cum_py_RV_minus_SV", "FigureB1.pdf"),
        ("cum_py_RVL_minus_RV", "FigureB2.pdf"),
        ("cum_py_RVL_minus_SV", "FigureB3.pdf"),
    ]

    other_tickers = [
        t for t in TICKERS
        if t not in ["SPY", FIGURE_B_SPECIAL_TICKER]
    ]

    grey_values = np.linspace(0.75, 0.25, len(other_tickers))
    grey_by_ticker = {
        ticker: f"{grey:.2f}"
        for ticker, grey in zip(other_tickers, grey_values)
    }

    for col, filename in specs:
        fig, ax = plt.subplots(figsize=PAPER_FIGSIZE)

        for ticker in TICKERS:
            df = comparisons[ticker]
            style = figure_b_style_for_ticker(ticker, grey_by_ticker)

            ax.plot(
                df.index,
                df[col],
                label=ticker,
                **style,
            )

        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
        paper_axis(ax, ylabel="Log BF", zoomed=False)

        fig.tight_layout()
        save_paper_pdf(fig, filename)

        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close(fig)


# ============================================================
# Figure C: selected ETFs cumulative log BF, separate frames
# FigureC1 = XLB
# FigureC2 = XLU
# FigureC3 = XLE
#
# Greyscale update:
#   Solid: RV-DLM minus SV-DLM
#   Dashed: RVL-DLM minus SV-DLM
#   Dotted: RVL-DLM minus RV-DLM
# ============================================================

def save_figure_C(comparisons, selected=SELECTED_TICKERS):
    series_specs = [
        ("cum_py_RV_minus_SV", "-", "RV-DLM minus SV-DLM"),
        ("cum_py_RVL_minus_SV", "--", "RVL-DLM minus SV-DLM"),
        ("cum_py_RVL_minus_RV", ":", "RVL-DLM minus RV-DLM"),
    ]

    for i, ticker in enumerate(selected, start=1):
        df = comparisons[ticker]

        fig, ax = plt.subplots(figsize=PAPER_FIGSIZE)

        for col, linestyle, label in series_specs:
            plot_kwargs = {
                "linewidth": PAPER_LINE_WIDTH,
                "linestyle": linestyle,
                "label": label,
            }

            if GREYSCALE_FIGURES:
                plot_kwargs["color"] = "black"

            ax.plot(
                df.index,
                df[col],
                **plot_kwargs,
            )

        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
        paper_axis(ax, ylabel="Log BF", zoomed=False)

        fig.tight_layout()
        save_paper_pdf(fig, f"FigureC{i}.pdf")

        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close(fig)


# ============================================================
# Figure D: RVL coefficients, full post-2010, separate frames
#
# D1 = XLB current x_t coeff
# D2 = XLB lagged x_{t-1} coeff
# D3 = XLU current x_t coeff
# D4 = XLU lagged x_{t-1} coeff
# D5 = XLE current x_t coeff
# D6 = XLE lagged x_{t-1} coeff
# ============================================================

def save_figure_D(fits, selected=SELECTED_TICKERS):
    k = 1

    for ticker in selected:
        df = fits[ticker]["RVL"].copy()
        df = df[df["scored"].astype(bool)].copy()

        # Current RV coefficient: theta4 in this code ordering
        plot_band_frame(
            df=df,
            mean_col="theta4",
            lower_col="theta4_lo90",
            upper_col="theta4_hi90",
            filename=f"FigureD{k}.pdf",
            ylabel="Parameter",
            hline=0.0,
            zoomed=False,
            ylim=(-0.8, 0.0),
        )
        k += 1

        # Lagged RV coefficient: theta3 in this code ordering
        plot_band_frame(
            df=df,
            mean_col="theta3",
            lower_col="theta3_lo90",
            upper_col="theta3_hi90",
            filename=f"FigureD{k}.pdf",
            ylabel="Parameter",
            hline=0.0,
            zoomed=False,
            ylim=(-0.2, 0.5),
        )
        k += 1


# ============================================================
# Figure E: RVL coefficients, 2023 zoom, separate frames
#
# E1 = XLB current x_t coeff
# E2 = XLB lagged x_{t-1} coeff
# E3 = XLU current x_t coeff
# E4 = XLU lagged x_{t-1} coeff
# E5 = XLE current x_t coeff
# E6 = XLE lagged x_{t-1} coeff
# ============================================================

def save_figure_E(fits, selected=SELECTED_TICKERS):
    k = 1
    zstart = pd.Timestamp(FIGURE_ZOOM_START)
    zend = pd.Timestamp(FIGURE_ZOOM_END)

    for ticker in selected:
        df = fits[ticker]["RVL"].copy()
        df = df[
            (df.index >= zstart)
            & (df.index <= zend)
            & (df["scored"].astype(bool))
        ].copy()

        # Current RV coefficient: theta4
        plot_band_frame(
            df=df,
            mean_col="theta4",
            lower_col="theta4_lo90",
            upper_col="theta4_hi90",
            filename=f"FigureE{k}.pdf",
            ylabel="Parameter",
            hline=0.0,
            zoomed=True,
            ylim=(-0.8, 0.0),
        )
        k += 1

        # Lagged RV coefficient: theta3
        plot_band_frame(
            df=df,
            mean_col="theta3",
            lower_col="theta3_lo90",
            upper_col="theta3_hi90",
            filename=f"FigureE{k}.pdf",
            ylabel="Parameter",
            hline=0.0,
            zoomed=True,
            ylim=(-0.2, 0.5),
        )
        k += 1


# ============================================================
# Figure F: XLE all RVL state coefficients, separate frames
#
# F1 = theta1 local intercept
# F2 = theta2 AR(1) log price coefficient, centered around 1
# F3 = theta3 lagged RV coefficient
# F4 = theta4 current RV coefficient
# ============================================================

def save_figure_F(fits, ticker="XLE"):
    df = fits[ticker]["RVL"].copy()
    df = df[df["scored"].astype(bool)].copy()

    theta2_ylim = centered_ylim_around_one(
        df,
        cols=("theta2", "theta2_lo90", "theta2_hi90"),
        center=1.0,
        min_half_width=0.002,
        pad_frac=0.20,
    )

    # F1: local intercept
    plot_band_frame(
        df=df,
        mean_col="theta1",
        lower_col="theta1_lo90",
        upper_col="theta1_hi90",
        filename="FigureF1.pdf",
        ylabel="Parameter",
        hline=0.0,
        zoomed=False,
        ylim=None,
    )

    # F2: AR(1) coefficient centered around 1
    plot_band_frame(
        df=df,
        mean_col="theta2",
        lower_col="theta2_lo90",
        upper_col="theta2_hi90",
        filename="FigureF2.pdf",
        ylabel="Parameter",
        hline=1.0,
        zoomed=False,
        ylim=theta2_ylim,
    )

    # F3: lagged RV coefficient
    plot_band_frame(
        df=df,
        mean_col="theta3",
        lower_col="theta3_lo90",
        upper_col="theta3_hi90",
        filename="FigureF3.pdf",
        ylabel="Parameter",
        hline=0.0,
        zoomed=False,
        ylim=(-0.2, 0.5),
    )

    # F4: current RV coefficient
    plot_band_frame(
        df=df,
        mean_col="theta4",
        lower_col="theta4_lo90",
        upper_col="theta4_hi90",
        filename="FigureF4.pdf",
        ylabel="Parameter",
        hline=0.0,
        zoomed=False,
        ylim=(-0.8, 0.0),
    )


# ============================================================
# Figure G: price-scale contemporaneous RV effect
#
# G1 = XLB
# G2 = XLU
# G3 = XLE
# ============================================================

def save_figure_G(fits, selected=SELECTED_TICKERS):
    for i, ticker in enumerate(selected, start=1):
        df = fits[ticker]["RVL"].copy()
        df = df[df["scored"].astype(bool)].copy()

        effect = np.exp(
            df["theta4"].astype(float).values
            *
            df["x_t"].astype(float).values
        )

        fig, ax = plt.subplots(figsize=PAPER_FIGSIZE)

        plot_kwargs = {
            "linewidth": PAPER_LINE_WIDTH,
        }

        if GREYSCALE_FIGURES:
            plot_kwargs["color"] = "black"

        ax.plot(
            df.index,
            effect,
            **plot_kwargs,
        )

        ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)

        paper_axis(ax, ylabel="Effect", zoomed=False)

        fig.tight_layout()
        save_paper_pdf(fig, f"FigureG{i}.pdf")

        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close(fig)


# ============================================================
# Save all figures exactly as referenced in LaTeX
# ============================================================

save_figure_A(data_by_ticker)
save_figure_B(comparisons)
save_figure_C(comparisons)
save_figure_D(fits)
save_figure_E(fits)
save_figure_F(fits, ticker="XLE")
save_figure_G(fits)

print("\nAll LaTeX-ready figure PDFs were written to:", FIGURE_DIR.resolve())

print("\nGREYSCALE / PRINT-SAFE FIGURE NOTES")
print("Figure B: dashed line denotes SPY.")
print(f"Figure B: dotted line denotes {FIGURE_B_SPECIAL_TICKER}, the formerly purple ETF.")
print("Figure C: solid = RV-DLM minus SV-DLM.")
print("Figure C: dashed = RVL-DLM minus SV-DLM.")
print("Figure C: dotted = RVL-DLM minus RV-DLM.")


# ============================================================
# Optional: save tables and filters
# ============================================================

if SAVE_TABLES:
    score_summary.to_csv(
        FIGURE_DIR / "fixed_hyperparameter_score_summary.csv",
        index=False,
    )

    pairwise_rows = []

    for ticker in TICKERS:
        cmp = comparisons[ticker]

        pairwise_rows.append(
            {
                "ticker": ticker,
                "comparison": "RV-DLM minus SV-DLM",
                "score_basis": "log p(y)",
                "final_logbf": cmp["cum_py_RV_minus_SV"].iloc[-1],
                "alpha": ALPHA,
                "rv_beta": RV_BETA,
                "sv_beta": SV_BETA,
                "delta": DELTA,
            }
        )

        pairwise_rows.append(
            {
                "ticker": ticker,
                "comparison": "RVL-DLM minus RV-DLM",
                "score_basis": "log p(y)",
                "final_logbf": cmp["cum_py_RVL_minus_RV"].iloc[-1],
                "alpha": ALPHA,
                "rv_beta": RV_BETA,
                "sv_beta": SV_BETA,
                "delta": DELTA,
            }
        )

        pairwise_rows.append(
            {
                "ticker": ticker,
                "comparison": "RVL-DLM minus SV-DLM",
                "score_basis": "log p(y)",
                "final_logbf": cmp["cum_py_RVL_minus_SV"].iloc[-1],
                "alpha": ALPHA,
                "rv_beta": RV_BETA,
                "sv_beta": SV_BETA,
                "delta": DELTA,
            }
        )

    pairwise_summary = pd.DataFrame(pairwise_rows)

    pairwise_summary.to_csv(
        FIGURE_DIR / "fixed_hyperparameter_pairwise_logbf_summary.csv",
        index=False,
    )

    print("\nPAIRWISE LOG BF SUMMARY")
    print(
        pairwise_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:,.4f}",
        )
    )

    print("\nRVL-DLM MINUS RV-DLM FINAL LOG BF, SORTED")
    print(
        pairwise_summary[
            pairwise_summary["comparison"] == "RVL-DLM minus RV-DLM"
        ]
        .sort_values("final_logbf", ascending=False)
        .to_string(
            index=False,
            float_format=lambda x: f"{x:,.4f}",
        )
    )

    print(
        f"\nFormer purple line in Figure B is: {FIGURE_B_SPECIAL_TICKER}"
    )

if SAVE_FILTERS:
    filter_dir = FIGURE_DIR / "filter_outputs"
    filter_dir.mkdir(parents=True, exist_ok=True)

    for ticker in TICKERS:
        for model_name in ["SV", "RV", "RVL"]:
            fits[ticker][model_name].to_csv(
                filter_dir / f"{ticker.lower()}_{model_name.lower()}_fixed_filter.csv"
            )

    print("\nSaved filters to:", filter_dir.resolve())
