import os
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Soccer Predictor",
    page_icon="⚽",
    layout="wide",
)

LEAGUES = [
    "EPL",
    "La Liga",
    "Ligue 1",
    "Bundesliga",
    "Serie A",
    "Betway Premiership",
    "Turkish Süper Lig",
]

LOG_FILE = "predictions.csv"
MAX_GOALS = 10

st.title("⚽ Soccer Predictor")
st.caption(
    "Multi-league forecasting • Odds analysis • "
    "Backtesting • Paper-trading only"
)


def poisson_pmf(rate, max_goals=MAX_GOALS):
    rate = max(float(rate), 0.01)
    values = [
        math.exp(-rate + k * math.log(rate) - math.lgamma(k + 1))
        for k in range(max_goals + 1)
    ]
    # Include a final overflow bucket for scores above max_goals.
    values.append(max(0.0, 1.0 - sum(values)))
    return np.array(values) / sum(values)


def clean_history(df):
    """Accept common football-data.co.uk CSV column names."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    required = ["HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required historical columns: "
            + ", ".join(missing)
        )

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(
            df["Date"], dayfirst=True, errors="coerce"
        )
    else:
        df["Date"] = pd.NaT

    for col in ["FTHG", "FTAG"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required).copy()
    df = df[
        (df["FTHG"] >= 0)
        & (df["FTAG"] >= 0)
        & (df["FTHG"] <= 20)
        & (df["FTAG"] <= 20)
    ]

    return df.sort_values("Date").reset_index(drop=True)


def team_mean(df, team, venue, scored, prior, strength=5):
    if venue == "home":
        rows = df[df["HomeTeam"] == team]
        values = rows[scored]
    elif venue == "away":
        rows = df[df["AwayTeam"] == team]
        values = rows[scored]
    else:
        values = pd.Series(dtype=float)

    n = len(values)
    return (
        values.sum() + strength * prior
    ) / (n + strength)


def rates_for(df, home, away):
    """Smoothed attack/defence rates; no future matches are used."""
    home_avg = max(df["FTHG"].mean(), 0.2)
    away_avg = max(df["FTAG"].mean(), 0.2)

    home_scored = team_mean(
        df, home, "home", "FTHG", home_avg
    )
    home_conceded = team_mean(
        df, home, "home", "FTAG", away_avg
    )
    away_scored = team_mean(
        df, away, "away", "FTAG", away_avg
    )
    away_conceded = team_mean(
        df, away, "away", "FTHG", home_avg
    )

    # Blend attack and opposition defence with league scoring rates.
    lam_home = (
        home_scored / home_avg
        * away_conceded / home_avg
        * home_avg
    )
    lam_away = (
        away_scored / away_avg
        * home_conceded / away_avg
        * away_avg
    )

    return max(0.1, lam_home), max(0.1, lam_away)


def score_matrix(lam_home, lam_away):
    h = poisson_pmf(lam_home)
    a = poisson_pmf(lam_away)
    return np.outer(h, a)


def market_probabilities(matrix):
    home_win = draw = away_win = 0.0

    for h in range(matrix.shape[0]):
        for a in range(matrix.shape[1]):
            if h > a:
                home_win += matrix[h, a]
            elif h == a:
                draw += matrix[h, a]
            else:
                away_win += matrix[h, a]

    def total_over(line):
        return sum(
            matrix[h, a]
            for h in range(matrix.shape[0])
            for a in range(matrix.shape[1])
            if h + a > line
        )

    def btts():
        return float(matrix[1:, 1:].sum())

    return {
        "Home win": home_win,
        "Draw": draw,
        "Away win": away_win,
        "1X": home_win + draw,
        "X2": draw + away_win,
        "12": home_win + away_win,
        "Over 1.5 goals": total_over(1),
        "Over 2.5 goals": total_over(2),
        "Over 3.5 goals": total_over(3),
        "Under 1.5 goals": 1 - total_over(1),
        "Under 2.5 goals": 1 - total_over(2),
        "Under 3.5 goals": 1 - total_over(3),
        "Both teams to score — Yes": btts(),
        "Both teams to score — No": 1 - btts(),
    }


def best_scores(matrix, count=5):
    items = []
    for h in range(matrix.shape[0] - 1):
        for a in range(matrix.shape[1] - 1):
            items.append((h, a, matrix[h, a]))
    return sorted(items, key=lambda x: x[2], reverse=True)[:count]


def odds_analysis(probability, odds, bankroll, fraction=0.25):
    """Fractional-Kelly guidance; never places a bet."""
    if odds is None or pd.isna(odds) or odds <= 1:
        return None

    p = float(probability)
    q = 1 - p
    ev = p * odds - 1
    full_kelly = max(0.0, (odds * p - 1) / (odds - 1))
    stake_fraction = min(0.02, full_kelly * fraction)

    return {
        "Decimal odds": odds,
        "Model probability": p,
        "Break-even probability": 1 / odds,
        "Expected return per unit": ev,
        "Suggested stake": bankroll * stake_fraction,
        "Potential value": ev > 0,
    }


def backtest(df):
    """Chronological holdout: later matches never train the model."""
    if len(df) < 50:
        return None, None

    split = int(len(df) * 0.8)
    train = df.iloc[:split].copy()
    test = df.iloc[split:].copy()

    if train.empty or test.empty:
        return None, None

    rows = []
    for _, match in test.iterrows():
        home, away = match["HomeTeam"], match["AwayTeam"]
        lh, la = rates_for(train, home, away)
        matrix = score_matrix(lh, la)
        probs = market_probabilities(matrix)

        actual = (
            0 if match["FTHG"] > match["FTAG"]
            else 1 if match["FTHG"] == match["FTAG"]
            else 2
        )
        vector = np.array([
            probs["Home win"], probs["Draw"], probs["Away win"]
        ])
        outcome = np.eye(3)[actual]

        rows.append({
            "actual": actual,
            "predicted": int(np.argmax(vector)),
            "brier": float(np.sum((vector - outcome) ** 2)),
            "log_loss": float(-np.log(max(vector[actual], 1e-12))),
            "confidence": float(vector.max()),
            "correct": int(np.argmax(vector) == actual),
        })

    results = pd.DataFrame(rows)
    summary = {
        "Matches tested": len(results),
        "1X2 accuracy": results["correct"].mean(),
        "Multiclass Brier score": results["brier"].mean(),
        "Log loss": results["log_loss"].mean(),
    }

    results["confidence_band"] = pd.cut(
        results["confidence"],
        bins=[0, .4, .5, .6, .7, .8, 1.000001],
        include_lowest=True,
    )
    calibration = (
        results.groupby("confidence_band", observed=False)
        .agg(
            matches=("correct", "size"),
            mean_confidence=("confidence", "mean"),
            actual_accuracy=("correct", "mean"),
        )
        .reset_index()
    )
    return summary, calibration


with st.sidebar:
    st.header("Configuration")
    league = st.selectbox("League", LEAGUES)
    bankroll = st.number_input(
        "Bankroll (your currency)",
        min_value=0.0,
        value=1000.0,
        step=100.0,
    )
    kelly_fraction = st.select_slider(
        "Kelly fraction",
        options=[0.0, 0.1, 0.25, 0.5],
        value=0.25,
    )
    st.caption(
        "Recommendations only. No automatic betting. "
        "A positive estimated edge is not a guarantee."
    )

st.subheader("1. Load historical results")
history_file = st.file_uploader(
    "Upload this league's historical results CSV",
    type=["csv"],
    key="history",
)
st.caption(
    "Expected columns: HomeTeam, AwayTeam, FTHG, FTAG. "
    "A Date column is recommended."
)

if history_file is None:
    st.info(
        "Upload a historical results CSV to activate predictions. "
        "For fair testing, use several seasons where possible."
    )
    st.stop()

try:
    history = clean_history(pd.read_csv(history_file))
except Exception as exc:
    st.error(f"Could not read historical data: {exc}")
    st.stop()

if len(history) < 50:
    st.warning("Small training dataset: estimates may be unstable.")

tab_predict, tab_test, tab_log = st.tabs([
    "Predictions & odds", "Backtesting", "Prediction log"
])

with tab_predict:
    st.subheader(f"{league}: upcoming fixtures")
    fixture_file = st.file_uploader(
        "Upload upcoming fixtures CSV",
        type=["csv"],
        key="fixtures",
    )

    st.caption(
        "Required: HomeTeam, AwayTeam. Optional decimal odds columns: "
        "odds_home, odds_draw, odds_away. Odds must correspond to "
        "the same full-time 1X2 market."
    )

    if fixture_file is not None:
        fixtures = pd.read_csv(fixture_file)
        fixtures.columns = [str(c).strip() for c in fixtures.columns]
        if not {"HomeTeam", "AwayTeam"}.issubset(fixtures.columns):
            st.error("Fixtures need HomeTeam and AwayTeam columns.")
        else:
            outputs = []
            for _, fixture in fixtures.iterrows():
                home, away = str(fixture["HomeTeam"]), str(fixture["AwayTeam"])
                lh, la = rates_for(history, home, away)
                matrix = score_matrix(lh, la)
                probs = market_probabilities(matrix)

                row = {
                    "League": league,
                    "HomeTeam": home,
                    "AwayTeam": away,
                    "Expected home goals": round(lh, 2),
                    "Expected away goals": round(la, 2),
                    **{k: round(v, 4) for k, v in probs.items()},
                    "Top correct scores": ", ".join(
                        f"{h}-{a} ({p:.1%})"
                        for h, a, p in best_scores(matrix, 3)
                    ),
                }

                for side, market in [
