import json
from pathlib import Path
import sys

import dash
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from dashboard_support import (
    MISSION_PHASES,
    build_alerts,
    build_crew_profiles,
    build_forecast_frame,
    build_radar_scores,
    build_twin_sync,
    enrich_simulation_data,
    get_phase_for_window,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SIMULATION_FILE = DATA_DIR / "simulation_output.json"
BIOGEARS_FILE = DATA_DIR / "biogears_output.csv"
ANALYTICS_FILE = DATA_DIR / "analytics_summary.json"
MONTE_CARLO_FILE = DATA_DIR / "monte_carlo_results.json"

GRAPH_CONFIG = {
    "displaylogo": False,
    "responsive": True,
}

PLOT_FONT = "Trebuchet MS"
METRIC_COLORS = {
    "heart_rate": "#ff6b57",
    "biogears": "#4f7cff",
    "fatigue": "#f4b544",
    "sleep": "#2fb49d",
    "forecast": "#7f8cff",
    "stress": "#c75a9f",
}
EVENT_COLORS = {
    "motion_sickness": "#f16745",
    "sleep_disruption": "#ffb000",
    "sinus_congestion": "#2fb49d",
    "eva_exertion": "#d964a7",
    "exercise_load": "#6c7dff",
    "stress_spike": "#9b72cf",
    "none": "#94a3b8",
}


def generate_dummy_data():
    rng = np.random.default_rng(42)
    event_weights = {
        "motion_sickness": 1.4,
        "sleep_disruption": 1.8,
        "sinus_congestion": 0.9,
        "eva_exertion": 1.9,
    }
    data = []

    for astronaut_id in [1, 2, 3]:
        fatigue_level = rng.uniform(2.0, 3.4)
        baseline_hr = rng.normal(72 + astronaut_id, 2.0)

        for timestamp_h in np.arange(0, 72, 0.5):
            circadian_wave = np.sin((timestamp_h / 24.0) * 2 * np.pi + astronaut_id / 3)
            sleep_wave = np.cos((timestamp_h / 12.0) * 2 * np.pi)
            is_eva_window = 24.0 <= timestamp_h < 26.0
            if is_eva_window:
                event_active = True
                event_type = "eva_exertion"
            else:
                event_active = bool(rng.choice([True, False], p=[0.11, 0.89]))
                event_type = rng.choice(list(event_weights.keys())) if event_active else "none"
            event_weight = event_weights.get(event_type, 0.0)

            fatigue_level = np.clip(
                fatigue_level + rng.normal(0.03 + event_weight * 0.06, 0.28),
                1.0,
                9.0,
            )
            sleep_quality = np.clip(
                6.5 + sleep_wave * 1.1 - event_weight * 0.35 + rng.normal(0, 0.55),
                2.5,
                9.7,
            )
            heart_rate = np.clip(
                baseline_hr
                + circadian_wave * 3.8
                + fatigue_level * 0.9
                + event_weight * 2.4
                + rng.normal(0, 2.6),
                58,
                108,
            )
            biogears_hr = np.clip(heart_rate + rng.normal(0.8 + event_weight * 0.9, 1.5), 58, 112)
            at_risk = fatigue_level >= 7.0 or sleep_quality <= 4.2 or event_type == "eva_exertion"
            respiratory_rate = np.clip(
                13.5 + fatigue_level * 0.7 + event_weight * 1.1 + rng.normal(0, 0.7),
                11.0,
                24.0,
            )
            hrv_proxy = np.clip(
                62 - fatigue_level * 3.4 - event_weight * 4.5 + sleep_quality * 1.3 + rng.normal(0, 2.2),
                18,
                84,
            )
            recovery_time = np.clip(
                fatigue_level * 1.25 + max(0.0, 6.0 - sleep_quality) * 1.4 + event_weight * 1.1,
                2.0,
                20.0,
            )

            data.append(
                {
                    "astronaut_id": astronaut_id,
                    "timestamp_h": float(timestamp_h),
                    "heart_rate_bpm": float(heart_rate),
                    "biogears_hr_bpm": float(biogears_hr),
                    "fatigue_index": float(fatigue_level),
                    "sleep_quality": float(sleep_quality),
                    "event_active": event_active,
                    "event_type": event_type,
                    "at_risk": bool(at_risk),
                    "respiratory_rate_bpm": float(respiratory_rate),
                    "hrv_proxy": float(hrv_proxy),
                    "recovery_time_h": float(recovery_time),
                }
            )

            fatigue_level = np.clip(
                fatigue_level - max(0.0, (sleep_quality - 6.0) * 0.12),
                1.0,
                9.0,
            )

    return pd.DataFrame(data)


def safe_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def astronaut_label(astronaut_id):
    try:
        numeric_value = float(astronaut_id)
        if numeric_value.is_integer():
            return f"Astronaut {int(numeric_value)}"
    except (TypeError, ValueError):
        pass
    return f"Astronaut {astronaut_id}"


def make_slider_marks(start, end, total_marks=5):
    marks = {}
    for raw_value in np.linspace(start, end, total_marks):
        rounded_value = round(float(raw_value), 1)
        marks[rounded_value] = f"{rounded_value:g}h"
    return marks


def load_simulation_data():
    if SIMULATION_FILE.exists():
        try:
            df = pd.read_json(SIMULATION_FILE)
        except ValueError:
            df = pd.read_json(SIMULATION_FILE, lines=True)
    else:
        df = generate_dummy_data()

    required_columns = [
        "astronaut_id",
        "timestamp_h",
        "heart_rate_bpm",
        "fatigue_index",
        "sleep_quality",
    ]
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing required column: {column}")

    for column in ["timestamp_h", "heart_rate_bpm", "fatigue_index", "sleep_quality"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    optional_numeric_columns = [
        "biogears_hr_bpm",
        "respiratory_rate_bpm",
        "hrv_proxy",
        "recovery_time_h",
        "peak_fatigue",
        "risk_prob_fatigue",
    ]
    for column in optional_numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "event_active" not in df.columns:
        df["event_active"] = False
    df["event_active"] = df["event_active"].fillna(False).apply(to_bool)

    if "event_type" not in df.columns:
        df["event_type"] = "none"
    df["event_type"] = df["event_type"].fillna("none").astype(str)

    if "at_risk" not in df.columns:
        df["at_risk"] = (
            (df["fatigue_index"] >= 7.0)
            | (df["sleep_quality"] <= 4.2)
            | (df["event_type"] == "eva_exertion")
        )
    df["at_risk"] = df["at_risk"].fillna(False).apply(to_bool)

    df = df.dropna(subset=required_columns)
    df = df.sort_values(["astronaut_id", "timestamp_h"]).reset_index(drop=True)
    return enrich_simulation_data(df)


def load_biogears_data():
    if not BIOGEARS_FILE.exists():
        return pd.DataFrame()

    bio = pd.read_csv(BIOGEARS_FILE)

    if "timestamp_h" not in bio.columns:
        return pd.DataFrame()

    # Accept BioGears column name and normalize it
    if "heart_rate_bpm" not in bio.columns:
        if "biogears_hr_bpm" in bio.columns:
            bio["heart_rate_bpm"] = bio["biogears_hr_bpm"]
        else:
            return pd.DataFrame()

    bio["timestamp_h"] = pd.to_numeric(bio["timestamp_h"], errors="coerce")
    bio["heart_rate_bpm"] = pd.to_numeric(bio["heart_rate_bpm"], errors="coerce")

    return bio.dropna(subset=["timestamp_h", "heart_rate_bpm"]).reset_index(drop=True)


def load_external_analytics():
    if not ANALYTICS_FILE.exists():
        return {}
    with open(ANALYTICS_FILE, "r", encoding="utf-8") as file:
        raw_payload = json.load(file)

    conclusions = []
    if isinstance(raw_payload.get("conclusions"), list):
        conclusions = [str(item).strip() for item in raw_payload["conclusions"] if str(item).strip()]
    elif isinstance(raw_payload.get("conclusions"), str) and raw_payload["conclusions"].strip():
        conclusions = [raw_payload["conclusions"].strip()]

    for key in ["conclusion_1", "conclusion_2", "conclusion"]:
        value = raw_payload.get(key)
        if isinstance(value, str) and value.strip():
            conclusions.append(value.strip())

    fatigue_load = safe_float(raw_payload.get("fatigue_load"))
    if fatigue_load is None:
        fatigue_load = safe_float(raw_payload.get("cumulative_fatigue_load_mean"))

    return {
        "risk_prob_fatigue": safe_float(raw_payload.get("risk_prob_fatigue")),
        "risk_prob_motion_sickness": safe_float(raw_payload.get("risk_prob_motion_sickness")),
        "mean_recovery_time_h": safe_float(raw_payload.get("mean_recovery_time_h")),
        "fatigue_load": fatigue_load,
        "peak_fatigue": safe_float(raw_payload.get("peak_fatigue")),
        "conclusions": conclusions[:3],
        "raw": raw_payload,
    }


def load_monte_carlo_results():
    if not MONTE_CARLO_FILE.exists():
        return pd.DataFrame()

    try:
        monte_df = pd.read_json(MONTE_CARLO_FILE)
    except ValueError:
        monte_df = pd.read_json(MONTE_CARLO_FILE, lines=True)

    for column in ["peak_fatigue", "recovery_time_h", "risk_prob_fatigue"]:
        if column in monte_df.columns:
            monte_df[column] = pd.to_numeric(monte_df[column], errors="coerce")
    return monte_df.dropna(how="all").reset_index(drop=True)


def filter_for_astronaut(dataframe, astronaut_id):
    return dataframe[dataframe["astronaut_id"].astype(str) == str(astronaut_id)]


def filter_biogears_for_window(astronaut_id, hours):
    if bio_df.empty:
        return pd.DataFrame()

    filtered = bio_df[
        (bio_df["timestamp_h"] >= hours[0]) & (bio_df["timestamp_h"] <= hours[1])
    ]
    if "astronaut_id" in filtered.columns:
        filtered = filtered[filtered["astronaut_id"].astype(str) == str(astronaut_id)]
    return filtered.sort_values("timestamp_h").reset_index(drop=True)


def active_events(dataframe):
    return dataframe[
        dataframe["event_active"].fillna(False) & (dataframe["event_type"] != "none")
    ]


def build_event_segments(filtered_df):
    events_df = active_events(filtered_df).sort_values("timestamp_h").reset_index(drop=True)
    if events_df.empty:
        return pd.DataFrame()

    timestep_h = filtered_df["timestamp_h"].diff().dropna().median()
    if pd.isna(timestep_h) or timestep_h <= 0:
        timestep_h = 0.5

    segments = []
    current = None
    for row in events_df.itertuples(index=False):
        row_time = float(row.timestamp_h)
        row_type = str(row.event_type)

        if current is None:
            current = {"event_type": row_type, "start_h": row_time, "end_h": row_time + timestep_h}
            continue

        contiguous = row_time <= current["end_h"] + (timestep_h * 0.35)
        if row_type == current["event_type"] and contiguous:
            current["end_h"] = row_time + timestep_h
        else:
            segments.append(current)
            current = {"event_type": row_type, "start_h": row_time, "end_h": row_time + timestep_h}

    if current is not None:
        segments.append(current)

    segments_df = pd.DataFrame(segments)
    if segments_df.empty:
        return segments_df
    segments_df["event_label"] = segments_df["event_type"].str.replace("_", " ").str.title()
    return segments_df


def style_chart_figure(fig, title, height=360):
    fig.update_layout(
        title={
            "text": title,
            "x": 0.02,
            "xanchor": "left",
            "font": {"size": 20, "family": PLOT_FONT, "color": "#12233d"},
        },
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="#f7f9fc",
        font={"family": PLOT_FONT, "color": "#1f3559"},
        margin={"l": 52, "r": 28, "t": 74, "b": 48},
        height=height,
        hovermode="x unified",
        legend={
            "orientation": "h",
            "x": 0,
            "xanchor": "left",
            "y": 1.04,
            "yanchor": "bottom",
        },
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(31, 53, 89, 0.08)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(31, 53, 89, 0.08)", zeroline=False)
    return fig


def empty_figure(title, message, height=320):
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 15, "family": PLOT_FONT, "color": "#5a6d91"},
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return style_chart_figure(fig, title, height=height)


def compute_summary(filtered_df, astronaut_id, hours):
    latest_row = filtered_df.sort_values("timestamp_h").iloc[-1]
    events_df = active_events(filtered_df)
    avg_hr = float(filtered_df["heart_rate_bpm"].mean())
    peak_hr = float(filtered_df["heart_rate_bpm"].max())
    avg_sleep = float(filtered_df["sleep_quality"].mean())
    mean_fatigue = float(filtered_df["fatigue_index"].mean())
    peak_fatigue = float(filtered_df["fatigue_index"].max())
    avg_stress = float(filtered_df["stress_index"].mean())
    event_count = int(len(events_df))
    fatigue_delta = float(filtered_df["fatigue_index"].iloc[-1] - filtered_df["fatigue_index"].iloc[0])
    window_hours = float(hours[1] - hours[0])
    top_event = events_df["event_type"].value_counts().idxmax() if not events_df.empty else "none"

    fatigue_risk = float(
        np.clip(
            (peak_fatigue / 9.0) * 60
            + max(0.0, 6.2 - avg_sleep) * 8
            + max(0.0, avg_stress - 42.0) * 0.55
            + event_count * 1.8,
            4,
            97,
        )
    )
    readiness_score = float(
        np.clip(
            latest_row["recovery_reserve_pct"] * 0.46
            + latest_row["circadian_alignment_pct"] * 0.26
            + avg_sleep * 3.4
            - max(0.0, avg_hr - 84.0) * 0.9,
            12,
            99,
        )
    )
    recovery_time_h = float(
        np.clip(
            peak_fatigue * 1.6
            + max(0.0, 6.0 - avg_sleep) * 1.7
            + max(0.0, avg_stress - 45.0) * 0.1
            + event_count * 0.35,
            3,
            36,
        )
    )

    if fatigue_risk < 35 and readiness_score >= 75:
        mission_state = "Stable"
        tone = "good"
    elif fatigue_risk < 65:
        mission_state = "Watch"
        tone = "warn"
    else:
        mission_state = "High Attention"
        tone = "alert"

    if fatigue_delta > 0.6:
        fatigue_direction = "rising"
    elif fatigue_delta < -0.6:
        fatigue_direction = "recovering"
    else:
        fatigue_direction = "steady"

    top_event_label = (
        top_event.replace("_", " ").title() if top_event != "none" else "No active disruptions"
    )

    if mission_state == "Stable":
        headline = f"{astronaut_label(astronaut_id)} is maintaining a controlled physiological profile."
    elif mission_state == "Watch":
        headline = f"{astronaut_label(astronaut_id)} is entering a caution window and should be watched closely."
    else:
        headline = f"{astronaut_label(astronaut_id)} requires high-attention monitoring and recovery planning."

    narrative = (
        f"Average heart rate is {avg_hr:.1f} bpm, oxygen saturation is {latest_row['oxygen_saturation_pct']:.1f}%, "
        f"and stress load is {latest_row['stress_index']:.0f}/100. The dominant disruption pattern is "
        f"{top_event_label.lower()}."
    )

    return {
        "astronaut_id": astronaut_id,
        "avg_hr": avg_hr,
        "peak_hr": peak_hr,
        "avg_sleep": avg_sleep,
        "mean_fatigue": mean_fatigue,
        "peak_fatigue": peak_fatigue,
        "event_count": event_count,
        "fatigue_risk": fatigue_risk,
        "readiness_score": readiness_score,
        "recovery_time_h": recovery_time_h,
        "window_hours": window_hours,
        "mission_state": mission_state,
        "tone": tone,
        "fatigue_direction": fatigue_direction,
        "top_event_label": top_event_label,
        "headline": headline,
        "narrative": narrative,
        "avg_stress": avg_stress,
        "latest_oxygen": float(latest_row["oxygen_saturation_pct"]),
        "latest_hydration": float(latest_row["hydration_level_pct"]),
        "latest_circadian": float(latest_row["circadian_alignment_pct"]),
        "latest_recovery_reserve": float(latest_row["recovery_reserve_pct"]),
        "latest_stress": float(latest_row["stress_index"]),
        "latest_at_risk": bool(latest_row.get("at_risk", False)),
        "fatigue_load_estimate": float(mean_fatigue * window_hours),
    }


def create_status_badges(summary, twin_sync, current_phase):
    badges = [
        html.Span(
            "Real simulation feed" if SIMULATION_FILE.exists() else "Demo mission data",
            className="status-pill status-pill--navy",
        ),
        html.Span(
            f"{current_phase['name']} window",
            className="status-pill status-pill--indigo",
        ),
        html.Span(
            "Analytics summary loaded" if external_analytics else "Derived analytics only",
            className="status-pill status-pill--slate",
        ),
        html.Span(
            f"Twin sync {twin_sync['score']:.0f}%",
            className="status-pill status-pill--teal",
        ),
        html.Span(
            "At-risk flag active" if summary["latest_at_risk"] else f"Readiness {summary['readiness_score']:.0f}/100",
            className="status-pill status-pill--amber",
        ),
    ]
    return badges


def create_kpi_cards(summary, twin_sync):
    risk_value = external_analytics.get("risk_prob_fatigue")
    recovery_value = external_analytics.get("mean_recovery_time_h")
    fatigue_load = external_analytics.get("fatigue_load")
    cards = [
        {
            "label": "Average Heart Rate",
            "value": f"{summary['avg_hr']:.1f} bpm",
            "note": f"Peak {summary['peak_hr']:.1f} bpm",
            "tone": "coral",
        },
        {
            "label": "Oxygen Saturation",
            "value": f"{summary['latest_oxygen']:.1f}%",
            "note": "Derived biomedical estimate",
            "tone": "teal",
        },
        {
            "label": "Fatigue Risk",
            "value": f"{((risk_value * 100) if risk_value is not None else summary['fatigue_risk']):.0f}%",
            "note": "From analytics_summary.json" if risk_value is not None else summary["fatigue_direction"].title(),
            "tone": "amber",
        },
        {
            "label": "Recovery Horizon",
            "value": f"{(recovery_value if recovery_value is not None else summary['recovery_time_h']):.1f} h",
            "note": f"Fatigue load {(fatigue_load if fatigue_load is not None else summary['fatigue_load_estimate']):.1f}",
            "tone": "slate",
        },
        {
            "label": "Twin Sync",
            "value": f"{twin_sync['score']:.0f}%",
            "note": twin_sync["label"],
            "tone": "indigo",
        },
        {
            "label": "Risk Flag",
            "value": "Active" if summary["latest_at_risk"] else "Nominal",
            "note": f"Hydration {summary['latest_hydration']:.0f}% | Stress {summary['latest_stress']:.0f}/100",
            "tone": "teal",
        },
    ]

    return [
        html.Div(
            className=f"stat-card tone-{card['tone']}",
            children=[
                html.Span(card["label"], className="stat-card__label"),
                html.Strong(card["value"], className="stat-card__value"),
                html.Span(card["note"], className="stat-card__note"),
            ],
        )
        for card in cards
    ]


def create_metric_bar(label, value_text, percent):
    return html.Div(
        className="metric-bar",
        children=[
            html.Div(
                className="metric-bar__header",
                children=[
                    html.Span(label, className="metric-bar__label"),
                    html.Span(value_text, className="metric-bar__value"),
                ],
            ),
            html.Div(
                className="metric-bar__track",
                children=[
                    html.Div(
                        className="metric-bar__fill",
                        style={"width": f"{max(4, min(percent, 100)):.0f}%"},
                    )
                ],
            ),
        ],
    )


def create_analytics_panel(summary, twin_sync, current_phase):
    risk_value = external_analytics.get("risk_prob_fatigue")
    recovery_value = external_analytics.get("mean_recovery_time_h")
    fatigue_load = external_analytics.get("fatigue_load")
    conclusions = external_analytics.get("conclusions", [])

    fatigue_percent = (risk_value * 100) if risk_value is not None else summary["fatigue_risk"]
    recovery_hours = recovery_value if recovery_value is not None else summary["recovery_time_h"]
    fatigue_load_value = fatigue_load if fatigue_load is not None else summary["fatigue_load_estimate"]

    analytics_children = [
        html.Div(
            className=f"analysis-callout analysis-callout--{summary['tone']}",
            children=[
                html.Span(summary["mission_state"], className="analysis-callout__tag"),
                html.H3(summary["headline"], className="analysis-callout__title"),
                html.P(summary["narrative"], className="analysis-callout__text"),
            ],
        ),
        html.Div(
            className="metric-stack",
            children=[
                create_metric_bar(
                    "Fatigue risk",
                    f"{fatigue_percent:.0f}%",
                    fatigue_percent,
                ),
                create_metric_bar(
                    "Recovery horizon",
                    f"{recovery_hours:.1f} h",
                    (recovery_hours / 36.0) * 100,
                ),
                create_metric_bar(
                    "Twin sync",
                    f"{twin_sync['score']:.0f}%",
                    twin_sync["score"],
                ),
            ],
        ),
        html.Div(
            className="analytics-note-grid",
            children=[
                html.Div(
                    className="analytics-note",
                    children=[
                        html.Span("Mission phase", className="analytics-note__label"),
                        html.Strong(current_phase["name"], className="analytics-note__value"),
                    ],
                ),
                html.Div(
                    className="analytics-note",
                    children=[
                        html.Span("Dominant disruption", className="analytics-note__label"),
                        html.Strong(summary["top_event_label"], className="analytics-note__value"),
                    ],
                ),
                html.Div(
                    className="analytics-note",
                    children=[
                        html.Span("Fatigue load", className="analytics-note__label"),
                        html.Strong(f"{fatigue_load_value:.1f}", className="analytics-note__value"),
                    ],
                ),
                html.Div(
                    className="analytics-note",
                    children=[
                        html.Span("Twin mode", className="analytics-note__label"),
                        html.Strong(twin_sync["mode"], className="analytics-note__value"),
                    ],
                ),
            ],
        ),
    ]

    team_risk = safe_float(risk_value)
    team_recovery = safe_float(recovery_value)
    if conclusions:
        analytics_children.append(
            html.Div(
                className="analytics-note-grid",
                children=[
                    html.Div(
                        className="analytics-note",
                        children=[
                            html.Span("Conclusion 1", className="analytics-note__label"),
                            html.Strong(conclusions[0], className="analytics-note__value"),
                        ],
                    )
                ]
                + (
                    [
                        html.Div(
                            className="analytics-note",
                            children=[
                                html.Span("Conclusion 2", className="analytics-note__label"),
                                html.Strong(conclusions[1], className="analytics-note__value"),
                            ],
                        )
                    ]
                    if len(conclusions) > 1
                    else []
                ),
            )
        )

    if team_risk is not None or team_recovery is not None:
        parts = []
        if team_risk is not None:
            parts.append(f"fatigue risk {risk_value*100:.0f}%")
        if team_recovery is not None:
            parts.append(f"mean recovery {team_recovery:.1f} h")
        analytics_children.append(
            html.P(
                f"analytics_summary.json is actively driving this panel with {', '.join(parts)}.",
                className="analytics-footnote",
            )
        )
    else:
        analytics_children.append(
            html.P(
                "No analytics_summary.json file is loaded, so this panel is falling back to derived live estimates.",
                className="analytics-footnote",
            )
        )

    return analytics_children


def create_phase_rail(hours, current_phase):
    segments = []
    for phase in MISSION_PHASES:
        overlap = max(0.0, min(hours[1], phase["end"]) - max(hours[0], phase["start"]))
        selected_ratio = overlap / max(phase["end"] - phase["start"], 1)
        classes = ["phase-segment"]
        if phase["name"] == current_phase["name"]:
            classes.append("phase-segment--active")
        elif overlap > 0:
            classes.append("phase-segment--selected")

        segments.append(
            html.Div(
                className=" ".join(classes),
                style={"--phase-color": phase["color"]},
                children=[
                    html.Div(
                        className="phase-segment__bar",
                        children=[
                            html.Div(
                                className="phase-segment__fill",
                                style={"width": f"{selected_ratio * 100:.0f}%"},
                            )
                        ],
                    ),
                    html.Span(phase["name"], className="phase-segment__name"),
                    html.Span(
                        f"{phase['start']:.0f}-{phase['end']:.0f}h",
                        className="phase-segment__time",
                    ),
                ],
            )
        )

    return [
        html.Div(className="phase-rail-track", children=segments),
        html.Div(
            className="phase-summary",
            children=[
                html.Span("Current phase", className="phase-summary__label"),
                html.Strong(current_phase["name"], className="phase-summary__value"),
                html.P(current_phase["objective"], className="phase-summary__text"),
            ],
        ),
    ]


def create_profile_panel(profile, summary, latest_row, twin_sync, current_phase):
    return [
        html.Div(
            className="profile-hero",
            children=[
                html.Span(profile["tag"], className="profile-hero__tag"),
                html.H3(profile["name"], className="profile-hero__name"),
                html.P(
                    f"{profile['role']} | {profile['call_sign']}",
                    className="profile-hero__meta",
                ),
            ],
        ),
        html.Div(
            className="profile-grid",
            children=[
                html.Div(
                    className="profile-cell",
                    children=[
                        html.Span("Specialty", className="profile-cell__label"),
                        html.Strong(profile["specialty"], className="profile-cell__value"),
                    ],
                ),
                html.Div(
                    className="profile-cell",
                    children=[
                        html.Span("Shift", className="profile-cell__label"),
                        html.Strong(profile["shift"], className="profile-cell__value"),
                    ],
                ),
                html.Div(
                    className="profile-cell",
                    children=[
                        html.Span("Flight Hours", className="profile-cell__label"),
                        html.Strong(f"{profile['flight_hours']} h", className="profile-cell__value"),
                    ],
                ),
                html.Div(
                    className="profile-cell",
                    children=[
                        html.Span("Mission Phase", className="profile-cell__label"),
                        html.Strong(current_phase["name"], className="profile-cell__value"),
                    ],
                ),
            ],
        ),
        html.Div(
            className="sync-panel",
            children=[
                html.Div(
                    className="sync-panel__score",
                    children=[
                        html.Span("Twin Sync", className="sync-panel__label"),
                        html.Strong(f"{twin_sync['score']:.0f}%", className="sync-panel__value"),
                    ],
                ),
                html.Div(
                    className="sync-panel__body",
                    children=[
                        html.P(twin_sync["label"], className="sync-panel__title"),
                        html.P(twin_sync["message"], className="sync-panel__text"),
                    ],
                ),
            ],
        ),
    ]


def create_vitals_panel(latest_row):
    respiratory_value = latest_row.get("respiratory_rate_bpm")

    if respiratory_value is None or pd.isna(respiratory_value):
        respiratory_value = 16.0
    hrv_value = latest_row.get("hrv_proxy")
    cards = [
        {
            "label": "O2 Saturation",
            "value": f"{latest_row['oxygen_saturation_pct']:.1f}%",
            "note": "Derived from fatigue and event load",
        },
        {
            "label": "Respiratory Rate",
            "value": (
                f"{respiratory_value:.1f} bpm"
                if respiratory_value is not None and not pd.isna(respiratory_value)
                else f"{latest_row['stress_index']:.0f}/100"
            ),
            "note": (
                "Direct T1 simulation field"
                if respiratory_value is not None and not pd.isna(respiratory_value)
                else "Operational cognitive load estimate"
            ),
        },
        {
            "label": "HRV Proxy",
            "value": (
                f"{hrv_value:.0f}"
                if hrv_value is not None and not pd.isna(hrv_value)
                else f"{latest_row['core_temp_c']:.2f} C"
            ),
            "note": (
                "Direct T1 simulation field"
                if hrv_value is not None and not pd.isna(hrv_value)
                else "Thermal regulation signal"
            ),
        },
        {
            "label": "Radiation Dose",
            "value": f"{latest_row['radiation_dose_msv']:.2f} mSv",
            "note": "Cumulative exposure in window",
        },
    ]
    return [
        html.Div(
            className="vital-card",
            children=[
                html.Span(card["label"], className="vital-card__label"),
                html.Strong(card["value"], className="vital-card__value"),
                html.Span(card["note"], className="vital-card__note"),
            ],
        )
        for card in cards
    ]


def create_alert_feed(alerts):
    return [
        html.Div(
            className=f"alert-item alert-item--{alert['tone']}",
            children=[
                html.Span(alert["title"], className="alert-item__title"),
                html.P(alert["body"], className="alert-item__body"),
            ],
        )
        for alert in alerts
    ]


def build_timeline_figure(filtered_df, bio_filtered, astronaut_id, hours):
    if filtered_df.empty:
        return empty_figure(
            "Health Timeline",
            "No telemetry is available for the current astronaut and time range.",
            height=430,
        )

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=filtered_df["timestamp_h"],
            y=filtered_df["heart_rate_bpm"],
            mode="lines",
            name="Heart Rate",
            line={"color": METRIC_COLORS["heart_rate"], "width": 3},
            hovertemplate="Time %{x:.1f} h<br>Heart rate %{y:.1f} bpm<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=filtered_df["timestamp_h"],
            y=filtered_df["fatigue_index"],
            mode="lines",
            name="Fatigue Index",
            line={"color": METRIC_COLORS["fatigue"], "width": 2.8},
            hovertemplate="Time %{x:.1f} h<br>Fatigue %{y:.1f}<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=filtered_df["timestamp_h"],
            y=filtered_df["sleep_quality"],
            mode="lines",
            name="Sleep Quality",
            line={"color": METRIC_COLORS["sleep"], "width": 2.4},
            hovertemplate="Time %{x:.1f} h<br>Sleep quality %{y:.1f}<extra></extra>",
        ),
        secondary_y=True,
    )

    if bio_filtered.empty and "biogears_hr_bpm" in filtered_df.columns:
        synthetic_bio = filtered_df[["timestamp_h", "biogears_hr_bpm"]].dropna()
        if not synthetic_bio.empty:
            bio_filtered = synthetic_bio.rename(columns={"biogears_hr_bpm": "heart_rate_bpm"})

    if not bio_filtered.empty:
        fig.add_trace(
            go.Scatter(
                x=bio_filtered["timestamp_h"],
                y=bio_filtered["heart_rate_bpm"],
                mode="lines",
                name="BioGears HR",
                line={"color": METRIC_COLORS["biogears"], "width": 2.5, "dash": "dash"},
                hovertemplate="Time %{x:.1f} h<br>BioGears HR %{y:.1f} bpm<extra></extra>",
            ),
            secondary_y=False,
        )

    events_df = active_events(filtered_df)
    max_hr = float(filtered_df["heart_rate_bpm"].max())
    if not bio_filtered.empty:
        max_hr = max(max_hr, float(bio_filtered["heart_rate_bpm"].max()))

    if not events_df.empty:
        for row in events_df.itertuples(index=False):
            fig.add_vline(
                x=row.timestamp_h,
                line_width=1.2,
                line_dash="dash",
                line_color=EVENT_COLORS.get(row.event_type, "#94a3b8"),
                opacity=0.32,
            )

        fig.add_trace(
            go.Scatter(
                x=events_df["timestamp_h"],
                y=np.full(len(events_df), max_hr + 3),
                mode="markers",
                name="Mission Events",
                marker={
                    "size": 9,
                    "color": [EVENT_COLORS.get(event_name, "#94a3b8") for event_name in events_df["event_type"]],
                    "line": {"width": 1.0, "color": "#ffffff"},
                },
                text=events_df["event_type"].str.replace("_", " ").str.title(),
                hovertemplate="Event %{text}<br>Time %{x:.1f} h<extra></extra>",
            ),
            secondary_y=False,
        )

    fig = style_chart_figure(
        fig,
        f"{astronaut_label(astronaut_id)} Physiology Timeline",
        height=450,
    )
    fig.update_xaxes(title_text="Mission time (hours)", range=[hours[0], hours[1]])
    fig.update_yaxes(
        title_text="Heart rate (bpm)",
        range=[max(50, filtered_df["heart_rate_bpm"].min() - 6), max_hr + 7],
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="Fatigue / Sleep score",
        range=[0, max(10, filtered_df[["fatigue_index", "sleep_quality"]].max().max() + 1)],
        secondary_y=True,
    )
    return fig


def build_heatmap_figure(full_df, hours):
    window_df = full_df[
        (full_df["timestamp_h"] >= hours[0]) & (full_df["timestamp_h"] <= hours[1])
    ]
    if window_df.empty:
        return empty_figure(
            "Crew Fatigue Heatmap",
            "No data is available in the selected window.",
        )

    heatmap_df = window_df.pivot_table(
        index="astronaut_id",
        columns="timestamp_h",
        values="fatigue_index",
        aggfunc="mean",
    )
    heatmap_df = heatmap_df.sort_index()
    heatmap_df.index = [astronaut_label(value) for value in heatmap_df.index]

    fig = px.imshow(
        heatmap_df,
        aspect="auto",
        color_continuous_scale=["#edf3fb", "#a4ddd3", "#ffd179", "#f16745"],
        labels={"x": "Mission time (hours)", "y": "Crew", "color": "Fatigue"},
    )
    fig = style_chart_figure(fig, "Crew Fatigue Heatmap", height=340)
    fig.update_layout(coloraxis_colorbar={"title": "Fatigue"})
    fig.update_xaxes(title_text="Mission time (hours)")
    fig.update_yaxes(title_text="Crew member")
    return fig


def build_event_figure(filtered_df):
    segments_df = build_event_segments(filtered_df)
    if segments_df.empty:
        return empty_figure(
            "Event Gantt Timeline",
            "No active mission events were recorded in this time window.",
        )

    fig = px.timeline(
        segments_df,
        x_start="start_h",
        x_end="end_h",
        y="event_label",
        color="event_type",
        color_discrete_map=EVENT_COLORS,
        custom_data=["start_h", "end_h"],
    )
    fig = style_chart_figure(fig, "Event Gantt Timeline", height=320)
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title_text="Mission time (hours)")
    fig.update_yaxes(title_text="Event type", autorange="reversed")
    fig.update_traces(
        hovertemplate="%{y}<br>Start %{customdata[0]:.1f} h<br>End %{customdata[1]:.1f} h<extra></extra>"
    )
    return fig


def build_readiness_figure(summary):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=summary["readiness_score"],
            number={"suffix": "/100", "font": {"size": 34, "color": "#12233d"}},
            title={"text": "Mission readiness", "font": {"size": 18, "color": "#12233d"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#65789d"},
                "bar": {"color": "#2fb49d"},
                "bgcolor": "#eef2f8",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": "#f8c4b8"},
                    {"range": [40, 70], "color": "#ffe0a6"},
                    {"range": [70, 100], "color": "#ccece3"},
                ],
                "threshold": {
                    "line": {"color": "#12233d", "width": 4},
                    "thickness": 0.78,
                    "value": summary["readiness_score"],
                },
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        font={"family": PLOT_FONT, "color": "#1f3559"},
        margin={"l": 28, "r": 28, "t": 62, "b": 28},
        height=320,
    )
    return fig


def build_radar_figure(summary, latest_row, twin_sync):
    scores = build_radar_scores(summary, latest_row, twin_sync)
    labels = list(scores.keys())
    values = list(scores.values())
    values.append(values[0])
    labels.append(labels[0])

    fig = go.Figure(
        go.Scatterpolar(
            r=values,
            theta=labels,
            fill="toself",
            line={"color": "#4f7cff", "width": 2.5},
            fillcolor="rgba(79, 124, 255, 0.18)",
            hovertemplate="%{theta}: %{r:.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        polar={
            "radialaxis": {"visible": True, "range": [0, 100], "gridcolor": "rgba(31, 53, 89, 0.12)"},
            "angularaxis": {"gridcolor": "rgba(31, 53, 89, 0.08)", "color": "#1f3559"},
            "bgcolor": "rgba(0, 0, 0, 0)",
        },
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        font={"family": PLOT_FONT, "color": "#1f3559"},
        margin={"l": 28, "r": 28, "t": 72, "b": 32},
        height=330,
        title={
            "text": "Crew Resilience Radar",
            "x": 0.02,
            "xanchor": "left",
            "font": {"size": 20, "family": PLOT_FONT, "color": "#12233d"},
        },
        showlegend=False,
    )
    return fig


def build_forecast_figure(filtered_df, forecast_df):
    if not monte_carlo_results_df.empty and "peak_fatigue" in monte_carlo_results_df.columns:
        fig = px.histogram(
            monte_carlo_results_df.dropna(subset=["peak_fatigue"]),
            x="peak_fatigue",
            nbins=20,
            color_discrete_sequence=["#7f8cff"],
        )
        fig.add_vline(x=7.0, line_color="#f16745", line_dash="dash")
        if "recovery_time_h" in monte_carlo_results_df.columns:
            avg_recovery = monte_carlo_results_df["recovery_time_h"].dropna().mean()
            if not pd.isna(avg_recovery):
                fig.add_annotation(
                    x=0.98,
                    y=0.95,
                    xref="paper",
                    yref="paper",
                    xanchor="right",
                    text=f"Mean recovery {avg_recovery:.1f} h",
                    showarrow=False,
                    font={"size": 12, "family": PLOT_FONT, "color": "#1f3559"},
                )
        fig = style_chart_figure(fig, "Monte Carlo Risk Projection", height=330)
        fig.update_xaxes(title_text="Peak fatigue")
        fig.update_yaxes(title_text="Run count")
        return fig

    if filtered_df.empty or forecast_df.empty:
        return empty_figure(
            "Short-Horizon Forecast",
            "Forecasting requires enough telemetry history to extrapolate the next mission block.",
        )

    history = filtered_df.sort_values("timestamp_h").tail(min(16, len(filtered_df)))
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=history["timestamp_h"],
            y=history["fatigue_index"],
            mode="lines",
            name="Recent fatigue",
            line={"color": METRIC_COLORS["fatigue"], "width": 2.4},
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df["timestamp_h"],
            y=forecast_df["fatigue_forecast"],
            mode="lines",
            name="Forecast fatigue",
            line={"color": METRIC_COLORS["forecast"], "width": 2.6, "dash": "dash"},
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df["timestamp_h"],
            y=forecast_df["readiness_forecast"],
            mode="lines",
            name="Forecast readiness",
            line={"color": METRIC_COLORS["stress"], "width": 2.6},
        ),
        secondary_y=True,
    )
    fig = style_chart_figure(fig, "Short-Horizon Forecast", height=330)
    fig.update_xaxes(title_text="Mission time (hours)")
    fig.update_yaxes(title_text="Fatigue score", range=[0, 10], secondary_y=False)
    fig.update_yaxes(title_text="Readiness /100", range=[0, 100], secondary_y=True)
    return fig


def build_window_caption(filtered_df, summary, hours, current_phase):
    return (
        f"Viewing {hours[0]:.1f}h to {hours[1]:.1f}h | "
        f"{len(filtered_df)} samples | "
        f"{summary['event_count']} active events | "
        f"{current_phase['name']}"
    )


df = load_simulation_data()
bio_df = load_biogears_data()
external_analytics = load_external_analytics()
monte_carlo_results_df = load_monte_carlo_results()
astronaut_ids = sorted(df["astronaut_id"].dropna().unique().tolist(), key=lambda value: str(value))
if not astronaut_ids:
    astronaut_ids = [1]

crew_profiles = build_crew_profiles(astronaut_ids)
time_min = float(df["timestamp_h"].min())
time_max = float(df["timestamp_h"].max())

app = dash.Dash(__name__)
app.title = "Astronaut Health Digital Twin"

app.layout = html.Div(
    className="page-shell",
    children=[
        dcc.Store(id="viewer-sync"),
        html.Div(id="dummy-output", style={"display": "none"}),
        html.Div(id="launch-dummy", style={"display": "none"}),
        html.Div(className="page-glow page-glow--left"),
        html.Div(className="page-glow page-glow--right"),
        html.Div(
            className="hero glass-card",
            children=[
                html.Div(
                    className="hero__copy",
                    children=[
                        html.P("T3 Dashboard Command Surface", className="eyebrow"),
                        html.H1(
                            "Astronaut Health Digital Twin Dashboard",
                            className="hero__title",
                        ),
                        html.P(
                            "Track physiology, mission phase, twin synchronization, crew resilience, "
                            "and recovery forecasting from one astronaut-focused command surface.",
                            className="hero__text",
                        ),
                        html.Div(id="status-badges", className="status-row"),
                    ],
                ),
                html.Div(
                    className="hero__visual",
                    children=[
                        html.Div(className="hero-orbit"),
                        html.Div(
                            className="hero__visual-card",
                            children=[
                                html.Span("Digital twin horizon", className="hero__visual-label"),
                                html.Strong("72 h", className="hero__visual-value"),
                                html.Span(
                                    "Crew physiology, mission phase, and biomedical risk fused into one view",
                                    className="hero__visual-note",
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        html.Div(
            className="controls-shell glass-card",
            children=[
                html.Div(
                    className="controls-pane",
                    children=[
                        html.Div(
                            className="field-block",
                            children=[
                                html.Label("Crew member", className="field-label"),
                                dcc.Dropdown(
                                    id="astronaut",
                                    className="control-dropdown",
                                    options=[
                                        {
                                            "label": astronaut_label(astronaut_id),
                                            "value": astronaut_id,
                                        }
                                        for astronaut_id in astronaut_ids
                                    ],
                                    value=astronaut_ids[0],
                                    clearable=False,
                                ),
                            ],
                        ),
                        html.Div(
                            className="field-block",
                            children=[
                                html.Label("Mission window", className="field-label"),
                                html.Div(
                                    className="slider-shell",
                                    children=[
                                        dcc.Slider(
                                            id="slider",
                                            min=time_min,
                                            max=time_max,
                                            value=time_min,
                                            marks=make_slider_marks(time_min, time_max),
                                            step=0.5
                                        )
                                    ],
                                ),
                                html.Div(id="window-caption", className="window-caption"),
                            ],
                        ),

                        html.Button(
                        "🚀 Launch 3D Twin Viewer",
                        id="launch-viewer",
                        n_clicks=0,
                        style={
                                "display": "inline-block",
                                "marginTop": "14px",
                                "padding": "10px 16px",
                                "backgroundColor": "#2563eb",
                                "color": "white",
                                "textDecoration": "none",
                                "borderRadius": "10px",
                                "fontWeight": "600"
                            }
                        ),












                    ],
                ),
                html.Div(id="kpi-cards", className="kpi-grid"),
            ],
        ),
        html.Div(
            className="mission-grid",
            children=[
                html.Section(
                    className="glass-card mission-track-card",
                    children=[
                        html.Div(
                            className="section-heading",
                            children=[
                                html.P("Mission Layer", className="section-kicker"),
                                html.H2("Mission Phase Timeline", className="section-title"),
                            ],
                        ),
                        html.Div(id="phase-rail", className="phase-rail"),
                    ],
                ),
                html.Section(
                    className="glass-card profile-shell",
                    children=[
                        html.Div(
                            className="section-heading",
                            children=[
                                html.P("Crew Layer", className="section-kicker"),
                                html.H2("Crew Profile and Biomedical Snapshot", className="section-title"),
                            ],
                        ),
                        html.Div(id="crew-profile", className="crew-profile"),
                        html.Div(className="profile-subheading", children="Biomedical Status"),
                        html.Div(id="vitals-panel", className="vitals-grid"),
                        html.Div(className="profile-subheading", children="Operational Alerts"),
                        html.Div(id="alert-feed", className="alert-feed"),
                    ],
                ),
            ],
        ),
        html.Div(
            className="dashboard-grid expanded-grid",
            children=[
                html.Section(
                    className="glass-card chart-card timeline-card",
                    children=[
                        html.Div(
                            className="section-heading",
                            children=[
                                html.P("Primary View", className="section-kicker"),
                                html.H2("Physiology Timeline", className="section-title"),
                            ],
                        ),
                        dcc.Graph(id="timeline-graph", config=GRAPH_CONFIG, className="graph-frame"),
                    ],
                ),
                html.Section(
                    className="glass-card chart-card heatmap-card",
                    children=[
                        html.Div(
                            className="section-heading",
                            children=[
                                html.P("Crew Patterns", className="section-kicker"),
                                html.H2("Fatigue Heatmap", className="section-title"),
                            ],
                        ),
                        dcc.Graph(id="heatmap-graph", config=GRAPH_CONFIG, className="graph-frame"),
                    ],
                ),
                html.Section(
                    className="glass-card analytics-shell analytics-card",
                    children=[
                        html.Div(
                            className="section-heading",
                            children=[
                                html.P("Analytics Panel", className="section-kicker"),
                                html.H2("Mission Insight", className="section-title"),
                            ],
                        ),
                        html.Div(id="analytics-panel", className="analytics-panel"),
                    ],
                ),
                html.Section(
                    className="glass-card chart-card radar-card",
                    children=[
                        html.Div(
                            className="section-heading",
                            children=[
                                html.P("Resilience Model", className="section-kicker"),
                                html.H2("Crew Resilience Radar", className="section-title"),
                            ],
                        ),
                        dcc.Graph(id="radar-graph", config=GRAPH_CONFIG, className="graph-frame"),
                    ],
                ),
                html.Section(
                    className="glass-card chart-card forecast-card",
                    children=[
                        html.Div(
                            className="section-heading",
                            children=[
                                html.P("Forecast Layer", className="section-kicker"),
                                html.H2("Risk Projection", className="section-title"),
                            ],
                        ),
                        dcc.Graph(id="forecast-graph", config=GRAPH_CONFIG, className="graph-frame"),
                    ],
                ),
                html.Section(
                    className="glass-card chart-card event-card",
                    children=[
                        html.Div(
                            className="section-heading",
                            children=[
                                html.P("Event Context", className="section-kicker"),
                                html.H2("Event Gantt Timeline", className="section-title"),
                            ],
                        ),
                        dcc.Graph(id="events-graph", config=GRAPH_CONFIG, className="graph-frame"),
                    ],
                ),
                html.Section(
                    className="glass-card chart-card readiness-card",
                    children=[
                        html.Div(
                            className="section-heading",
                            children=[
                                html.P("Readiness Signal", className="section-kicker"),
                                html.H2("Crew Readiness Gauge", className="section-title"),
                            ],
                        ),
                        dcc.Graph(id="readiness-graph", config=GRAPH_CONFIG, className="graph-frame"),
                    ],
                ),
            ],
        ),
    ],
)


@app.callback(
    Output("timeline-graph", "figure"),
    Output("heatmap-graph", "figure"),
    Output("events-graph", "figure"),
    Output("readiness-graph", "figure"),
    Output("radar-graph", "figure"),
    Output("forecast-graph", "figure"),
    Output("kpi-cards", "children"),
    Output("analytics-panel", "children"),
    Output("status-badges", "children"),
    Output("window-caption", "children"),
    Output("phase-rail", "children"),
    Output("crew-profile", "children"),
    Output("vitals-panel", "children"),
    Output("alert-feed", "children"),
    Output("viewer-sync", "data"),
    Input("astronaut", "value"),
    Input("slider", "value"),
)








def update_dashboard(astronaut_id, hours):
    hours = [0, float(hours)]
    filtered_df = filter_for_astronaut(df, astronaut_id)
    filtered_df = filtered_df[
        (filtered_df["timestamp_h"] >= hours[0]) & (filtered_df["timestamp_h"] <= hours[1])
    ]
    if filtered_df.empty:
        filtered_df = filter_for_astronaut(df, astronaut_id).copy()

    bio_filtered = filter_biogears_for_window(astronaut_id, hours)
    summary = compute_summary(filtered_df, astronaut_id, hours)
    latest_row = filtered_df.sort_values("timestamp_h").iloc[-1]
    current_phase = get_phase_for_window(hours)
    twin_sync = build_twin_sync(filtered_df, bio_filtered)
    alerts = build_alerts(summary, latest_row, twin_sync)
    profile = crew_profiles[str(astronaut_id)]
    forecast_df = build_forecast_frame(filtered_df, hours)

    return (
        build_timeline_figure(filtered_df, bio_filtered, astronaut_id, hours),
        build_heatmap_figure(df, hours),
        build_event_figure(filtered_df),
        build_readiness_figure(summary),
        build_radar_figure(summary, latest_row, twin_sync),
        build_forecast_figure(filtered_df, forecast_df),
        create_kpi_cards(summary, twin_sync),
        create_analytics_panel(summary, twin_sync, current_phase),
        create_status_badges(summary, twin_sync, current_phase),
        build_window_caption(filtered_df, summary, hours, current_phase),
        create_phase_rail(hours, current_phase),
        create_profile_panel(profile, summary, latest_row, twin_sync, current_phase),
        create_vitals_panel(latest_row),
        create_alert_feed(alerts),
        hours[1]
    )

app.clientside_callback(
    """
    function(n) {
        if (n > 0) {
            window.viewerWindow = window.open(
                "http://localhost:8000/viz3d/viewer.html",
                "viewerWindow"
            );
        }
        return "";
    }
    """,
    Output("launch-dummy", "children"),
    Input("launch-viewer", "n_clicks")
)

app.clientside_callback(
    """
    function(hour) {
        if (window.viewerWindow && !window.viewerWindow.closed) {
            window.viewerWindow.postMessage(
                { missionHour: hour },
                "*"
            );
        }
        return "";
    }
    """,
    Output("dummy-output", "children"),
    Input("viewer-sync", "data")
)




if __name__ == "__main__":
    app.run(debug=True)
