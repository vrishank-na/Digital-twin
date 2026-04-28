from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

EVENT_INTENSITY = {
    "motion_sickness": 1.35,
    "sleep_disruption": 1.55,
    "sinus_congestion": 0.8,
    "eva_exertion": 1.85,
    "exercise_load": 1.2,
    "stress_spike": 1.65,
    "none": 0.0,
}

CREW_TEMPLATES = [
    {
        "name": "Vrishank N A",
        "role": "Mission Commander",
        "specialty": "Flight systems and crew coordination",
        "shift": "Alpha Shift",
        "call_sign": "Orion-1",
        "age": 37,
        "flight_hours": 1260,
    },
    {
        "name": "Miles Carter",
        "role": "Biomedical Specialist",
        "specialty": "Countermeasures and physiology review",
        "shift": "Bravo Shift",
        "call_sign": "Nova-2",
        "age": 34,
        "flight_hours": 910,
    },
    {
        "name": "Rhea Iyer",
        "role": "Payload Systems Engineer",
        "specialty": "Habitat systems and digital twin fusion",
        "shift": "Adaptive Shift",
        "call_sign": "Helios-3",
        "age": 32,
        "flight_hours": 780,
    },
    {
        "name": "Lena Okafor",
        "role": "EVA Operations Specialist",
        "specialty": "Suit readiness and excursion planning",
        "shift": "Alpha Shift",
        "call_sign": "Atlas-4",
        "age": 39,
        "flight_hours": 1425,
    },
]

MISSION_PHASES = [
    {
        "name": "Launch Checkout",
        "start": 0,
        "end": 8,
        "color": "#6c7dff",
        "objective": "Baseline stabilization, sensor lock, and post-insertion checks.",
    },
    {
        "name": "Microgravity Adaptation",
        "start": 8,
        "end": 24,
        "color": "#4f7cff",
        "objective": "Vestibular adaptation and early workload monitoring.",
    },
    {
        "name": "Operations Block",
        "start": 24,
        "end": 42,
        "color": "#2fb49d",
        "objective": "Exercise protocols, payload tasks, and sustained crew activity.",
    },
    {
        "name": "Recovery Cycle",
        "start": 42,
        "end": 54,
        "color": "#f4b544",
        "objective": "Sleep recovery, circadian correction, and low-intensity monitoring.",
    },
    {
        "name": "EVA Preparation",
        "start": 54,
        "end": 64,
        "color": "#f16745",
        "objective": "Pre-EVA conditioning, readiness validation, and risk screening.",
    },
    {
        "name": "Post-EVA Recovery",
        "start": 64,
        "end": 72,
        "color": "#d964a7",
        "objective": "Recompression, hydration recovery, and twin recalibration.",
    },
]


def build_crew_profiles(astronaut_ids: List[object]) -> Dict[str, dict]:
    profiles = {}
    for index, astronaut_id in enumerate(astronaut_ids):
        template = CREW_TEMPLATES[index % len(CREW_TEMPLATES)].copy()
        template["astronaut_id"] = astronaut_id
        template["tag"] = f"Mission Day 03 | Crew {index + 1}"
        profiles[str(astronaut_id)] = template
    return profiles


def enrich_simulation_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = dataframe.copy()
    astronaut_numeric = pd.to_numeric(df["astronaut_id"], errors="coerce").fillna(1.0)
    event_intensity = df["event_type"].map(EVENT_INTENSITY).fillna(0.25)
    mission_cycle = np.sin((df["timestamp_h"] / 24.0) * 2 * np.pi + astronaut_numeric * 0.3)
    hydration_drift = np.mod(df["timestamp_h"], 24.0)

    df["oxygen_saturation_pct"] = np.clip(
        98.6
        - event_intensity * 0.8
        - np.maximum(df["fatigue_index"] - 5.0, 0) * 0.32
        + (df["sleep_quality"] - 6.0) * 0.22
        + mission_cycle * 0.18,
        93.8,
        99.7,
    )
    df["hydration_level_pct"] = np.clip(
        91.0
        - hydration_drift * 1.1
        - df["fatigue_index"] * 1.25
        + df["sleep_quality"] * 1.35
        - event_intensity * 2.8,
        58.0,
        98.0,
    )
    df["stress_index"] = np.clip(
        24.0
        + df["fatigue_index"] * 7.5
        + event_intensity * 14.0
        + np.maximum(df["heart_rate_bpm"] - 76.0, 0) * 0.55
        - df["sleep_quality"] * 2.4
        + (1.0 - mission_cycle) * 5.0,
        8.0,
        96.0,
    )
    df["circadian_alignment_pct"] = np.clip(
        72.0
        + mission_cycle * 14.0
        + df["sleep_quality"] * 1.7
        - df["fatigue_index"] * 3.6
        - event_intensity * 3.0,
        20.0,
        99.0,
    )
    df["core_temp_c"] = np.clip(
        36.8
        + event_intensity * 0.12
        + np.maximum(df["heart_rate_bpm"] - 78.0, 0) * 0.006
        - (df["sleep_quality"] - 6.0) * 0.03,
        36.2,
        38.3,
    )
    df["radiation_dose_msv"] = np.clip(
        0.06 + df["timestamp_h"] * 0.013 + astronaut_numeric * 0.02 + event_intensity * 0.03,
        0.05,
        2.6,
    )
    df["recovery_reserve_pct"] = np.clip(
        100.0 - df["stress_index"] * 0.68 - df["fatigue_index"] * 4.2 + df["sleep_quality"] * 4.8,
        6.0,
        96.0,
    )
    return df


def get_phase_for_window(hours: List[float]) -> dict:
    midpoint = (float(hours[0]) + float(hours[1])) / 2.0
    for phase in MISSION_PHASES:
        if phase["start"] <= midpoint < phase["end"]:
            return phase
    return MISSION_PHASES[-1]


def build_twin_sync(filtered_df: pd.DataFrame, bio_df: pd.DataFrame) -> dict:
    if filtered_df.empty:
        return {
            "score": 0.0,
            "mode": "Unavailable",
            "label": "No telemetry",
            "message": "Simulation telemetry is empty for the selected window.",
            "difference_bpm": None,
        }

    if bio_df.empty:
        hr_jitter = filtered_df["heart_rate_bpm"].diff().abs().fillna(0).mean()
        score = float(np.clip(88.0 - hr_jitter * 0.75, 70.0, 92.0))
        return {
            "score": score,
            "mode": "Estimated sync",
            "label": "BioGears pending",
            "message": "Twin confidence is inferred from simulation continuity because no BioGears feed is loaded.",
            "difference_bpm": None,
        }

    overlap = bio_df[
        (bio_df["timestamp_h"] >= filtered_df["timestamp_h"].min())
        & (bio_df["timestamp_h"] <= filtered_df["timestamp_h"].max())
    ].sort_values("timestamp_h")
    if overlap.empty or len(filtered_df) < 3:
        return {
            "score": 62.0,
            "mode": "Partial sync",
            "label": "Limited overlap",
            "message": "BioGears data exists but overlap with the selected simulation window is limited.",
            "difference_bpm": None,
        }

    simulation = filtered_df.sort_values("timestamp_h")
    sim_interp = np.interp(
        overlap["timestamp_h"],
        simulation["timestamp_h"],
        simulation["heart_rate_bpm"],
    )
    diff = np.abs(sim_interp - overlap["heart_rate_bpm"].to_numpy())
    mae = float(diff.mean())
    corr = np.corrcoef(sim_interp, overlap["heart_rate_bpm"].to_numpy())[0, 1]
    if np.isnan(corr):
        corr = 0.65

    score = float(np.clip(100.0 - mae * 4.2 + corr * 8.5, 35.0, 99.0))
    if score >= 88:
        label = "High fidelity"
        message = "Simulation and BioGears are tightly aligned across the selected mission window."
    elif score >= 72:
        label = "Moderate drift"
        message = "Twin alignment is usable, but heart-rate traces show measurable divergence."
    else:
        label = "Recalibration needed"
        message = "BioGears and simulation differ enough to justify parameter review."

    return {
        "score": score,
        "mode": "Measured sync",
        "label": label,
        "message": message,
        "difference_bpm": mae,
    }


def build_alerts(summary: dict, latest_row: pd.Series, twin_sync: dict) -> List[dict]:
    alerts = []

    if bool(latest_row.get("at_risk", False)):
        alerts.append(
            {
                "tone": "critical",
                "title": "Shared at_risk flag raised",
                "body": "The incoming simulation row is explicitly marked at_risk by the shared team pipeline.",
            }
        )
    if summary["fatigue_risk"] >= 70:
        alerts.append(
            {
                "tone": "critical",
                "title": "Fatigue threshold exceeded",
                "body": f"Peak fatigue is {summary['peak_fatigue']:.1f}/9 and readiness is trending down.",
            }
        )
    if latest_row["oxygen_saturation_pct"] < 95.2:
        alerts.append(
            {
                "tone": "watch",
                "title": "Oxygen saturation soft dip",
                "body": f"Latest oxygen estimate is {latest_row['oxygen_saturation_pct']:.1f}%, worth monitoring.",
            }
        )
    if latest_row["hydration_level_pct"] < 70:
        alerts.append(
            {
                "tone": "watch",
                "title": "Hydration reserve reduced",
                "body": f"Hydration is at {latest_row['hydration_level_pct']:.0f}%, suggesting recovery intake is needed.",
            }
        )
    if twin_sync["score"] < 80:
        alerts.append(
            {
                "tone": "info",
                "title": "Twin alignment check",
                "body": twin_sync["message"],
            }
        )
    if summary["event_count"] >= 5:
        alerts.append(
            {
                "tone": "watch",
                "title": "Event density elevated",
                "body": f"{summary['event_count']} active disruptions were recorded in the selected interval.",
            }
        )

    if not alerts:
        alerts.append(
            {
                "tone": "nominal",
                "title": "Nominal operations",
                "body": "Telemetry, event load, and digital twin confidence are all within a controlled range.",
            }
        )

    return alerts[:4]


def build_radar_scores(summary: dict, latest_row: pd.Series, twin_sync: dict) -> dict:
    return {
        "Cardio Stability": float(
            np.clip(
                100.0 - abs(summary["avg_hr"] - 72.0) * 2.4 - max(0.0, summary["peak_hr"] - 88.0) * 1.15,
                12.0,
                98.0,
            )
        ),
        "Sleep Recovery": float(
            np.clip(summary["avg_sleep"] * 10.0 + latest_row["recovery_reserve_pct"] * 0.35, 10.0, 99.0)
        ),
        "Hydration Reserve": float(np.clip(latest_row["hydration_level_pct"], 10.0, 99.0)),
        "Circadian Align": float(np.clip(latest_row["circadian_alignment_pct"], 10.0, 99.0)),
        "Twin Confidence": float(np.clip(twin_sync["score"], 10.0, 99.0)),
        "Event Resilience": float(
            np.clip(
                100.0 - summary["event_count"] * 9.0 - summary["fatigue_risk"] * 0.33 + summary["readiness_score"] * 0.22,
                8.0,
                96.0,
            )
        ),
    }


def build_forecast_frame(filtered_df: pd.DataFrame, hours: List[float]) -> pd.DataFrame:
    if filtered_df.empty:
        return pd.DataFrame()

    recent = filtered_df.sort_values("timestamp_h").tail(min(10, len(filtered_df)))
    future_hours = np.arange(float(hours[1]), float(hours[1]) + 12.1, 1.0)

    if len(recent) >= 2:
        fatigue_slope = np.polyfit(recent["timestamp_h"], recent["fatigue_index"], 1)[0]
        sleep_slope = np.polyfit(recent["timestamp_h"], recent["sleep_quality"], 1)[0]
        stress_slope = np.polyfit(recent["timestamp_h"], recent["stress_index"], 1)[0]
    else:
        fatigue_slope = 0.0
        sleep_slope = 0.0
        stress_slope = 0.0

    last = recent.iloc[-1]
    delta_t = future_hours - float(hours[1])
    fatigue_forecast = np.clip(
        float(last["fatigue_index"]) + fatigue_slope * delta_t + 0.08 * np.sin(delta_t / 2.8),
        1.0,
        9.0,
    )
    sleep_forecast = np.clip(
        float(last["sleep_quality"]) + sleep_slope * delta_t + 0.15 * np.cos(delta_t / 3.2),
        2.0,
        9.8,
    )
    stress_forecast = np.clip(
        float(last["stress_index"]) + stress_slope * delta_t + delta_t * 0.8,
        8.0,
        98.0,
    )
    readiness_forecast = np.clip(
        100.0 - fatigue_forecast * 5.9 + sleep_forecast * 4.2 - np.maximum(stress_forecast - 42.0, 0) * 0.35,
        8.0,
        99.0,
    )

    return pd.DataFrame(
        {
            "timestamp_h": future_hours,
            "fatigue_forecast": fatigue_forecast,
            "readiness_forecast": readiness_forecast,
        }
    )
