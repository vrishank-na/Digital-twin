import os
import pandas as pd
import numpy as np

# ---------------------------------
# Paths
# ---------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_FOLDER = os.path.join(PROJECT_ROOT, "data")


# ---------------------------------
# Load BioGears baseline
# ---------------------------------
def load_biogears():
    path = os.path.join(DATA_FOLDER, "biogears_output.csv")
    return pd.read_csv(path)


# ---------------------------------
# Create base simulation data
# 3 astronauts × 144 rows = 432
# 0 to 71.5 hours (0.5 step)
# ---------------------------------
def generate_base_data(seed=42):
    np.random.seed(seed)

    time_steps = np.arange(0, 72, 0.5)   # 144 steps

    rows = []

    for astronaut_id in [1, 2, 3]:
        for t in time_steps:
            rows.append({
                "astronaut_id": astronaut_id,
                "timestamp_h": round(t, 1),
                "heart_rate_bpm": np.random.normal(72, 4),
                "map_mmhg": np.random.normal(85, 4),
                "fatigue_index": np.random.uniform(2, 4),
                "sleep_quality": np.random.uniform(7, 9),
                "hrv_proxy": np.random.normal(55, 6),
                "event_type": "none",
                "event_active": False
            })

    return pd.DataFrame(rows)


# ---------------------------------
# Event 1: Motion sickness
# Exponential(mean=18h)
# Uniform duration(2,6)
# ---------------------------------
def apply_motion_sickness(df):
    for astro in [1, 2, 3]:

        current_time = np.random.exponential(18)

        while current_time < 72:

            duration = np.random.uniform(2, 6)
            end_time = current_time + duration

            mask = (
                (df["astronaut_id"] == astro) &
                (df["timestamp_h"] >= current_time) &
                (df["timestamp_h"] < end_time)
            )

            df.loc[mask, "heart_rate_bpm"] += 15
            df.loc[mask, "fatigue_index"] += 2.0
            df.loc[mask, "sleep_quality"] -= 2.0
            df.loc[mask, "hrv_proxy"] -= 12
            df.loc[mask, "event_type"] = "motion_sickness"
            df.loc[mask, "event_active"] = True

            current_time += np.random.exponential(18)

    return df


# ---------------------------------
# Event 2: Sleep disruption
# Every 8h window Bernoulli p=0.35
# ---------------------------------
def apply_sleep_disruption(df):
    for astro in [1, 2, 3]:

        for start in range(8, 72, 8):

            if np.random.rand() < 0.35:

                end = start + 8

                mask = (
                    (df["astronaut_id"] == astro) &
                    (df["timestamp_h"] >= start) &
                    (df["timestamp_h"] < end) &
                    (df["event_type"] == "none")
                )

                df.loc[mask, "fatigue_index"] += 3.0
                df.loc[mask, "sleep_quality"] -= 3.5
                df.loc[mask, "hrv_proxy"] -= 15
                df.loc[mask, "event_type"] = "sleep_disruption"
                df.loc[mask, "event_active"] = True

    return df


# ---------------------------------
# Event 3: Sinus congestion
# Poisson(mean=1)
# ---------------------------------
def apply_sinus_congestion(df):
    for astro in [1, 2, 3]:

        n_events = np.random.poisson(1)

        for _ in range(n_events):

            start = np.random.uniform(0, 60)
            duration = np.random.uniform(12, 24)
            end = start + duration

            mask = (
                (df["astronaut_id"] == astro) &
                (df["timestamp_h"] >= start) &
                (df["timestamp_h"] < end) &
                (df["event_type"] == "none")
            )

            df.loc[mask, "sleep_quality"] -= 2.0
            df.loc[mask, "hrv_proxy"] -= 10
            df.loc[mask, "event_type"] = "sinus_congestion"
            df.loc[mask, "event_active"] = True

    return df


# ---------------------------------
# Event 4: EVA exertion
# All astronauts 24h to 26h
# ---------------------------------
def apply_eva(df):
    mask = (
        (df["timestamp_h"] >= 24) &
        (df["timestamp_h"] < 26)
    )

    df.loc[mask, "heart_rate_bpm"] += 40
    df.loc[mask, "map_mmhg"] += 8
    df.loc[mask, "fatigue_index"] += 1.5
    df.loc[mask, "event_type"] = "eva_exertion"
    df.loc[mask, "event_active"] = True

    return df


# ---------------------------------
# Merge BioGears HR
# ---------------------------------
def merge_biogears(df, bg):

    bg_small = bg[[
        "timestamp_h",
        "biogears_hr_bpm"
    ]].copy()

    bg_small["timestamp_h"] = bg_small["timestamp_h"].round(1)

    df = df.merge(
        bg_small,
        on="timestamp_h",
        how="left"
    )

    return df


# ---------------------------------
# Risk flag
# fatigue > 7 OR motion sickness
# ---------------------------------
def apply_risk(df):

    df["at_risk"] = (
        (df["fatigue_index"] > 7.0) |
        (df["event_type"] == "motion_sickness")
    )

    return df


# ---------------------------------
# Save output
# ---------------------------------
def save_output(df):
    path = os.path.join(DATA_FOLDER, "simulation_output.json")

    df.to_json(
        path,
        orient="records",
        indent=2
    )

    print("Saved:", path)
    print("Total rows:", len(df))


# ---------------------------------
# Main
# ---------------------------------
def main():

    bg = load_biogears()

    df = generate_base_data(seed=42)

    df = apply_motion_sickness(df)
    df = apply_sleep_disruption(df)
    df = apply_sinus_congestion(df)
    df = apply_eva(df)

    df = merge_biogears(df, bg)

    df = apply_risk(df)

    save_output(df)


if __name__ == "__main__":
    main()