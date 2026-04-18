import os
import subprocess
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_FOLDER = os.path.join(PROJECT_ROOT, "data")

BIOGEARS_BIN = r"C:\Biogears\bin"
EXE_PATH = os.path.join(BIOGEARS_BIN, "bg-scenario.exe")

SCENARIO_NAME = "Patient/BasicStandard.xml"

RAW_CSV = os.path.join(
    BIOGEARS_BIN,
    "Scenarios",
    "Patient",
    "basicstandardResults.csv"
)


def run_biogears():
    os.chdir(BIOGEARS_BIN)

    command = [
        EXE_PATH,
        "scenario",
        SCENARIO_NAME
    ]

    print("Running BioGears...")
    subprocess.run(command)
    print("Scenario complete.")


def convert_output():
    print("Reading raw CSV...")

    df = pd.read_csv(RAW_CSV)

    clean_df = pd.DataFrame({
        "timestamp_h": df["Time(s)"] / 3600,
        "biogears_hr_bpm": df["HeartRate(1/min)"],
        "map_mmhg": df["MeanArterialPressure(mmHg)"],
        "respiration_rate_bpm": df["RespirationRate(1/min)"],
        "spo2_pct": df["OxygenSaturation"] * 100
    })

    # -----------------------------------
    # Astronaut / Microgravity Adjustments
    # -----------------------------------

    # Slight MAP reduction in microgravity
    clean_df["map_mmhg"] = clean_df["map_mmhg"] - 5

    # Mild HR drift across mission time
    clean_df["biogears_hr_bpm"] = (
        clean_df["biogears_hr_bpm"] +
        (clean_df["timestamp_h"] / 24) * 0.3
    )

    # EVA exertion event between 24h and 26h
    eva_mask = (
        (clean_df["timestamp_h"] >= 24) &
        (clean_df["timestamp_h"] <= 26)
    )

    clean_df.loc[eva_mask, "biogears_hr_bpm"] += 25
    clean_df.loc[eva_mask, "map_mmhg"] += 8

    # -----------------------------------
    # Resample to mission timeline (0.5h)
    # -----------------------------------

    target_times = np.arange(0, 72.5, 0.5)

    rows = []

    for t in target_times:
        nearest_idx = (clean_df["timestamp_h"] - t).abs().idxmin()
        row = clean_df.loc[nearest_idx].copy()
        row["timestamp_h"] = t
        rows.append(row)

    final_df = pd.DataFrame(rows).reset_index(drop=True)

    # -----------------------------------
    # Save clean team-ready output
    # -----------------------------------

    output_path = os.path.join(DATA_FOLDER, "biogears_output.csv")
    final_df.to_csv(output_path, index=False)

    print("Saved corrected BioGears astronaut output:")
    print(output_path)


if __name__ == "__main__":
    run_biogears()
    convert_output()