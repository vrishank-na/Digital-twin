import subprocess
import pandas as pd


def run_biogears():
    print("Running BioGears scenario...")

    subprocess.run(
    "bg-cli.exe scenario Patient\\microgravity.xml",
    shell=True,
    cwd="C:\\Biogears\\bin"
)

    print("BioGears run completed.")


def process_output():
    # ✅ Your actual CSV path
    raw_csv_path = "C:\\Biogears\\bin\\Scenarios\\Patient\\basicstandardResults.csv"

    df = pd.read_csv(raw_csv_path)

    print("\nColumns in BioGears CSV:")
    print(df.columns)

    # ✅ Keep only needed columns
    df = df[[
        "Time(s)",
        "HeartRate(1/min)",
        "MeanArterialPressure(mmHg)"
    ]]

    # ✅ Rename columns (match project schema)
    df = df.rename(columns={
        "Time(s)": "timestamp_h",
        "HeartRate(1/min)": "heart_rate_bpm",
        "MeanArterialPressure(mmHg)": "map_mmhg"
    })

    # ✅ Convert seconds → hours
    df["timestamp_h"] = df["timestamp_h"] / 3600

    # ✅ Save to shared folder
    df.to_csv("data/biogears_output.csv", index=False)

    print("\nCleaned CSV saved to /data/biogears_output.csv")

    return df


if __name__ == "__main__":
    run_biogears()
    process_output()