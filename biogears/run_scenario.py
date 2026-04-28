import argparse
import csv
import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FOLDER = PROJECT_ROOT / "data"
SCENARIO_FOLDER = PROJECT_ROOT / "biogears"

DEFAULT_BIOGEARS_CORE = PROJECT_ROOT.parent / "biogears" / "core"
DEFAULT_EXE_PATH = DEFAULT_BIOGEARS_CORE / "build" / "outputs" / "Release" / "bin" / "bg-scenario"
DEFAULT_SCENARIO = SCENARIO_FOLDER / "microgravity_fast_24h.xml"
OUTPUT_CSV = DATA_FOLDER / "biogears_output.csv"
CONSOLE_LOG = SCENARIO_FOLDER / "run_scenario_console.log"


RAW_COLUMN_MAP = {
    "heart_rate_bpm": ["HeartRate(1/min)", "HeartRate"],
    "map_mmhg": ["MeanArterialPressure(mmHg)", "MeanArterialPressure"],
    "respiration_rate_bpm": ["RespirationRate(1/min)", "RespirationRate"],
    "spo2_pct": ["OxygenSaturation"],
    "cardiac_output_l_min": ["CardiacOutput(L/min)", "CardiacOutput"],
    "blood_volume_l": ["BloodVolume(L)", "BloodVolume"],
    "central_venous_pressure_mmhg": ["CentralVenousPressure(mmHg)", "CentralVenousPressure"],
    "total_body_fluid_l": ["TotalBodyFluidVolume(L)", "TotalBodyFluidVolume"],
    "urine_production_ml_min": ["UrineProductionRate(mL/min)", "UrineProductionRate"],
}


def first_present(row, names):
    for name in names:
        if name in row and row[name] not in ("", None):
            try:
                return float(row[name])
            except ValueError:
                return None
    return None


def run_biogears(exe_path, scenario_path, biogears_core, verbose=False):
    command = [str(exe_path), str(scenario_path), "-q"]
    print("Running BioGears:")
    print(" ".join(command))
    if verbose:
        subprocess.run(command, cwd=biogears_core, check=True)
        return

    with CONSOLE_LOG.open("w", encoding="utf-8") as log_file:
        subprocess.run(
            command,
            cwd=biogears_core,
            check=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    print(f"BioGears console log saved to: {CONSOLE_LOG}")


def find_raw_results(scenario_path):
    scenario_stem = scenario_path.stem
    candidates = [
        scenario_path.with_name(f"{scenario_stem}Results.csv"),
        scenario_path.with_name(f"{scenario_stem.lower()}Results.csv"),
        scenario_path.with_name(f"{scenario_stem.lower()}results.csv"),
    ]
    existing = [path for path in candidates if path.exists()]
    if existing:
        return max(existing, key=lambda path: path.stat().st_mtime)

    matches = list(scenario_path.parent.glob("*Results.csv")) + list(scenario_path.parent.glob("*results.csv"))
    if not matches:
        raise FileNotFoundError(
            f"No BioGears results CSV found beside {scenario_path}. "
            "Run the scenario first or pass --raw-results."
        )
    return max(matches, key=lambda path: path.stat().st_mtime)


def read_raw_rows(raw_results):
    with raw_results.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = []
        for row in reader:
            if "Time(s)" in row:
                try:
                    timestamp_h = float(row["Time(s)"]) / 3600.0
                except (TypeError, ValueError):
                    continue
            elif "timestamp_h" in row:
                try:
                    timestamp_h = float(row["timestamp_h"])
                except (TypeError, ValueError):
                    continue
            else:
                continue

            clean_row = {"timestamp_h": timestamp_h}
            for output_name, raw_names in RAW_COLUMN_MAP.items():
                value = first_present(row, [output_name] + raw_names)
                if output_name == "spo2_pct" and value is not None and value <= 1.5:
                    value *= 100.0
                clean_row[output_name] = value

            if clean_row.get("heart_rate_bpm") is None:
                clean_row["heart_rate_bpm"] = first_present(row, ["biogears_hr_bpm"])

            if clean_row.get("heart_rate_bpm") is not None:
                clean_row["biogears_hr_bpm"] = clean_row["heart_rate_bpm"]
                rows.append(clean_row)
        return rows


def nearest_resample(rows, step_h=1.0, max_hours=None):
    if not rows:
        return []

    rows = sorted(rows, key=lambda row: row["timestamp_h"])
    end_h = max_hours if max_hours is not None else rows[-1]["timestamp_h"]
    target = 0.0
    output = []
    index = 0

    while target <= end_h + 1e-9:
        while (
            index + 1 < len(rows)
            and abs(rows[index + 1]["timestamp_h"] - target) <= abs(rows[index]["timestamp_h"] - target)
        ):
            index += 1
        sample = dict(rows[index])
        sample["timestamp_h"] = round(target, 3)
        output.append(sample)
        target += step_h
    return output


def convert_output(raw_results, max_hours=None, step_hours=1.0):
    print(f"Reading raw BioGears CSV: {raw_results}")
    rows = read_raw_rows(raw_results)
    final_rows = nearest_resample(rows, step_h=step_hours, max_hours=max_hours)

    if not final_rows:
        raise RuntimeError("BioGears output had no usable heart-rate rows.")

    DATA_FOLDER.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp_h",
        "biogears_hr_bpm",
        "heart_rate_bpm",
        "map_mmhg",
        "respiration_rate_bpm",
        "spo2_pct",
        "cardiac_output_l_min",
        "blood_volume_l",
        "central_venous_pressure_mmhg",
        "total_body_fluid_l",
        "urine_production_ml_min",
    ]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in final_rows:
            writer.writerow({name: row.get(name) for name in fieldnames})

    print(f"Saved dashboard-ready BioGears output: {OUTPUT_CSV}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run a BioGears scenario and export dashboard-ready CSV.")
    parser.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
    parser.add_argument("--exe", type=Path, default=Path(os.environ.get("BIOGEARS_SCENARIO_EXE", DEFAULT_EXE_PATH)))
    parser.add_argument("--core", type=Path, default=Path(os.environ.get("BIOGEARS_CORE", DEFAULT_BIOGEARS_CORE)))
    parser.add_argument("--raw-results", type=Path, default=None)
    parser.add_argument("--convert-only", action="store_true")
    parser.add_argument("--max-hours", type=float, default=None)
    parser.add_argument("--step-hours", type=float, default=1.0)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    scenario_path = args.scenario.expanduser().resolve()
    exe_path = args.exe.expanduser().resolve()
    core_path = args.core.expanduser().resolve()

    if not scenario_path.exists():
        raise FileNotFoundError(f"Scenario not found: {scenario_path}")
    if not args.convert_only and not exe_path.exists():
        raise FileNotFoundError(f"BioGears executable not found: {exe_path}")
    if not args.convert_only and not core_path.exists():
        raise FileNotFoundError(f"BioGears core directory not found: {core_path}")

    if not args.convert_only:
        run_biogears(exe_path, scenario_path, core_path, verbose=args.verbose)

    raw_results = args.raw_results.expanduser().resolve() if args.raw_results else find_raw_results(scenario_path)
    convert_output(raw_results, max_hours=args.max_hours, step_hours=args.step_hours)


if __name__ == "__main__":
    main()
