import os
import json
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_FOLDER = os.path.join(PROJECT_ROOT, "data")


def run_single_sim():
    # Initial fatigue state
    fatigue = np.random.uniform(2, 4)
    fatigue_trace = []

    had_motion_sickness = False
    recovery_times = []

    for t in np.arange(0, 72, 0.5):

        # ---------------------------------
        # Natural workload + natural recovery
        # ---------------------------------
        fatigue += np.random.uniform(0.00, 0.025)
        fatigue -= np.random.uniform(0.00, 0.02)
        fatigue = max(1.5, fatigue)

        # ---------------------------------
        # Night / sleep recovery window
        # ---------------------------------
        hour = t % 24

        if hour >= 22 or hour <= 6:
            fatigue -= np.random.uniform(0.02, 0.08)
            fatigue = max(1.5, fatigue)

        # ---------------------------------
        # Motion sickness random chance
        # ---------------------------------
        if np.random.rand() < 0.007:
            fatigue += 1.4
            had_motion_sickness = True
            recovery_times.append(np.random.uniform(2, 6))

        # ---------------------------------
        # Sleep disruption every 8h block
        # ---------------------------------
        if t % 8 == 0 and t > 0:
            if np.random.rand() < 0.35:
                fatigue += 1.0

        # ---------------------------------
        # EVA exertion event
        # ---------------------------------
        if 24 <= t < 26:
            fatigue += 0.2

        fatigue_trace.append(fatigue)

    # ---------------------------------
    # Risk Logic
    # ---------------------------------
    risk_fatigue = max(fatigue_trace) > 8.0

    cumulative_load = float(np.sum(fatigue_trace))

    return {
        "risk_fatigue": risk_fatigue,
        "motion_sickness": had_motion_sickness,
        "recovery_times": recovery_times,
        "cumulative_load": cumulative_load
    }


def monte_carlo(n_runs=100):
    fatigue_risks = 0
    motion_cases = 0

    all_recovery = []
    all_loads = []

    for _ in range(n_runs):

        result = run_single_sim()

        if result["risk_fatigue"]:
            fatigue_risks += 1

        if result["motion_sickness"]:
            motion_cases += 1

        all_recovery.extend(result["recovery_times"])
        all_loads.append(result["cumulative_load"])

    summary = {
        "monte_carlo_runs": n_runs,

        "risk_prob_fatigue":
            round(fatigue_risks / n_runs, 3),

        "risk_prob_motion_sickness":
            round(motion_cases / n_runs, 3),

        "mean_recovery_time_h":
            round(float(np.mean(all_recovery)), 2)
            if all_recovery else 0,

        "p90_recovery_time_h":
            round(float(np.percentile(all_recovery, 90)), 2)
            if all_recovery else 0,

        "cumulative_fatigue_load_mean":
            round(float(np.mean(all_loads)), 2),

        "cumulative_fatigue_load_sd":
            round(float(np.std(all_loads)), 2),

        "conclusions": [
            "Fatigue risk depends on balance between workload and recovery.",
            "Motion sickness remains an important early-mission uncertainty."
        ]
    }

    return summary


def save_output(summary):
    path = os.path.join(
        DATA_FOLDER,
        "monte_carlo_summary.json"
    )

    with open(path, "w") as f:
        json.dump(summary, f, indent=2)

    print("Saved:", path)


def main():
    summary = monte_carlo(n_runs=100)
    save_output(summary)


if __name__ == "__main__":
    main()