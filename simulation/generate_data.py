import json
import numpy as np
import random

# ----------------------------
# CONFIG
# ----------------------------
TOTAL_HOURS = 72
STEP_MINUTES = 5
NUM_ASTRONAUTS = 1

steps = int((TOTAL_HOURS * 60) / STEP_MINUTES)

data = []

# ----------------------------
# SIMULATION
# ----------------------------
for astronaut_id in range(1, NUM_ASTRONAUTS + 1):

    fatigue = 0

    for i in range(steps):

        # ---- TIME ----
        time_h = i * (STEP_MINUTES / 60)

        # ---- FATIGUE MODEL (more realistic growth) ----
        fatigue += 0.08

        hour_in_day = time_h % 24
        if 22 <= hour_in_day or hour_in_day <= 6:
            fatigue -= 0.05  # recovery

        fatigue = np.clip(fatigue, 0, 10)

        # ---- PHYSIOLOGY ----
        heart_rate = 70 + fatigue * 2 + np.random.normal(0, 2)
        map_pressure = 95 + np.random.normal(0, 1)

        hrv = np.clip(100 - fatigue * 6 + np.random.normal(0, 2), 40, 100)
        sleep_quality = np.clip(8 - fatigue * 0.5 + np.random.normal(0, 1), 1, 10)

        # ---- EVENT SYSTEM (IMPROVED) ----
        event_type = "none"
        event_active = False

        # 1️⃣ Physiological triggers (MAIN LOGIC)
        if heart_rate > 78:
            event_type = "motion_sickness"
            event_active = True

        elif fatigue > 4:
            event_type = "sleep_disruption"
            event_active = True

        elif hrv < 90:
            event_type = "congestion"
            event_active = True

        # 2️⃣ Mission events (structured)
        if 30 < time_h < 35:
            event_type = "EVA"
            event_active = True
            heart_rate += 20

        # 3️⃣ Random anomalies (CRUCIAL for realism)
        if random.random() < 0.02:
            event_type = random.choice(["motion_sickness", "congestion"])
            event_active = True
            heart_rate += random.uniform(5, 15)

        # ---- RISK LOGIC ----
        at_risk = (
            fatigue > 6 or
            heart_rate > 95 or
            hrv < 60
        )

        # ---- STORE ----
        entry = {
            "astronaut_id": int(astronaut_id),
            "timestamp_h": float(round(time_h, 2)),
            "heart_rate_bpm": float(round(heart_rate, 2)),
            "map_mmhg": float(round(map_pressure, 2)),
            "fatigue_index": float(round(fatigue, 2)),
            "sleep_quality": float(round(sleep_quality, 2)),
            "hrv_proxy": float(round(hrv, 2)),
            "event_type": str(event_type),
            "event_active": bool(event_active),
            "at_risk": bool(at_risk)
        }

        data.append(entry)

# ----------------------------
# SAVE OUTPUT
# ----------------------------
with open("data/simulation_output.json", "w") as f:
    json.dump(data, f, indent=2)

print("✅ Simulation output generated successfully!")