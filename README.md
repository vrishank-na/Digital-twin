# Astronaut Health Digital Twin

[![Dashboard Preview](https://via.placeholder.com/800x400/4f46e5/ffffff?text=Astronaut+Health+Digital+Twin)](https://github.com/vishwambhararh/Digital-twin)

## Project Overview

**Astronaut Health Digital Twin** is a comprehensive physiological simulation platform modeling astronaut health during long-duration spaceflight. It integrates:

- **Discrete-event simulation** for mission stressors (EVA, sleep disruption, motion sickness)
- **BioGears engine** for high-fidelity physiological modeling under microgravity analogs
- **Monte Carlo risk projection** for fatigue/recovery forecasting
- **Real-time dashboard** with crew resilience radar, phase-aware analytics, and twin synchronization
- **3D visualization** integration for immersive mission context

**Core Capabilities:**
- Multi-crew fatigue heatmap tracking
- BioGears scenario execution (microgravity cabin + countermeasures)
- Mission phase timeline with event gantt overlay
- Physiological twin sync (simulation vs. BioGears outputs)
- Risk probability forecasting from 100 Monte Carlo runs

## Project Structure

```
Digital-twin/
├── README.md                    # This document
├── requirements.txt             # Python dependencies (dash, pandas, plotly, numpy)
├── index.html                   # BioGears microgravity results visualizer
├── dashboard.py                 # Dashboard launcher
├── dashboard_support.py         # Core simulation logic & BioGears integration
├── TODO.md                      # Development roadmap
├── data/
│   ├── analytics_summary.json   # AI-generated mission insights
│   ├── simulation_output.json   # Synthetic mission telemetry
│   └── biogears_output.csv      # Processed BioGears results
├── analytics/
│   ├── __init__.py
│   └── monte_carlo.py           # 100-run fatigue/recovery Monte Carlo
├── biogears/                    # BioGears physiological scenarios
│   ├── microgravity_scenario.xml     # Primary 7-day cabin analog (DETAILED BELOW)
│   ├── microgravity_fast_24h.xml     # 24-hour accelerated scenario
│   ├── weekly_nutrition_sleep_exercise.xml # Full 168h weekly schedule
│   ├── microgravity_scenarioResults.csv   # 100k+ samples physiological output
│   ├── README.txt
│   └── run_scenario.py          # BioGears scenario executor
├── dashboard/                   # Plotly Dash mission control UI
│   ├── app.py                   # Main dashboard (15+ interactive charts)
│   ├── dashboard_support.py     # Duplicate support utils
│   └── assets/
│       ├── dashboard.css        # Mission control styling
│       └── dashboard_expansion.css
├── simulation/                  # Event-driven astronaut modeling
│   ├── __init__.py
│   ├── event_simulator.py       # EVA, sleep disruption, exercise events
│   └── generate_data.py         # 72h synthetic mission data generator
└── viz3d/                      # 3D astronaut visualization
    ├── Astronaut.glb            # 3D astronaut model
    └── viewer.html              # Three.js mission timeline viewer
```

## Key Features

### 1. **Mission Control Dashboard** (`dashboard/app.py`)
- **15+ interactive Plotly charts**: Physiology timeline, crew heatmap, event gantt, resilience radar
- **Real-time twin sync**: Compares discrete simulation vs. BioGears physiology
- **Mission phase awareness**: Launch, EVA windows, sleep cycles
- **Risk forecasting**: Monte Carlo fatigue probability + recovery horizon
- **Crew profiles**: 3 astronauts with specialties, flight hours, shift patterns
- **BioGears integration**: Live vitals from microgravity CSV (HR, MAP, SpO2, respiration)

**Launch:** `cd Digital-twin && python dashboard.py`

### 2. **BioGears Microgravity Scenarios** (`biogears/`)
Primary scenario: **`microgravity_scenario.xml`** (detailed below) produces `microgravity_scenarioResults.csv` with 100k+ physiological samples over 168 hours.

### 3. **3D Visualization** (`viz3d/viewer.html`)
- Astronaut.glb model synchronized with dashboard timeline
- Launch button opens immersive 3D twin viewer
- Mission hour messaging between dashboard ↔ 3D viewer

### 4. **Monte Carlo Analytics** (`analytics/monte_carlo.py`)
```
"monte_carlo_runs": 100
"risk_prob_fatigue": 0.18 (18%)
"risk_prob_motion_sickness": 0.66 (66%)
"mean_recovery_time_h": 3.97h
"p90_recovery_time_h": 5.66h
"cumulative_fatigue_load_mean": 611.85
```

**AI Insights** (`data/analytics_summary.json`):
> "Fatigue risk depends on balance between workload and recovery. Motion sickness remains an important early-mission uncertainty."

## **Microgravity XML Scenarios** (Dedicated Section)

### Primary: `biogears/microgravity_scenario.xml`
**Name:** `MicrogravityWeeklyNutritionSleepExercise`  
**Description:** One-week spacecraft cabin analog with scheduled nutrition, sleep, and exercise countermeasures. BioGears lacks direct gravity modeling; uses controllable cabin/activity profile.

**Key Parameters:**
```
Patient: StandardMale.xml
Environment: SpacecraftCabinAnalog
- Temperature: 22°C, Pressure: 760 mmHg, Humidity: 45%
- Cabin gases: O2=21.0%, CO2=0.5%, N2=78.5%
- Clothing: 0.3 clo (spacesuit/light garment analog)
```

**Data Requests** (50+ physiological channels sampled @0.0017 Hz):
```
Core Vitals: HeartRate(1/min), MeanArterialPressure(mmHg), RespirationRate(1/min), 
             OxygenSaturation(unitless), CoreTemperature(degC)

Hemodynamics: CardiacOutput(L/min), BloodVolume(L), CentralVenousPressure(mmHg)

Metabolism: TotalMetabolicRate(kcal/day), O2Consumption(mL/min), CO2Production(mL/min)

Fluids: TotalBodyFluidVolume(L), UrineProductionRate(mL/min), GlomerularFiltrationRate(mL/min)

Workload: TotalWorkRateLevel(unitless), AchievedExerciseLevel(unitless), FatigueLevel(unitless)

Sleep: SleepTime(s), WakeTime(s), BiologicalDebt(unitless), AttentionLapses(unitless)

Nutrition: StomachContents-{Carb/Fat/Protein/Water}(g/mL), LiverGlycogen(g), MuscleGlycogen(g)
```

**7-Day Schedule** (168 simulated hours):
```
Daily Pattern (06:30 wake - 22:30 sleep):
├── 07:30 Breakfast (70g carb, 18g fat, 25g protein, 500mL water)
├── 12:30 Lunch (95g carb, 28g fat, 35g protein, 600mL water)
├── 18:00 Exercise (45min @ 45% intensity countermeasure)
└── 19:00 Dinner (85g carb, 25g fat, 35g protein, 600mL water)
```

**Output:** `microgravity_scenarioResults.csv` → 100k+ rows integrated into dashboard BioGears panel.

### Additional Scenarios:
- **`microgravity_fast_24h.xml`**: Accelerated 24h cabin analog
- **`weekly_nutrition_sleep_exercise.xml`**: Identical 168h reference

**Execution:** `python biogears/run_scenario.py microgravity_scenario.xml`

## Quick Start

```bash
# 1. Install dependencies
cd Digital-twin
pip install -r requirements.txt

# 2. Launch dashboard (auto-loads BioGears + simulation data)
python dashboard.py

# 3. Open BioGears visualizer
open index.html

# 4. Launch 3D viewer (from dashboard button)
# Opens synchronized astronaut.glb viewer
```

## Screenshots

1. **Mission Control Dashboard**: [Full-screen physiology fusion](https://via.placeholder.com/1200x800/1e293b/ffffff?text=Dashboard)
2. **Crew Fatigue Heatmap**: Multi-astronaut risk patterns
3. **BioGears Vitals**: Live microgravity scenario overlay
4. **Resilience Radar**: 6-factor crew readiness
5. **Event Gantt**: EVA/sleep disruption timeline

## Development

**Active Tasks** (`TODO.md`):
```
1. Enhanced BioGears field mapping (HR→heart_rate_bpm, SpO2→spo2_pct*100)
2. Remove CSV fallbacks in app.py 
3. Expand biogears vitals graph (MAP, respiration, fluids)
4. Test integration end-to-end
5. Polish 3D sync protocol
```

**Tech Stack:**
```
Frontend: Plotly Dash, Chart.js, Three.js (GLTF)
Backend: Python 3.10+, pandas, numpy
Engine: BioGears 6.3.0-beta (XML scenarios)
Data: JSON/CSV (100k+ physiological samples)
```

## Current Mission State

```
Analytics Summary:
├── Fatigue Risk: 18% (Monte Carlo p50)
├── Motion Sickness Risk: 66% (early mission)
├── Mean Recovery: 3.97h (p90: 5.66h)
└── Cumulative Fatigue Load: 611.85 ± 183.72
```

## Contributing

1. Run BioGears scenarios: `python biogears/run_scenario.py`
2. Enhance analytics: Edit `analytics/monte_carlo.py`
3. Add crew profiles: `dashboard_support.py → build_crew_profiles()`
4. Extend events: `simulation/event_simulator.py`



