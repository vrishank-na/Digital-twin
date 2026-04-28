BioGears Integration Module

Purpose:
- Run BioGears scenarios
- Extract physiological outputs
- Convert to shared team schema

Main file:
run_scenario.py

Scenarios:
- microgravity_scenario.xml
  BioGears-valid spacecraft-cabin analog with scheduled meals, sleep, and daily exercise countermeasures.
- microgravity_fast_24h.xml
  Fast default dashboard scenario. Same cabin/day structure, but only 24 simulated hours.
- weekly_nutrition_sleep_exercise.xml
  BioGears-valid baseline weekly lifestyle scenario without the cabin analog.

Run and save BioGears output:
1. From ~/VSCode/Digital-twin:
   python3 biogears/run_scenario.py

2. The script searches for the generated BioGears Results CSV beside the scenario and writes:
   data/biogears_output.csv

   The dashboard CSV is sampled hourly by default. BioGears may still write internal progress messages every simulated 60s; the runner captures that console noise into:
   biogears/run_scenario_console.log

3. If BioGears has already produced a raw Results CSV, convert only:
   python3 biogears/run_scenario.py --convert-only --raw-results biogears/microgravity_scenarioResults.csv

4. Dashboard output is hourly by default. To change it during conversion:
   python3 biogears/run_scenario.py --convert-only --raw-results data/biogears_output.csv --step-hours 1

Notes:
- BioGears does not expose a gravity scalar in this scenario schema. The microgravity file is therefore a cabin/activity analog, not a direct 0g physics model.
- A full 168-hour BioGears run can be slow. Use --max-hours only during conversion; it does not shorten the underlying BioGears simulation.
- Use --verbose only if you want to see BioGears' internal per-minute progress messages in the terminal.
- Run the full week only when needed:
  python3 biogears/run_scenario.py --scenario biogears/microgravity_scenario.xml
