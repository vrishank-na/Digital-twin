# Digital Twin BioGears Microgravity Integration
## Progress: 0/5 ✅

### 1. [ ] Create enhanced field mapping in dashboard_support.py
   - Map ALL microgravity_scenarioResults.csv columns (HeartRate, MAP, SpO2, CO2 prod, etc.)
   - Ensure spo2_pct= OxygenSaturation*100, map_mmhg=MeanArterialPressure(mmHg), etc.
   - Add missing derived fields for dashboard KPIs (hrv_proxy, recovery_time_h, etc.)

### 2. [ ] Edit app.py - Remove biogears_output.csv fallback
   - Delete BIOGEARS_FILE definition
   - Make load_biogears_data() use ONLY MICROGRAVITY_RESULTS_FILE
   - Update empty message to reference microgravity CSV only

### 3. [ ] Update BioGears vitals graph in app.py
   - Ensure build_biogears_figure() plots all available fields (HR, MAP, RR, SpO2, etc.)
   - Add traces for new mapped fields

### 4. [ ] Test data loading
   - Run `cd Digital-twin && python dashboard/app.py`
   - Verify BioGears panel shows data (no fallback message)
   - Check twin sync computes, graphs plot microgravity HR/etc.

### 5. [ ] Completion
   - Update TODO progress
   - attempt_completion

