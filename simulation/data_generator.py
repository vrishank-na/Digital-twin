import pandas as pd
import numpy as np

def generate_data(n_astronauts=3, n_hours=72, timestep_h=0.5, seed=42):
    # This is a stub for T1 to import. 
    # Logic for distributions (Normal, Beta, Log-normal) goes here on Day 2.
    columns = [
        'astronaut_id', 'timestamp_h', 'heart_rate_bpm', 'map_mmhg', 
        'fatigue_index', 'sleep_quality', 'hrv_proxy', 'event_type', 
        'event_active', 'at_risk', 'biogears_hr_bpm'
    ]
    return pd.DataFrame(columns=columns)