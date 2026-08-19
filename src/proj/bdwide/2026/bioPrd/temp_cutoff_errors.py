import pandas as pd
import numpy as np
import io
import contextlib
import os
import sys

# Ensure the module can be imported
sys.path.append('c:\\SYSTEMS\\PROG\\PYTHON\\TalentPlatform-Python\\src\\proj\\bdwide\\2026\\bioPrd')

from predict_stacking_model import predict_time_to_target, load_and_resample

def get_actual(filepath):
    res_df = load_and_resample(filepath)
    turbs = res_df['TURB'].values
    ages = res_df['Age'].values
    if len(turbs) == 0: return np.nan
    peak_idx = np.argmax(turbs)
    after_peak_turbs = turbs[peak_idx:]
    after_peak_ages = ages[peak_idx:]
    
    target_indices = np.where(after_peak_turbs <= 0.253)[0]
    if len(target_indices) > 0:
        return after_peak_ages[target_indices[0]]
    return np.nan

files = [
    'data/B_8000765_W24009_MAIN.CSV',
    'data/B_8000765_W25004_MAIN.CSV',
    'data/B_8000765_W23008_MAIN.CSV',
    'data/B_8000765_W24001_MAIN.CSV',
    'data/B_8000765_W25007_MAIN.CSV',
    'data/B_8000765_W25003_MAIN.CSV'
]
cutoffs = [1, 3, 5, 7, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

for f in files:
    try:
        actual = get_actual(f)
        if pd.isna(actual):
            continue
            
        print(f"\n=======================================================")
        print(f" Case: {os.path.basename(f)}")
        print(f"=======================================================")
        print("Cutoff(h) | Actual(h) | Predicted(h) | Error(h)")
        print("-" * 55)
        
        for c in cutoffs:
            if c > actual:
                continue
                
            f_io = io.StringIO()
            with contextlib.redirect_stdout(f_io):
                pred = predict_time_to_target(f, c)
                
            if pred is not None:
                err = abs(pred - actual)
                print(f"{c:9d} | {actual:9.2f} | {pred:12.2f} | {err:8.2f}")
            else:
                print(f"{c:9d} | {actual:9.2f} | {'N/A':>12} | {'N/A':>8}")
    except Exception as e:
        print(f"Failed for {f}: {e}")
