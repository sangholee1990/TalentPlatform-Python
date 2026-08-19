import sys
import os
import joblib
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# Re-use the lognorm_pdf from the main script
def lognorm_pdf(x, A, mu, sigma, C):
    return C + A * np.exp(- (np.log(x + 1e-5) - mu)**2 / (2 * sigma**2))

def fit_survival_curve(ages, turbs):
    try:
        p0 = [turbs.max(), np.log(ages[np.argmax(turbs)]+1e-5), 1.0, 0.05]
        bounds = ([0, -np.inf, 0, 0], [np.inf, np.inf, np.inf, np.inf])
        popt, _ = curve_fit(lognorm_pdf, ages, turbs, p0=p0, bounds=bounds, maxfev=10000)
        return popt
    except:
        return None

def find_target_time(ages, turbs, target=0.253):
    if len(turbs) == 0: return np.nan
    peak_idx = np.argmax(turbs)
    after_peak_turbs = turbs[peak_idx:]
    after_peak_ages = ages[peak_idx:]
    
    target_indices = np.where(after_peak_turbs <= target)[0]
    if len(target_indices) > 0:
        return after_peak_ages[target_indices[0]]
    return np.nan

def load_and_resample(filepath, interval=1.0):
    df = pd.read_csv(filepath, encoding='cp949', skiprows=[1,2])
    candidate_cols = ['Age', 'TURB', 'PRESS', 'AFOAM', 'BASE', 'pH', 'TEMP']
    cols_to_use = [c for c in candidate_cols if c in df.columns]
    df = df[cols_to_use].dropna(subset=['Age', 'TURB'])
    
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(axis=1, how='all').sort_values('Age').reset_index(drop=True)
    
    max_age = int(np.ceil(df['Age'].max()))
    ages_grid = np.arange(0, max_age + interval, interval)
    
    resampled_data = {'Age': ages_grid}
    for col in df.columns:
        if col != 'Age':
            s = df[col].ffill().bfill()
            if s.isnull().all():
                continue
            f_interp = interp1d(df['Age'], s, bounds_error=False, fill_value=(s.iloc[0], s.iloc[-1]))
            resampled_data[col] = f_interp(ages_grid)
            
    resampled_df = pd.DataFrame(resampled_data)
    return resampled_df

def predict_time_to_target(filepath, cutoff_age=None, model_path='stacking_lgbm_model.pkl'):
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return None
        
    model = joblib.load(model_path)
    res_df = load_and_resample(filepath)
    
    if cutoff_age is None:
        cutoff_age = res_df['Age'].max()
    
    history = res_df[res_df['Age'] <= cutoff_age].copy()
    if len(history) < 2:
        print(f"Error: Not enough data points before cutoff {cutoff_age}h.")
        return None
        
    ages_hist = history['Age'].values
    turbs_hist = history['TURB'].values
    
    # 1. Biological Curve Prediction (Bio_Pred_Time)
    pred_time_B = 150.0 # Default if fail
    ages_future = np.arange(cutoff_age+1.0, 200.0, 1.0)
    popt = fit_survival_curve(ages_hist, turbs_hist)
    if popt is not None:
        pred_curve_B = lognorm_pdf(ages_future, *popt)
        full_ages_B = np.concatenate([ages_hist, ages_future])
        full_turbs_B = np.concatenate([turbs_hist, pred_curve_B])
        pred_t = find_target_time(full_ages_B, full_turbs_B, target=0.253)
        if pd.notnull(pred_t):
            pred_time_B = pred_t
            
    # 2. Extract Multivariate Features (Early Warning Indicators)
    turb_curr = history['TURB'].iloc[-1]
    turb_trend_1 = turb_curr - history['TURB'].iloc[-5] if len(history) >= 5 else 0
    turb_trend_2 = history['TURB'].iloc[-5] - history['TURB'].iloc[-10] if len(history) >= 10 else 0
    turb_accel = turb_trend_1 - turb_trend_2
    
    press_curr = history['PRESS'].iloc[-1] if 'PRESS' in history else 0
    afoam_curr = history['AFOAM'].iloc[-1] if 'AFOAM' in history else 0
    
    ph_curr = history['pH'].iloc[-1] if 'pH' in history else 7.0
    ph_trend = ph_curr - history['pH'].iloc[-5] if 'pH' in history and len(history) >= 5 else 0
    
    base_curr = history['BASE'].iloc[-1] if 'BASE' in history else 0
    base_trend = base_curr - history['BASE'].iloc[-5] if 'BASE' in history and len(history) >= 5 else 0
    
    # Create feature dataframe in same order as training
    features = [
        'Cutoff_Age', 'Bio_Pred_Time', 
        'TURB_current', 'TURB_trend', 'TURB_accel',
        'PRESS_current', 'AFOAM_current',
        'pH_current', 'pH_trend',
        'BASE_current', 'BASE_trend'
    ]
    
    X_test = pd.DataFrame([{
        'Cutoff_Age': cutoff_age,
        'Bio_Pred_Time': pred_time_B,
        'TURB_current': turb_curr,
        'TURB_trend': turb_trend_1,
        'TURB_accel': turb_accel,
        'PRESS_current': press_curr,
        'AFOAM_current': afoam_curr,
        'pH_current': ph_curr,
        'pH_trend': ph_trend,
        'BASE_current': base_curr,
        'BASE_trend': base_trend
    }], columns=features)
    
    # Predict
    predicted_time = model.predict(X_test)[0]
    
    print(f"--- Prediction Results for {os.path.basename(filepath)} at {cutoff_age}h ---")
    print(f"Loaded Stacking AI Model: {model_path}")
    print(f"Input Features:")
    for f in features:
        print(f"  {f}: {X_test.iloc[0][f]:.4f}")
    print(f"\n=> Predicted Time to TURB=0.25: {predicted_time:.2f} hours")
    return predicted_time

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_stacking_model.py <path_to_csv> [cutoff_age]")
        sys.exit(1)
        
    filepath = sys.argv[1]
    cutoff_age = float(sys.argv[2]) if len(sys.argv) > 2 else None
    
    predict_time_to_target(filepath, cutoff_age)
