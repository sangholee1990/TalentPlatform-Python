import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import joblib
import warnings
warnings.filterwarnings('ignore')

# Relaxed Target to accommodate borderline cases like W24001
TARGET_TURB = 0.253 

# --- Biological Survival Curve ---
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

# --- Data Processing ---
def load_and_resample_full(filepath, interval=1.0):
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
    resampled_df['Case'] = os.path.basename(filepath)
    return resampled_df, df

def find_target_time(ages, turbs, target=TARGET_TURB):
    if len(turbs) == 0: return np.nan
    peak_idx = np.argmax(turbs)
    after_peak_turbs = turbs[peak_idx:]
    after_peak_ages = ages[peak_idx:]
    
    target_indices = np.where(after_peak_turbs <= target)[0]
    if len(target_indices) > 0:
        return after_peak_ages[target_indices[0]]
    return np.nan

def generate_stacking_dataset(files, cutoffs, interval=1.0):
    dataset = []
    
    for f in files:
        case_name = os.path.basename(f).split('_MAIN')[0]
        res_df, orig_df = load_and_resample_full(f, interval)
        true_time = find_target_time(orig_df['Age'].values, orig_df['TURB'].values, target=TARGET_TURB)
        
        for cutoff in cutoffs:
            history = res_df[res_df['Age'] <= cutoff].copy()
            if len(history) < 2:
                continue
                
            ages_hist = history['Age'].values
            turbs_hist = history['TURB'].values
            
            # 1. Biological Curve Prediction (Bio_Pred_Time)
            pred_time_B = np.nan
            ages_future = np.arange(cutoff+interval, 200.0, interval)
            popt = fit_survival_curve(ages_hist, turbs_hist)
            if popt is not None:
                pred_curve_B = lognorm_pdf(ages_future, *popt)
                full_ages_B = np.concatenate([ages_hist, ages_future])
                full_turbs_B = np.concatenate([turbs_hist, pred_curve_B])
                pred_time_B = find_target_time(full_ages_B, full_turbs_B, target=TARGET_TURB)
            
            # 2. Extract Multivariate Features (Early Warning Indicators)
            turb_curr = history['TURB'].iloc[-1]
            turb_trend_1 = turb_curr - history['TURB'].iloc[-5] if len(history) >= 5 else 0
            turb_trend_2 = history['TURB'].iloc[-5] - history['TURB'].iloc[-10] if len(history) >= 10 else 0
            turb_accel = turb_trend_1 - turb_trend_2 # 2nd derivative
            
            press_curr = history['PRESS'].iloc[-1] if 'PRESS' in history else 0
            afoam_curr = history['AFOAM'].iloc[-1] if 'AFOAM' in history else 0
            
            ph_curr = history['pH'].iloc[-1] if 'pH' in history else 7.0
            ph_trend = ph_curr - history['pH'].iloc[-5] if 'pH' in history and len(history) >= 5 else 0
            
            base_curr = history['BASE'].iloc[-1] if 'BASE' in history else 0
            base_trend = base_curr - history['BASE'].iloc[-5] if 'BASE' in history and len(history) >= 5 else 0
            
            dataset.append({
                'Case': case_name,
                'Cutoff_Age': cutoff,
                'True_Time': true_time,
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
            })
            
    return pd.DataFrame(dataset)

def main():
    print("Loading and preparing Direct Stacking Dataset...")
    files = glob.glob('data/*.CSV')
    train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)
    
    cutoffs = [1, 3, 5, 7, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    train_df = generate_stacking_dataset(train_files, cutoffs)
    test_df = generate_stacking_dataset(test_files, cutoffs)
    
    # Drop rows without True_Time in train
    train_df = train_df.dropna(subset=['True_Time'])
    
    # Feature engineering for ML
    # If Bio_Pred_Time is NaN, fill it with a large number or 150
    train_df['Bio_Pred_Time'] = train_df['Bio_Pred_Time'].fillna(150)
    test_df['Bio_Pred_Time'] = test_df['Bio_Pred_Time'].fillna(150)
    
    features = [
        'Cutoff_Age', 'Bio_Pred_Time', 
        'TURB_current', 'TURB_trend', 'TURB_accel',
        'PRESS_current', 'AFOAM_current',
        'pH_current', 'pH_trend',
        'BASE_current', 'BASE_trend'
    ]
    
    X_train = train_df[features]
    y_train = train_df['True_Time']
    
    print("Training Direct Stacking AI Model (LightGBM)...")
    stacking_model = LGBMRegressor(n_estimators=150, max_depth=7, learning_rate=0.03, random_state=42)
    stacking_model.fit(X_train, y_train)
    
    # Save the trained model
    joblib.dump(stacking_model, 'stacking_lgbm_model.pkl')
    print("Model saved to 'stacking_lgbm_model.pkl'")
    
    # Predict on test
    X_test = test_df[features]
    test_df['Stacking_Pred_Time'] = stacking_model.predict(X_test)
    
    test_df['Error_Bio'] = test_df['Bio_Pred_Time'] - test_df['True_Time']
    test_df['Error_Stacking'] = test_df['Stacking_Pred_Time'] - test_df['True_Time']
    
    test_df.to_csv('stacking_results.csv', index=False)
    
    print("\n=== Direct Stacking Model Results (MAE) ===")
    summary = test_df.groupby('Cutoff_Age').apply(
        lambda x: pd.Series({
            'Count': len(x),
            'MAE_BioCurve': np.nanmean(np.abs(x['Error_Bio'])),
            'MAE_Stacking_AI': np.nanmean(np.abs(x['Error_Stacking']))
        })
    ).reset_index()
    print(summary)
    
    os.makedirs('plots_stacking', exist_ok=True)
    cases = test_df['Case'].unique()
    
    for case in cases:
        case_df = test_df[test_df['Case'] == case].sort_values('Cutoff_Age')
        
        plt.figure(figsize=(10, 6))
        true_time = case_df['True_Time'].iloc[0]
        if pd.notnull(true_time):
            plt.axhline(true_time, color='black', linestyle='--', linewidth=2, label=f'True Target ({true_time:.1f}h)')
            
        plt.plot(case_df['Cutoff_Age'], case_df['Bio_Pred_Time'], 
                 marker='s', linestyle='-', color='orange', alpha=0.5, label='Model B (Bio Curve)')
                 
        plt.plot(case_df['Cutoff_Age'], case_df['Stacking_Pred_Time'], 
                 marker='D', linestyle='-', color='purple', linewidth=3, label='Direct Stacking AI')
                 
        plt.title(f'Direct Stacking Convergence for {case}\n(Target TURB={TARGET_TURB})')
        plt.xlabel('Cutoff Age (hours)')
        plt.ylabel('Predicted Time to Target')
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend()
        
        all_y = case_df[['True_Time', 'Bio_Pred_Time', 'Stacking_Pred_Time']].values.flatten()
        all_y = [y for y in all_y if pd.notnull(y) and not np.isinf(y) and y < 200]
        if all_y:
            min_y = max(0, min(all_y) - 20)
            max_y = min(150, max(all_y) + 20)
            plt.ylim(min_y, max_y)
            
        plt.tight_layout()
        plt.savefig(f'plots_stacking/{case}_stacking_convergence.png')
        plt.close()

if __name__ == "__main__":
    main()
