!pip install neuralprophet xgboost pandas numpy matplotlib 
import pandas as pd
import numpy as np
import xgboost as xgb
from neuralprophet import NeuralProphet
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
import json
import os
import warnings
from google.colab import files

warnings.filterwarnings("ignore")

VAULT_FILE = "flux_vault_state.json"  # Persistent vault storage file

# JSON-based
def load_vault() -> dict:
    """Load vault state from JSON file, or create a fresh one."""
    if os.path.exists(VAULT_FILE):
        with open(VAULT_FILE, "r") as f:
            state = json.load(f)
        print(f" Vault loaded — Balance: ₹{state['balance']:,.2f} | Runs: {state['total_runs']}")
        return state
    else:
        print(" No vault found. Initializing fresh vault at ₹0.00")
        return {"balance": 0.0, "total_runs": 0, "history": []}

def save_vault(state: dict):
    """Persist vault state back to JSON."""
    with open(VAULT_FILE, "w") as f:
        json.dump(state, f, indent=2)
    print(f" Vault saved — Balance: ₹{state['balance']:,.2f}")


# 1. FILE UPLOAD & DATA INGESTION


REQUIRED_COLUMNS = ['Net_Income', 'Fuel_or_Expense', 'Loan_Repayment', 'Emergency_Expense']

print("Starting Flux AI-CFO Engine v2.0 (Hybrid Edition)...")
print("Please upload your CSV dataset...")

try:
    uploaded = files.upload()
    if not uploaded:
        raise ValueError("No file was uploaded. Please re-run and select a CSV file.")

    filename = list(uploaded.keys())[0]
    if not filename.lower().endswith('.csv'):
        raise ValueError(f"Expected a .csv file, but got: '{filename}'. Please upload a CSV.")

    df_raw = pd.read_csv(io.BytesIO(uploaded[filename]))
    print(f"Successfully loaded: '{filename}' — {len(df_raw)} rows, {len(df_raw.columns)} columns")

    # Validate required columns
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df_raw.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}\n"
            f"Your CSV has: {list(df_raw.columns)}\n"
            f"Required: {REQUIRED_COLUMNS}"
        )

    # Check for sufficient data
    if len(df_raw) < 30:
        raise ValueError(f"Need at least 30 rows of data for forecasting. Your file has {len(df_raw)} rows.")

    # Check for nulls in key columns
    null_counts = df_raw[REQUIRED_COLUMNS].isnull().sum()
    if null_counts.any():
        print(f"  Warning: Null values detected:\n{null_counts[null_counts > 0]}")
        print("   Filling nulls with column medians...")
        df_raw[REQUIRED_COLUMNS] = df_raw[REQUIRED_COLUMNS].fillna(df_raw[REQUIRED_COLUMNS].median())

except ValueError as e:
    print(f"\n DATA ERROR: {e}")
    raise SystemExit("Fix the issue above and re-run the cell.")
except Exception as e:
    print(f"\n UNEXPECTED ERROR during file upload: {e}")
    raise

# 2. FEATURE ENGINEERING


worker_df = df_raw.copy()
worker_df['y']                = worker_df['Net_Income']        / 30
worker_df['Fuel_or_Expense']  = worker_df['Fuel_or_Expense']   / 30
worker_df['Loan_Repayment']   = worker_df['Loan_Repayment']    / 30
worker_df['Emergency_Expense']= worker_df['Emergency_Expense'] / 30

essential_expenses = worker_df[['Fuel_or_Expense', 'Loan_Repayment', 'Emergency_Expense']].mean().sum() * 30

# Assign dates from the end of today backwards
worker_df['ds'] = pd.date_range(end=pd.Timestamp.today(), periods=len(worker_df), freq='D')
prophet_df = worker_df[['ds', 'y']].copy()


# 3. TRAIN / TEST SPLIT


split_idx    = int(len(prophet_df) * 0.80)
train_df     = prophet_df.iloc[:split_idx].copy().reset_index(drop=True)
test_df      = prophet_df.iloc[split_idx:].copy().reset_index(drop=True)

print(f"\n Train/Test Split: {len(train_df)} train rows / {len(test_df)} test rows (80/20)")

# 4. BASE MODEL — NeuralProphet


print(" Training NeuralProphet base model...")

try:
    m = NeuralProphet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
        learning_rate=0.1,
        epochs=150,
        quantiles=[0.10, 0.90],   # ← confidence interval bands
    )
    m.add_country_holidays(country_name='IN')
    m.fit(train_df, freq="D")
except Exception as e:
    print(f" NeuralProphet training failed: {e}")
    raise

# Predict on FULL dataset (train + test) to get residuals and test-set evaluation
full_forecast       = m.predict(prophet_df)
yhat_col            = [c for c in full_forecast.columns if c.startswith('yhat1') and 'quantile' not in c][0]
yhat_low_col        = [c for c in full_forecast.columns if 'quantile 0.1' in c]
yhat_high_col       = [c for c in full_forecast.columns if 'quantile 0.9' in c]

has_ci = bool(yhat_low_col and yhat_high_col)
yhat_low_col  = yhat_low_col[0]  if has_ci else None
yhat_high_col = yhat_high_col[0] if has_ci else None

prophet_df['np_yhat'] = full_forecast[yhat_col].values
prophet_df['residual'] = prophet_df['y'] - prophet_df['np_yhat']

if has_ci:
    prophet_df['np_low']  = full_forecast[yhat_low_col].values
    prophet_df['np_high'] = full_forecast[yhat_high_col].values


# 5. RESIDUAL FEATURES + XGBoost
print("⚙️  Engineering features for residual boosting...")

prophet_df['day_of_week'] = prophet_df['ds'].dt.dayofweek
prophet_df['is_weekend']  = (prophet_df['day_of_week'] >= 5).astype(int)
prophet_df['y_roll_7']    = prophet_df['y'].shift(1).rolling(window=7).mean()

features = ['day_of_week', 'is_weekend', 'y_roll_7']

# XGBoost trained ONLY on the train portion
xgb_train_df = prophet_df.iloc[:split_idx].dropna().copy()
xgb_test_df  = prophet_df.iloc[split_idx:].dropna().copy()

print(" Training XGBoost on residuals (train set only)...")
try:
    xgb_model = xgb.XGBRegressor(
        n_estimators=100, learning_rate=0.05,
        max_depth=4, objective='reg:squarederror'
    )
    xgb_model.fit(xgb_train_df[features], xgb_train_df['residual'])
except Exception as e:
    print(f" XGBoost training failed: {e}")
    raise

# 6. HONEST MAPE EVALUATION


def calc_mape(actual, predicted):
    mask = actual > 0
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

# Base model MAPE on test set
base_test_preds   = xgb_test_df['np_yhat']
base_mape         = calc_mape(xgb_test_df['y'].values, base_test_preds.values)

# Hybrid model MAPE on test set
hybrid_test_preds = xgb_test_df['np_yhat'] + xgb_model.predict(xgb_test_df[features])
hybrid_mape       = calc_mape(xgb_test_df['y'].values, hybrid_test_preds)


# 7. FUTURE FORECAST (Next 30 Days) + Confidence Intervals


print(" Generating 30-day future hybrid forecast with confidence bands...")

clean_df            = prophet_df[['ds', 'y']].copy()
future_df           = m.make_future_dataframe(clean_df, periods=30)
future_base         = m.predict(future_df)

future_forecast               = future_base[['ds', yhat_col]].copy()
future_forecast['day_of_week']= future_forecast['ds'].dt.dayofweek
future_forecast['is_weekend'] = (future_forecast['day_of_week'] >= 5).astype(int)

if has_ci:
    future_forecast['np_low']  = future_base[yhat_low_col].values
    future_forecast['np_high'] = future_base[yhat_high_col].values

# Rolling predict with XGBoost residuals
history_y        = list(prophet_df['y'].values)
final_predictions, low_bands, high_bands = [], [], []

for i in range(len(future_forecast)):
    roll7 = np.mean(history_y[-7:])
    feats = pd.DataFrame({
        'day_of_week': [future_forecast.loc[i, 'day_of_week']],
        'is_weekend':  [future_forecast.loc[i, 'is_weekend']],
        'y_roll_7':    [roll7]
    })
    residual  = xgb_model.predict(feats)[0]
    base      = future_forecast.loc[i, yhat_col]
    final_y   = base + residual

    history_y.append(final_y)
    final_predictions.append(final_y)

    if has_ci:
        low_bands.append(future_forecast.loc[i, 'np_low']  + residual)
        high_bands.append(future_forecast.loc[i, 'np_high'] + residual)

future_forecast['final_flux_forecast'] = final_predictions
if has_ci:
    future_forecast['forecast_low']  = low_bands
    future_forecast['forecast_high'] = high_bands

# Only the actual future 30 days (not the historical portion returned by make_future_dataframe)
future_only = future_forecast[future_forecast['ds'] > prophet_df['ds'].max()].reset_index(drop=True)

# 8. VAULT LOGIC


vault_state      = load_vault()
projected_income = sum(future_only['final_flux_forecast'])
surplus          = projected_income - essential_expenses
coverage_ratio   = projected_income / essential_expenses

if surplus > 0:
    vault_add = surplus * 0.40
    vault_state['balance'] += vault_add
    vault_action = f" SURPLUS  — Depositing ₹{vault_add:,.2f} (40% of surplus)"
else:
    shortfall = abs(surplus)
    withdraw  = min(vault_state['balance'], shortfall)
    vault_state['balance'] -= withdraw
    vault_action = f" DEFICIT  — Releasing ₹{withdraw:,.2f} from vault"

vault_state['total_runs'] += 1
vault_state['history'].append({
    "run":            vault_state['total_runs'],
    "projected_income": round(projected_income, 2),
    "essential_costs":  round(essential_expenses, 2),
    "surplus_deficit":  round(surplus, 2),
    "vault_balance":    round(vault_state['balance'], 2),
})

save_vault(vault_state)


# 9. TERMINAL DASHBOARD

print("\n" + "═"*58)
print("    FLUX VAULT REPORT  |  AI-CFO for Gig Workers")
print("═"*58)
print(f"  Projected 30-Day Income : ₹{projected_income:>12,.2f}")
print(f"  Calculated 30-Day Costs : ₹{essential_expenses:>12,.2f}")
print(f"  Coverage Ratio          : {coverage_ratio:>11.2f}x")
print("─"*58)
print(f"  {vault_action}")
print(f"  Vault Balance           : ₹{vault_state['balance']:>12,.2f}  (Run #{vault_state['total_runs']})")
print("═"*58)
print("    MODEL ACCURACY  (Evaluated on Held-Out Test Set)")
print("─"*58)
print(f"  Base Model  (NeuralProphet) MAPE : {base_mape:>6.1f}% error")
print(f"  Hybrid Model (NP + XGBoost) MAPE : {hybrid_mape:>6.1f}% error")
print(f"  Net Accuracy Improvement         : {base_mape - hybrid_mape:>6.1f}%")
print("═"*58 + "\n")

# 10. VISUALIZATION  (2-panel: Forecast + Vault History)

fig, axes = plt.subplots(1, 2, figsize=(18, 6))
fig.suptitle('Flux AI',fontsize=15, fontweight='bold', y=1.01)

# ── Panel 1: Forecast ──────────────────────────────────────
ax1 = axes[0]

ax1.plot(prophet_df['ds'], prophet_df['y'],
         label='Historical Income', color='#333333', alpha=0.55, linewidth=1.2)
ax1.plot(future_only['ds'], future_only[yhat_col],
         label='NP Base Forecast', linestyle=':', color='#2E86AB', linewidth=1.8)
ax1.plot(future_only['ds'], future_only['final_flux_forecast'],
         label='Flux Hybrid Forecast', color='#F24236', linewidth=2.5)

if has_ci and 'forecast_low' in future_only.columns:
    ax1.fill_between(
        future_only['ds'],
        future_only['forecast_low'],
        future_only['forecast_high'],
        alpha=0.18, color='#F24236', label='80% Confidence Band'
    )

ax1.axhline(y=essential_expenses / 30, color='#27AE60',
            linestyle='--', linewidth=1.5, label='Daily Survival Threshold')
ax1.set_title('30-Day Income Forecast (Hybrid Residual Boosting)', fontweight='bold')
ax1.set_xlabel('Date')
ax1.set_ylabel('Daily Net Income (₹)')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.25)

# ── Panel 2: Vault Balance History ────────────────────────
ax2 = axes[1]

if len(vault_state['history']) > 1:
    runs     = [h['run']           for h in vault_state['history']]
    balances = [h['vault_balance'] for h in vault_state['history']]
    colors   = ['#27AE60' if vault_state['history'][i]['surplus_deficit'] >= 0
                else '#E74C3C' for i in range(len(runs))]

    ax2.bar(runs, balances, color=colors, edgecolor='white', linewidth=0.6)
    ax2.plot(runs, balances, color='#2C3E50', marker='o', linewidth=1.5, markersize=5)

    surplus_patch  = mpatches.Patch(color='#27AE60', label='Surplus Run')
    deficit_patch  = mpatches.Patch(color='#E74C3C', label='Deficit Run')
    ax2.legend(handles=[surplus_patch, deficit_patch], fontsize=9)
else:
    ax2.text(0.5, 0.5, f"Vault Balance\n₹{vault_state['balance']:,.2f}\n\n(Run more cycles\nto see history)",
             ha='center', va='center', fontsize=13, transform=ax2.transAxes,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#EAF4FB', edgecolor='#2E86AB'))

ax2.set_title('Vault Balance Over Runs', fontweight='bold')
ax2.set_xlabel('Run #')
ax2.set_ylabel('Vault Balance (₹)')
ax2.grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig('flux_forecast_chart.png', dpi=150, bbox_inches='tight')
plt.show()
print(" Chart saved as 'flux_forecast_chart.png'")

# 11. EXPORT

future_only.to_csv('flux_30_day_forecast.csv', index=False)
print(" Forecast saved → 'flux_30_day_forecast.csv'")
print(f" Vault state saved → '{VAULT_FILE}' (re-upload this file next run to persist balance!)") 
