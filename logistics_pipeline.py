import numpy as np, pandas as pd, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.optimize import linear_sum_assignment

rng = np.random.default_rng(42)
N = 5000
# ---------- 1. Simulate data ----------
df = pd.DataFrame({
    "distance_km": rng.gamma(3, 60, N).clip(5, 900),
    "package_weight_kg": rng.gamma(2, 8, N).clip(0.5, 120),
    "num_stops": rng.integers(1, 15, N),
    "traffic_index": rng.uniform(0, 1, N),
    "weather": rng.choice(["clear", "rain", "fog", "storm"], N, p=[.6, .25, .1, .05]),
    "carrier": rng.choice(["A", "B", "C"], N),
    "dispatch_hour": rng.integers(0, 24, N),
    "is_weekend": rng.integers(0, 2, N),
    "warehouse_load_pct": rng.uniform(30, 100, N),
})
w_eff = df.weather.map({"clear": 0, "rain": 1.5, "fog": 2.5, "storm": 6})
c_eff = df.carrier.map({"A": 0, "B": 1.0, "C": -0.8})
peak = df.dispatch_hour.isin([7, 8, 9, 17, 18, 19]).astype(int)
df["delivery_time_hrs"] = (
    1.0 + df.distance_km / 55 + 0.03 * df.package_weight_kg + 0.35 * df.num_stops
    + 3.0 * df.traffic_index * (1 + peak) + w_eff + c_eff
    + 0.04 * np.maximum(df.warehouse_load_pct - 70, 0) ** 1.3 / 3
    - 0.5 * df.is_weekend + rng.normal(0, 0.9, N)).clip(0.5)
df.loc[rng.choice(N, 100, replace=False), "traffic_index"] = np.nan   # missing values
print(df.describe().round(2).to_string())

# ---------- 2. Prepare ----------
df["is_peak_hour"] = df.dispatch_hour.isin([7, 8, 9, 17, 18, 19]).astype(int)
df["distance_per_stop"] = df.distance_km / df.num_stops
X, y = df.drop(columns="delivery_time_hrs"), df.delivery_time_hrs
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
cat, num = ["weather", "carrier"], [c for c in X.columns if c not in ["weather", "carrier"]]
from sklearn.impute import SimpleImputer
def prep(scale=False):
    n = [("imp", SimpleImputer(strategy="median"))] + ([("sc", StandardScaler())] if scale else [])
    return ColumnTransformer([("num", Pipeline(n), num), ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
models = {
    "Linear Regression": Pipeline([("p", prep(True)), ("m", LinearRegression())]),
    "Decision Tree": Pipeline([("p", prep()), ("m", DecisionTreeRegressor(max_depth=8, random_state=42))]),
    "Random Forest": Pipeline([("p", prep()), ("m", RandomForestRegressor(300, n_jobs=-1, random_state=42))]),
    "Gradient Boosting": Pipeline([("p", prep()), ("m", GradientBoostingRegressor(random_state=42))]),
}
kf = KFold(5, shuffle=True, random_state=42)
rows = []
for name, m in models.items():
    cv = -cross_val_score(m, Xtr, ytr, cv=kf, scoring="neg_root_mean_squared_error")
    m.fit(Xtr, ytr); p = m.predict(Xte)
    rows.append([name, cv.mean(), cv.std(), np.sqrt(mean_squared_error(yte, p)), mean_absolute_error(yte, p), r2_score(yte, p)])
res = pd.DataFrame(rows, columns=["Model", "CV RMSE", "CV Std", "Test RMSE", "Test MAE", "Test R2"]).round(3)
print(res.to_string(index=False))

# ---------- 3. Tune ----------
params = {"m__n_estimators": [200, 300, 500], "m__learning_rate": [0.03, 0.05, 0.1],
          "m__max_depth": [2, 3, 4, 5], "m__subsample": [0.7, 0.85, 1.0], "m__min_samples_leaf": [1, 5, 10]}
rs = RandomizedSearchCV(models["Gradient Boosting"], params, n_iter=20, cv=kf,
                        scoring="neg_root_mean_squared_error", random_state=42, n_jobs=-1)
rs.fit(Xtr, ytr)
best = rs.best_estimator_; p = best.predict(Xte)
tuned = dict(rmse=np.sqrt(mean_squared_error(yte, p)), mae=mean_absolute_error(yte, p), r2=r2_score(yte, p), cv=-rs.best_score_)
print("best params", rs.best_params_); print("tuned", {k: round(v, 3) for k, v in tuned.items()})

# ---------- 4. Importance + plots ----------
from sklearn.inspection import permutation_importance
pi = permutation_importance(best, Xte, yte, n_repeats=10, random_state=42, scoring="neg_root_mean_squared_error")
imp = pd.Series(pi.importances_mean, index=Xte.columns).sort_values(ascending=False)
print(imp.round(3).to_string())
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].scatter(yte, p, s=6, alpha=.4, color="#1F4E79"); ax[0].plot([0, yte.max()], [0, yte.max()], "r--")
ax[0].set_xlabel("Actual (hrs)"); ax[0].set_ylabel("Predicted (hrs)"); ax[0].set_title("Predicted vs Actual (tuned GBM)")
imp.head(8)[::-1].plot.barh(ax=ax[1], color="#2E75B6"); ax[1].set_title("Permutation importance (RMSE increase, hrs)")
plt.tight_layout(); plt.savefig("fig1.png", dpi=150)

# ---------- 5. Optimization: assign 60 orders to 3 carriers (capacity-limited) ----------
sample = Xte.sample(60, random_state=1).reset_index(drop=True)
CARRIERS, SLOTS = ["A", "B", "C"], 20                       # 20 slots per carrier
rate_per_hr = {"A": 10.0, "B": 8.0, "C": 13.0}              # cost per predicted hour
rate_per_km = {"A": 0.50, "B": 0.40, "C": 0.65}             # cost per km
cost_cols, time_cols = [], []
for c in CARRIERS:
    tmp = sample.copy(); tmp["carrier"] = c
    t = best.predict(tmp)                                   # predicted delivery time if sent via carrier c
    cost = t * rate_per_hr[c] + tmp.distance_km * rate_per_km[c]
    cost_cols += [cost] * SLOTS; time_cols += [t] * SLOTS
C, Tm = np.column_stack(cost_cols), np.column_stack(time_cols)
r, cidx = linear_sum_assignment(C)
opt_cost, opt_time = C[r, cidx].sum(), Tm[r, cidx].sum()
# baseline: random-ish current allocation = original carrier labels
base_cost = base_time = 0
for i, row in sample.iterrows():
    cc = row.carrier; k = ["A", "B", "C"].index(cc) * 20
    base_cost += C[i, k]; base_time += Tm[i, k]
print("baseline cost/time", round(base_cost, 1), round(base_time, 1), "optimised", round(opt_cost, 1), round(opt_time, 1))
print("cost saving %", round(100 * (base_cost - opt_cost) / base_cost, 1), "time saving %", round(100 * (base_time - opt_time) / base_time, 1))

# Delay-risk what-if: peak vs off-peak dispatch
w = Xte.copy(); w["dispatch_hour"] = 11; w["is_peak_hour"] = 0
w0 = Xte.copy(); w0["dispatch_hour"] = 8; w0["is_peak_hour"] = 1
print("avg hrs peak vs offpeak", round(best.predict(w0).mean(), 2), round(best.predict(w).mean(), 2))
json.dump(dict(res=res.to_dict("records"), tuned=tuned, params=rs.best_params_, imp=imp.round(3).to_dict(),
  opt=dict(base_cost=base_cost, opt_cost=opt_cost, base_time=base_time, opt_time=opt_time),
  peak=[float(best.predict(w0).mean()), float(best.predict(w).mean())]), open("results.json", "w"), default=float)
df.to_csv("sim.csv", index=False)
