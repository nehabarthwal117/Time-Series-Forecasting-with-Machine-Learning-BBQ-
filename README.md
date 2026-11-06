# Time-Series-Forecasting-with-Machine-Learning-Barbecue Sauce Demand Forecasting
Project for Digital Transformation and Innovation​ (MDSSB-DSAI-01)

### Preventing Stockouts & Minimizing Holding Costs during Grilling Season

**Business Context:**
As supply chain data scientists for a top national condiment brand, empty shelves during the peak summer grilling season lead to millions of euros in lost revenue and brand switching. Conversely, over-stocking perishable condiments ties up warehouse working capital and risks stock spoilage.

**Core Mission:**
Build an end-to-end, production-grade time-series forecasting model using **Machine Learning (`skforecast` + `scikit-learn` / `LightGBM` / `XGBoost`)** to accurately predict **30-day barbecue sauce demand** (July 30 – August 28, 2026) while leveraging exogenous nationwide climate data (temperature, sunshine, precipitation from DWD Germany).

### Project Architecture & Workflow:
1. **Pre-processing:** Continuous calendar alignment (`freq='D'`), store closure imputation, and weather interpolation.
2. **Stationarity Diagnostics:** Augmented Dickey-Fuller (ADF) & KPSS tests, differencing analysis.
3. **Feature Engineering:** Autoregressive lags, rolling window aggregations, cyclical calendar transforms, German public holidays, and exogenous climate interactions.
4. **Model Exploration:** Support Vector Machine (SVR), Random Forest, LightGBM, and XGBoost.
5. **Forecasting Strategies:** Recursive Forecasting (`ForecasterRecursive`) vs. Direct Forecasting (`ForecasterDirect`).
6. **Validation Schemes:** Expanding Window vs. Sliding Window backtesting across 180 out-of-sample days (6 folds × 30 days).
7. **Model Evaluation:** MAE, MSE, RMSE, MAPE, and WAPE.
8. **Feature Importance & Ablation Study:** Assessing whether exogenous climate variables improve predictive accuracy.
9. **Final 30-Day Forecast:** 30-day ahead demand prediction with 90% confidence intervals.
10. **Supply Chain Optimization:** Safety stock, Reorder Point (ROP), and warehouse inventory recommendations.

## 1. Data Ingestion & Pre-processing

In Germany, retail store closures occur strictly on Sundays and certain Mondays under retail closing regulations (*Ladenschlussgesetz*), as well as national public holidays. 
- A standard time-series model requires a continuous, regular frequency (daily `freq='D'`).
- Missing days in `bbq_sauce_sales.csv` correspond to non-operating store closure days, where sales are naturally **0 bottles**.
- For `climate_data_2026.csv`, duplicate reporting entries are averaged, and missing days on closed dates are smoothly interpolated across the continuous calendar.
## 2. Stationarity Diagnostics & Differentiation Analysis

Non-stationary time series with shifting trends can cause autoregressive models to overfit spurious patterns. We perform two complementary diagnostic tests:
1. **Augmented Dickey-Fuller (ADF) Test:** Null hypothesis ($H_0$) is that a unit root exists (non-stationary). $p < 0.05$ indicates stationarity.
2. **KPSS Test:** Null hypothesis ($H_0$) is that the series is trend/level stationary. $p < 0.05$ rejects stationarity.

## 3. Comprehensive Feature Engineering

To capture demand fluctuations, we engineer 4 distinct feature families:
1. **Calendar & Seasonality:** Day of week, month, day of year, weekend flag, peak grilling season flag (May–September).
2. **Cyclical Encodings:** Sine and cosine transformations for day of week ($	au=7$), day of year ($	au=365.25$), and month ($	au=12$).
3. **German Public Holidays:** Nationwide public holidays and bridge days (shopping surges on holiday eves) using `holidays.Germany`.
4. **Exogenous Climate Variables:** Mean/max/min temperature, sunshine hours, precipitation, temperature spread, 3-day & 7-day rolling weather metrics, and a composite BBQ weather index.

## 4. Train-Test Splitting & Validation Schemes

In time-series demand forecasting, standard random k-fold cross-validation is invalid because it introduces future data leakage. 
Instead, we use **time series backtesting** with multiple 30-day forecast horizons over an out-of-sample **180-day test period** (6 consecutive folds × 30 days).

We evaluate two fundamental cross-validation window strategies:
- **Expanding Window (`fixed_train_size=False`):** The training set grows with each fold, retaining all historical observations.
- **Sliding Window (`fixed_train_size=True`):** The training set rolls forward with a fixed window length, emphasizing recent trends.

## 5. Model Architecture & Multi-Strategy Benchmarking

We benchmark 4 diverse machine learning regression architectures:
1. **Support Vector Machine (SVR):** Kernel trick ($RBF$) mapping non-linear relationships with $L_2$ regularization.
2. **Random Forest:** Ensemble of bagging decision trees reducing variance.
3. **LightGBM:** Fast gradient-boosted decision trees using histogram-based leaf-wise splitting.
4. **XGBoost:** Regularized extreme gradient boosting optimizing second-order loss gradients.

Along with two forecasting strategies:
- **Recursive Forecasting (`ForecasterRecursive`):** A single model predicts $t+1$, feeds its prediction back into the lag feature vector, and iterates to $t+30$.
- **Direct Forecasting (`ForecasterDirect`):** Trains 30 separate specialized regression models, one for each horizon step $h=1 \dots 30$.

## 6. Feature Importance & Exogenous Weather Impact Analysis

A central question in supply chain planning: **Do nationwide weather metrics significantly improve barbecue sauce demand forecasting?**
We evaluate this using:
1. **Tree-Based Feature Importance (Gini/Split Gain)** from LightGBM.
2. **Ablation Comparison:** Comparing LightGBM trained with weather features against an identical model trained solely on calendar/holiday features.

## 7. Final 30-Day Demand Forecast (July 30 – August 28, 2026)

We train the best performing model architecture on the full historical dataset (2021-01-03 through 2026-07-29) and project daily customer demand for the **next 30 calendar days**, including analytical 90% confidence bands.

## 8. Supply Chain Inventory & Warehouse Stocking Plan

Using classical statistical inventory control theory, we compute the required safety stock to protect against demand variability during replenishment lead time:

$$\text{Safety Stock} (SS) = Z_{\alpha} \times \sigma_{\text{daily}} \times \sqrt{L}$$
$$\text{Reorder Point} (ROP) = \text{Lead Time Demand} + SS$$

Where:
- $Z_{\alpha}$ is the standard normal critical value for service level $\alpha$ (e.g. 1.645 for 95%, 2.326 for 99%).
- $\sigma_{\text{daily}}$ is the forecast standard error (estimated from backtesting RMSE).
- $L$ is the supplier replenishment lead time (assumed $L = 5$ days).

## 9.Interactive Streamlit Dashboard 

