# Methodology

Weekly dengue incidence is forecast with a random forest that learns the week-to-week growth of
the epidemic, and probabilistic forecasts are produced by Monte-Carlo simulation over weather
scenarios. This document describes the data preparation, the model, how it is trained and
validated, and how the simulation works. Links point to the modules that implement each step.

## 1. Data

[InfoDengue](https://info.dengue.mat.br) publishes weekly surveillance data for Brazilian
municipalities. This project uses four state capitals (Belo Horizonte, Rio de Janeiro, São Paulo
and Vitória) from epidemiological week 2010-01 to 2025-53, i.e. 835 weeks per city.

| Symbol | Column(s) | Meaning |
|---|---|---|
| $p_{c,t}$ | `p_inc100k` | estimated incidence (cases per 100 000 inhabitants) in city $c$ and week $t$ |
| $x_{c,t} = \ln p_{c,t}$ | `log_p_inc100k` | log incidence, the state of the model |
| $\mathbf{w}_{c,t} \in \mathbb{R}^6$ | `tempmin`, `tempmed`, `tempmax`, `umidmin`, `umidmed`, `umidmax` | weekly means of the daily minimum, mean and maximum temperature and relative humidity |
| $s_t \in \lbrace 1, \dots, 53 \rbrace$ | `epi_week` | epidemiological week of the year |

Epidemiological weeks run from Sunday to Saturday, and week 1 is the week that contains 4 January
([`utils/epi_calendar.py`](../utils/epi_calendar.py)). All modelling happens at this weekly
resolution; nothing is aggregated to months or quarters.

Preparation ([`utils/feature_extraction.py`](../utils/feature_extraction.py)):

- The week code `SE` (YYYYWW) is the time key. The raw date column contains one misdated week
  (Wednesday 2018-04-04 instead of Sunday 2018-04-08), so dates are derived from `SE`. The
  pipeline stops if a week is missing or duplicated, because lags are computed by shifting rows.
- Zero incidence is treated as missing: in this dataset zeros only occur in reporting gaps
  (Vitória 2021 and the last three weeks of 2025), not as a genuine absence of cases, and
  $\ln 0$ is undefined.
- Weeks with missing weather (blocks in 2012, 2016, 2017 and 2020) are not imputed. Rows with
  incomplete features are left out of training.

## 2. Model

### 2.1 Growth formulation

The model describes how the log incidence changes from one week to the next:

$$
x_{c,t+1} = x_{c,t} + f\big(s_t,\ c,\ x_{c,t}, \dots, x_{c,t-L},\ \mathbf{w}_{c,t+1}, \mathbf{w}_{c,t}, \dots, \mathbf{w}_{c,t-L}\big) + \varepsilon_{c,t+1}
$$

$L$ is the number of lagged weeks (`n_lags` in `FEATURE_SETTINGS`) and $\varepsilon$ an error with
mean zero, modelled for the simulation in section 4.2. For $L = 0$ the model reduces to
$x_{t+1} = x_t + f(s_t, c, x_t, \mathbf{w}_t, \mathbf{w}_{t+1})$.
The regression target is the weekly growth rate

$$
\Delta_{c,t+1} = x_{c,t+1} - x_{c,t} = \ln \frac{p_{c,t+1}}{p_{c,t}},
$$

which is positive while an epidemic grows (reproduction number above one) and negative while it declines.

Why the model predicts the change and not the level:

1. **Persistence is built in.** With $f \equiv 0$ the model is the persistence forecast
   $\hat x_{t+1} = x_t$, a strong baseline for weekly data: it explains 97 % of the variance of
   $x_{t+1}$. The model only has to learn the deviations from it.
2. **Trees cannot extrapolate.** A tree ensemble predicts averages of training targets, so a model
   of the level can never forecast an epidemic larger than the largest one in its training years.
   In the record season 2024, a level model plateaued at the training maximum. Growth rates of
   similar size recur in epidemics of any size.
3. **Comparability.** Growth rates are on the same scale for all cities and years, independent of
   population size and epidemic size.

### 2.2 Features

| Group | Columns |
|---|---|
| state | `log_p_inc100k_(t)`, …, `log_p_inc100k_(t-L)` |
| weather of the predicted week | `tempmax_(t+1)`, …, `umidmin_(t+1)` |
| weather of the current and lagged weeks | `tempmax_(t)`, …, `umidmin_(t-L)` |
| season | `epi_week` |
| city | `city_encoded` (0 to 3) |

Suffixes give the time relative to week $t$: `(t+1)` is the next week, `(t)` the current week and
`(t-k)` the week $k$ weeks earlier. The training feature table and the simulation
(`build_features` in [`model_definition/weekly_model.py`](../model_definition/weekly_model.py))
produce the same columns, and a unit test asserts that they produce identical feature vectors.

### 2.3 Random forest

$f$ is estimated by a random forest (an XGBoost variant with the same interface is available):

$$
\hat f(\mathbf{z}) = \frac{1}{B} \sum_{b=1}^{B} T_b(\mathbf{z})
$$

Each regression tree $T_b$ is grown on a bootstrap sample of the training rows, choosing every
split among a random subset of the features so that the squared error is minimised. Averaging many
decorrelated trees reduces the variance of the estimate, and the forest captures non-linear effects
and interactions (e.g. a temperature anomaly acting differently in summer and winter) without
specifying them in advance.

### 2.4 Identification with observed weather

The model is trained with the observed weather of the predicted week, $\mathbf{w}_{t+1}$, as a
feature, so it estimates the conditional expectation

$$
\hat f \approx \mathbb{E}\big[\Delta_{t+1} \mid \text{state at week } t,\ \mathbf{w}_{t+1}\big].
$$

At forecast time $\mathbf{w}_{t+1}$ is unknown, so the Monte-Carlo simulation (section 4) supplies
plausible values from a weather generator. This conditional approach separates the epidemic
dynamics from the weather uncertainty. Consequently, the metrics of section 3, computed with the
observed $\mathbf{w}_{t+1}$, measure the one-step model under perfect knowledge of the weather,
while the simulation is the forecast that is actually available.

## 3. Training and validation

### 3.1 Metric: skill relative to persistence

Hyperparameters are selected and models are validated with the skill score relative to persistence:

$$
S = 1 - \frac{\sum_i \big(x_{i,t+1} - \hat x_{i,t+1}\big)^2}{\sum_i \big(x_{i,t+1} - x_{i,t}\big)^2}
  = 1 - \frac{\sum_i \big(\Delta_i - \hat\Delta_i\big)^2}{\sum_i \Delta_i^2}
$$

$S$ is the fraction of the squared error of the persistence forecast that the model removes:
$S = 1$ is a perfect forecast, $S = 0$ is no better than persistence and $S < 0$ is worse. The
second form shows that persistence predicts $\hat\Delta = 0$ in the growth formulation. Because
$x_t$ is known, errors of $\hat\Delta$ and of $\hat x_{t+1}$ are identical.

The coefficient of determination $R^2$ is not used. It compares the model with the overall mean,
and since weekly incidence is strongly autocorrelated, persistence alone reaches $R^2 \approx 0.97$:
a model can reach $R^2 = 0.94$ and still be much worse than repeating last week's value.

### 3.2 Backtesting

A time-series model has to be validated on periods after its training data. The pipeline uses
expanding-window backtesting by year ([`model_synthesis/backtesting.py`](../model_synthesis/backtesting.py)),
configured in [`utils/run_complete_pipeline.py`](../utils/run_complete_pipeline.py):

- the last $n_\text{test}$ years form the **hold-out test set**, which is used only once, at the end;
- the remaining **development years** end with $K$ consecutive **validation windows** of
  $n_\text{val}$ years; fold $k$ is trained on all years before its window.

With data from 2010 to 2025 and the defaults $n_\text{test} = 2$, $n_\text{val} = 2$, $K = 4$:

| Fold | Training years | Validation years |
|---|---|---|
| 1 | 2010–2015 | 2016–2017 |
| 2 | 2010–2017 | 2018–2019 |
| 3 | 2010–2019 | 2020–2021 |
| 4 | 2010–2021 | 2022–2023 |
| final model | 2010–2023 | test: 2024–2025 |

Each row (week $t$ with its target week $t+1$) belongs to the year of the **predicted** week, so
no training row has its target inside the validation window. Lagged features may reach back into
earlier years, which is information available at forecast time.

The backtest score of a set of hyperparameters $\theta$ pools the out-of-fold errors of all folds:

$$
S_\text{bt}(\theta) = 1 - \frac{\sum_{k=1}^{K} \sum_{i \in V_k} \big(\Delta_i - \hat\Delta_i^{(k,\theta)}\big)^2}{\sum_{k=1}^{K} \sum_{i \in V_k} \Delta_i^2}
$$

Here $V_k$ are the rows of the validation window of fold $k$ and $\hat\Delta^{(k,\theta)}$ the
predictions of a model fitted with $\theta$ on the training years of that fold. The skill of each
fold is logged as well, because the gain over persistence varies strongly between periods.

### 3.3 Hyperparameter search and final model

Optuna's TPE sampler (with a fixed seed) searches for the hyperparameters that maximise
$S_\text{bt}(\theta)$ ([`model_synthesis/optuna_training.py`](../model_synthesis/optuna_training.py)).
The search space of the random forest:

| Hyperparameter | Range |
|---|---|
| `n_estimators` | 50 – 300 |
| `max_depth` | 5 – 30 |
| `min_samples_split` | 2 – 20 |
| `min_samples_leaf` | 1 – 10 |
| `max_features` | `sqrt`, `log2`, all features |

XGBoost has its own search space in
[`model_synthesis/train_xgboost_with_optuna.py`](../model_synthesis/train_xgboost_with_optuna.py).

The model type is chosen with `TRAIN_SETTINGS["model"]` (`"random_forest"` or `"xgboost"`). The best
parameters $\hat\theta$ are refitted on all development years and evaluated once on the hold-out
years: test skill $S_\text{test}$, RMSE and MAE. The out-of-fold residuals of $\hat\theta$ are used
to fit the error model of the simulation (section 4.2). Every MLflow run records the name of the
dataset (parameter `dataset` and dataset inputs for the development and test rows), the parameters,
metrics, diagnostic plots, the error model and the model itself. The model is registered as
`next_week_prediction`, from where the validation scripts load its latest version.

**Number of lagged weeks.** The same backtest is used to choose $L$. With the current data, adding
lagged weeks ($L = 1, \dots, 10$) did not improve $S_\text{bt}$, so $L = 0$ is the default.

## 4. Monte-Carlo simulation

The simulation ([`model_validation/monte_carlo_simulation.py`](../model_validation/monte_carlo_simulation.py),
engine in [`model_definition/weekly_model.py`](../model_definition/weekly_model.py)) produces $N$
trajectories of the weekly incidence for the $H = 52$ weeks after a start week $t_0$.

### 4.1 Weather generator

For city $c$ and epidemiological week $s$, the climatology is estimated from the development years
([`utils/climate_tables.py`](../utils/climate_tables.py)), pooling all weeks within $h = 2$ weeks of
$s$, circularly over the turn of the year:

$$
\mathcal{W}_{c,s} = \lbrace \mathbf{w}_{c,t} : d(s_t, s) \le h \rbrace, \qquad d(a, b) = \min\big(|a - b|,\ 53 - |a - b|\big)
$$

$$
\mu_{c,s} = \frac{1}{|\mathcal{W}_{c,s}|} \sum_{\mathbf{w} \in \mathcal{W}_{c,s}} \mathbf{w}, \qquad
\Sigma_{c,s} = \frac{1}{|\mathcal{W}_{c,s}| - 1} \sum_{\mathbf{w} \in \mathcal{W}_{c,s}} \big(\mathbf{w} - \mu_{c,s}\big)\big(\mathbf{w} - \mu_{c,s}\big)^\top
$$

Pooling is necessary because a single week has only one observation per year, too few to estimate
a $6 \times 6$ covariance matrix. Weather is drawn from a multivariate normal distribution, which
keeps the correlations between the six variables:

$$
\mathbf{w}^{(i)}_{c,t+1} \sim \mathcal{N}\big(\mu_{c,s_{t+1}},\ \Sigma_{c,s_{t+1}}\big)
$$

### 4.2 Model error

The model predicts the expected weekly change; the simulation adds the error it cannot predict.
Without this error the 95 % bands would contain only about 30 % of the observations. The error
model ([`model_definition/error_model.py`](../model_definition/error_model.py)) is estimated from the
out-of-fold residuals of the backtest, i.e. from honest one-step errors on years the model had not
seen. Two properties of these residuals shape it:

- their spread shrinks with the number of weekly cases, as expected from counting noise;
- consecutive residuals are negatively correlated in weeks with few cases (a week reported too high
  is followed by an apparent drop) and positively correlated in large epidemics (the model misses
  the growth for several weeks in a row).

Both follow from a small state-space model. The reported log incidence $y_t$ is the true log
incidence $x_t$ plus counting noise, and the true log incidence follows the forecasting model up to
a persistent process error:

$$
y_t = x_t + \nu_t, \qquad \nu_t \sim \mathcal{N}\big(0,\ \tau^2 / n_t\big)
$$

$$
x_{t+1} = x_t + \hat f(\cdot) + \eta_{t+1}, \qquad \eta_{t+1} = \phi\, \eta_t + \xi_{t+1}, \qquad \xi_{t+1} \sim \mathcal{N}\big(0,\ \sigma_\eta^2 (1 - \phi^2)\big)
$$

$n_t$ is the number of weekly cases ($\tau^2 = 1$ for pure Poisson noise) and $\sigma_\eta$ the
stationary standard deviation of the process error. The residual of an observed weekly change,
$e_{t+1} = \eta_{t+1} + \nu_{t+1} - \nu_t$, then has the moments

$$
\mathbb{E}\big[e_{t+1}^2\big] = \sigma_\eta^2 + \frac{2\tau^2}{n_t}, \qquad
\mathbb{E}\big[e_{t+1}\, e_t\big] = \phi\,\sigma_\eta^2 - \frac{\tau^2}{n_t}
$$

Both are linear in $(\sigma_\eta^2,\ \tau^2,\ \phi\,\sigma_\eta^2)$, so the three parameters are
estimated together by least squares over all residuals and all pairs of consecutive residuals
(method of moments). The second moment explains the sign change of the correlation: counting noise
dominates when $n_t$ is small, the persistent process error when $n_t$ is large. A typical fit gives
$\sigma_\eta \approx 0.21$, $\phi \approx 0.17$ and $\tau^2 \approx 0.7$, close to Poisson noise.

The weeks around Christmas and New Year (epidemiological weeks 51–53) are left out of the fit:
transient reporting dips make their residuals two to three times larger, and treating them as
process error would carry them forward into every later week. The parameters are logged with the
MLflow run (metrics and `error_model.json`), together with a diagnostic plot (`error_model.png`), so
the simulation always uses the error model that belongs to the model it loads.

### 4.3 Initial condition

The simulation starts from the observed state at $t_0$ (incidence, weather and $L$ lagged weeks).
The observed value contains counting noise, so the true starting level is drawn as

$$
x^{(i)}_{c,t_0} = y^\text{obs}_{c,t_0} + \delta^{(i)} - \nu^{(i)}_{t_0}, \qquad
\delta^{(i)} \sim \mathcal{N}\big(0, \sigma_0^2\big), \qquad
\nu^{(i)}_{t_0} \sim \mathcal{N}\big(0,\ \tau^2 / n_{t_0}\big)
$$

$\sigma_0$ is an optional additional uncertainty of the start value, for instance from incomplete
reporting of the most recent weeks (`uncertainty_in_start`, $\sigma_0 = 0.1$). The process error
starts in its stationary distribution, $\eta^{(i)}_{t_0} \sim \mathcal{N}(0, \sigma_\eta^2)$.

### 4.4 Recursion

Each sample is propagated week by week, and the true level is fed back as the next state:

$$
x^{(i)}_{c,t+1} = x^{(i)}_{c,t} + \hat f\big(s_t,\ c,\ x^{(i)}_{c,t}, \dots, x^{(i)}_{c,t-L},\ \mathbf{w}^{(i)}_{c,t+1}, \mathbf{w}^{(i)}_{c,t}, \dots, \mathbf{w}^{(i)}_{c,t-L}\big) + \eta^{(i)}_{t+1}
$$

$$
y^{(i)}_{c,t+1} = x^{(i)}_{c,t+1} + \nu^{(i)}_{t+1}, \qquad t = t_0, \dots, t_0 + H - 1
$$

The process error accumulates over the horizon, while the counting noise only affects the reported
value $y$ of each week; all summaries refer to $y$, the quantity that is observed. Weeks after $t_0$
use the weather drawn in earlier steps. All $N$ samples are simulated together, so each week costs
a single model call on an $N$-row feature matrix.

### 4.5 Summaries

For every predicted week, the samples are summarised by

- the geometric mean $\exp\big(\frac{1}{N}\sum_i y^{(i)}_t\big)$ (log-scale figure) and the
  arithmetic mean $\frac{1}{N}\sum_i e^{y^{(i)}_t}$ (natural-scale figure);
- the 95 % interval between the empirical 2.5 % and 97.5 % quantiles. Unlike
  mean ± 1.96 standard deviations, it follows the skewness of the distribution and never becomes negative;
- weekly cases $p \cdot \text{pop} / 10^5$ with the same statistics, written to
  `data/prediction/predicted_weekly_cases.csv`.

### 4.6 Calibration

The intervals are checked out of sample ([`model_validation/calibration.py`](../model_validation/calibration.py)):
forecasts are started every third week of the hold-out years 2024–2025 in all cities and compared
with the observations of the following 26 weeks. A central $q$ interval of a well calibrated
forecast contains a share $q$ of the observations. Over 3 182 forecast weeks:

| Interval | Without model error | With model error |
|---|---|---|
| 50 % | 10 % | 52 % |
| 75 % | 16 % | 78 % |
| 95 % | 28 % | 95 % |

With the error model, the coverage of the 95 % interval stays between 92 % and 98 % at every horizon
from 1 to 26 weeks, including the record epidemic of 2024.

### 4.7 Limitations

- **Constant persistence.** In large epidemics (more than about 1 000 weekly cases), consecutive
  residuals are more strongly correlated (up to about 0.6) than the fitted constant $\phi$ implies,
  so intervals during big epidemics may be somewhat too narrow at long horizons. A persistence that
  depends on $n_t$ would be a refinement.
- **Reporting dips around New Year** are not simulated, so observations in epidemiological weeks
  51–53 can fall outside the intervals.
- **Weather without memory.** The weekly weather draws are independent, whereas real anomalies
  persist for weeks and average out over a season when drawn independently. Resampling complete
  weather years of the development period, aligned by epidemiological week, would keep this
  persistence; given the weak weather sensitivity of the model, its effect on the intervals is
  expected to be small.
- **Normal noise.** The out-of-fold residuals have heavier tails than a normal distribution, so
  extreme weeks are slightly more frequent than the simulation assumes.

## 5. Code map

| Step | Module |
|---|---|
| Epidemiological calendar | [`utils/epi_calendar.py`](../utils/epi_calendar.py) |
| Cleaning, features and target | [`utils/feature_extraction.py`](../utils/feature_extraction.py) |
| Weekly climatology | [`utils/climate_tables.py`](../utils/climate_tables.py) |
| Backtesting and skill score | [`model_synthesis/backtesting.py`](../model_synthesis/backtesting.py) |
| Optuna search, final fit, MLflow | [`model_synthesis/optuna_training.py`](../model_synthesis/optuna_training.py) |
| Random forest and XGBoost | [`model_synthesis/`](../model_synthesis/) |
| Error model | [`model_definition/error_model.py`](../model_definition/error_model.py) |
| Simulation engine | [`model_definition/weekly_model.py`](../model_definition/weekly_model.py) |
| One-step check with sampled weather | [`model_validation/one_step_prediction.py`](../model_validation/one_step_prediction.py) |
| Monte-Carlo forecast | [`model_validation/monte_carlo_simulation.py`](../model_validation/monte_carlo_simulation.py) |
| Calibration of the intervals | [`model_validation/calibration.py`](../model_validation/calibration.py) |
| Settings (model, lags, backtesting, trials) | [`utils/run_complete_pipeline.py`](../utils/run_complete_pipeline.py) |

## 6. Reproducing the results

From the project root, with the virtual environment activated:

```bash
python utils/run_complete_pipeline.py              # features, backtested training, model registration
python model_validation/monte_carlo_simulation.py  # Monte-Carlo forecasts and figures
python model_validation/calibration.py             # coverage of the prediction intervals
python -m pytest                                   # unit tests
```
