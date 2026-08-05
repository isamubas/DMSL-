# Almond Demand & Supply — Time Series Analysis and Forecasting

Forecasting the **demand and supply of almonds (shelled basis)** across the major
producing and consuming regions, using the USDA **Production, Supply & Distribution
(PSD)** database.

The headline result is a negative one, and it is the point of the project:
**across 22 forecasting models — including LSTM and GRU neural networks — no advanced
method beat a well-chosen simple baseline.** This repo documents that carefully enough
to be trusted.

| File | Description |
|---|---|
| [`time_series_analysis_prediction_model_of_supply_of_almonds.ipynb`](time_series_analysis_prediction_model_of_supply_of_almonds.ipynb) | The full analysis — data audit, maps, 22-model bake-off, feature importance, forecasts |
| `almonds_raw.csv` | The `Almonds, Shelled Basis` slice of PSD (7,960 rows, 40 countries, MY1960–2020, metric tonnes) |

---

## Terminology

- **Market year (`MY`)** — the 12-month *marketing* cycle for a commodity, beginning at
  harvest, not on 1 January. For almonds the US market year runs roughly **1 August –
  31 July**, so `MY2000` spans Aug 2000 – Jul 2001. It varies by country and commodity,
  which is why USDA carries it as its own field.
- **Shelled basis** — quantities as shelled kernel weight, comparable across countries
  regardless of whether nuts are traded in-shell.
- **MT** — metric tonnes.

## Data

Source: [USDA FAS PSD Online](https://apps.fas.usda.gov/psdonline/), also distributed on
Kaggle as `psd_alldata.csv` (all commodities, ~1.89M rows, 183 MB).

`almonds_raw.csv` is the almonds-only extract, reproducible with:

```python
import pandas as pd
parts = [ch[ch.Commodity_Description == 'Almonds, Shelled Basis']
         for ch in pd.read_csv('psd_alldata.csv', chunksize=300_000)]
pd.concat(parts).to_csv('almonds_raw.csv', index=False)
```

The full 183 MB file is intentionally **not tracked** — the extract is 692 KB and
nothing here uses the other commodities.

### Three traps in PSD that break naive analyses

1. **`Market_Year` is the time axis — not `Calendar_Year`/`Month`.** Those record when
   USDA *published* an estimate. Indexing on them collapses all 7,960 almond rows into
   **17 distinct timestamps** and sends 1,992 rows to `NaT` (`Month` is coded `0` for
   many records). `Market_Year` gives a clean 61-point annual series.
2. **The data is annual**, so there is no seasonality. This rules out Seasonal Naive,
   SARIMA's seasonal terms, `seasonal_decompose`, and most of Prophet's value.
3. **Spain and Italy become all-zero placeholders from MY2001**, when USDA folds them
   into a single `European Union` line. Read literally, the world's second-largest
   producer collapses to nothing. They must be masked, and are modellable only over
   MY1960–2000.

Both PSD accounting identities (`Supply = Beginning Stocks + Production + Imports`,
`Distribution = Consumption + Exports + Ending Stocks`) hold **to the tonne**, which
validates the reshape.

---

## Findings

### Demand and supply

- **Supply is extraordinarily concentrated.** The US is **80.2%** of world production
  (MY2020) and its share has *risen* over the period. World almond supply is effectively
  a bet on California.
- **Demand is dispersed** — the EU, India, China and the Middle East are the large
  consumers. The EU is structurally a net importer.
- Production, consumption and exports trend strongly upward. **Imports are the noisiest
  and least predictable component in every region.**

### Model comparison

Rolling-origin (walk-forward) one-step-ahead validation, 20 series × 22 models.
Never `train_test_split` — a random split would train on 2019 and test on 1985.

| # | Model | Median MAPE | Worst | Wins | Tier |
|---|---|---|---|---|---|
| 1 | **Drift** | **15.99** | 49.9 | 5 | baseline |
| 2 | LightGBM | 16.51 | 55.3 | 1 | ML |
| 3 | ARIMA | 16.84 | 51.2 | 2 | statistical |
| 4 | Naive | 17.09 | 48.0 | 0 | baseline |
| 5 | ExtraTrees | 17.95 | 71.5 | 0 | ML |
| 6 | KNN | 18.18 | 67.7 | 2 | ML |
| 7 | GradientBoosting | 19.04 | 81.3 | 1 | ML |
| 8 | **LSTM** | 19.11 | 65.0 | 0 | deep learning |
| 9 | **GRU** | 19.21 | 50.5 | 0 | deep learning |
| 10 | RandomForest | 19.71 | 79.6 | 1 | ML |
| 11 | ETS-Holt | 19.81 | **44.8** | 3 | statistical |
| 12 | XGBoost | 20.05 | 64.2 | 0 | ML |
| 13 | Ridge | 21.97 | 189.7 | 0 | ML |
| 14 | MLP (128-64-32) | 23.56 | 84.5 | 0 | neural net |
| 15 | Lasso | 23.68 | 151.9 | 1 | ML |
| 16 | SVR | 24.70 | 206.6 | 2 | ML |
| 17 | ElasticNet | 26.15 | 188.0 | 0 | ML |
| 18 | MLP (64-32) | 27.85 | 113.5 | 1 | neural net |
| 19 | VAR | 29.16 | 131.4 | 1 | statistical |
| 20 | **LinearRegression** | **44.29** | **458.2** | 0 | ML |
| 21 | Mean | 46.50 | 448.6 | 0 | baseline |

By tier — the statistical tier has both the best median **and** by far the best
worst case, i.e. it is the most *robust*, not just the most accurate:

| Tier | Median | Mean | Worst |
|---|---|---|---|
| Tier 1 — baseline | 22.89 | 35.79 | 448.56 |
| **Tier 2 — statistical** | **19.81** | **24.80** | **131.39** |
| Tier 3 — ML / deep learning | 21.41 | 29.94 | 458.18 |

**Every model is reported, including the ones that lose.** Dropping weak performers
would not remove bias — it would create it. The case that simple methods win only exists
because the complex ones were actually tested.

### Why test models that "shouldn't" work here?

Standard advice says use models built for temporal structure and don't reach for
general-purpose regressors. That advice is sound, and these results confirm it. So why
run fifteen models the textbook would skip?

1. **A baseline comparison is only evidence if the alternatives were tried.** "Use Naive
   or Drift to prove your complex models add value" requires the complex models to be in
   the table. Reporting only ETS and ARIMA turns "simple wins here" from a finding into
   an assertion.
2. **Negative results carry mechanism.** The extrapolation-ceiling plot exists only
   because tree models were run. It explains *why* classical models win, which is worth
   more than knowing *that* they win.
3. **The received wisdom is a heuristic, and heuristics have exceptions.** Two turned up:
   **LightGBM placed 2nd overall**, ahead of ARIMA — hard to square with "trees don't do
   time series," and only visible after fixing a crippling default. And the **LSTM beat a
   4× larger MLP on 16 of 20 series**, a real result about architecture substituting for
   data that a blanket "no deep learning at n=58" would have missed.
4. **It surfaced a concrete hazard** — plain `LinearRegression`, a very common choice for
   lag-feature forecasting, with a 458% worst case.
5. **The cost was trivial** — a few minutes of CPU.

**The honest cost:** multiple comparisons. With 22 models × 20 series, some model wins a
series by luck, so the per-series winner is noisy. Three habits keep it honest — judge on
the **median across series**, **report every model** so nothing is selected after the
fact, and treat a winner's own MAPE as **optimistically biased**. This is exploratory
research: the goal is understanding the problem, not defending a prior about which family
*should* win. Committing in advance to report all results, including the embarrassing
ones, is what separates curiosity from cherry-picking.

### Why the advanced models lose — the extrapolation ceiling

A tree-based model predicts by averaging training observations, so it **can never output
a value outside its training range**. Training on US production through 2012 and
predicting 2013–2020:

```
training maximum       :   920,793 MT
actual 2020            : 1,360,780 MT
RandomForest 2020 pred :   859,770 MT   ← identical to its 2013 prediction
RF prediction range    :   859,770 – 859,770 MT   (a flat line for 8 years)
```

Real production rose ~48% and the model could not follow. This is a property of the
model class, not a tuning failure. ARIMA, Drift and Holt avoid it because they model the
*change* between periods — differencing removes the trend, so the trend extrapolates by
construction.

### Deep learning: architecture beat depth, but not simplicity

- **LSTM beat the plain MLP on 16 of 20 series** despite being the *smaller* model
  (~3,000 parameters vs ~12,900). An MLP receives the lag features as an unordered
  vector and must learn temporal ordering from data; a recurrent net gets that ordering
  as an architectural prior. **When data is scarce, a good inductive bias substitutes
  for data.**
- **Depth alone did almost nothing.** The deeper MLP has the better median (23.56 vs
  27.85) but is worse on 10 of 20 series — a coin flip. Changing the *structure* helped;
  adding *capacity* did not.
- But **LSTM beat the best simple model on 0 of 20 series** (GRU: 2 of 20). Better than
  the other ML models is not the same as better than a well-chosen baseline.
- With 58 training rows for the longest series, this is a sample-size result, not
  evidence that recurrent nets are bad at forecasting.

### VAR: loses on accuracy, wins on coherence

VAR ranks 19th of 21 for per-series accuracy — it is over-parameterised, with ~36
coefficients across a four-variable two-lag system fitted on as few as 17 observations.

But per-series accuracy is not the only thing that matters. Forecasting each component
independently means nothing forces production to equal consumption + exports + stock
change. Checking the implied US stock build:

| Approach | Implied stock change | vs history |
|---|---|---|
| Historical actual (last 20y) | 5,975 MT/yr | 1.0× |
| **Independent forecasts** | **116,318 MT/yr** | **19.5×** |
| **VAR system forecast** | **33,901 MT/yr** | **5.7×** |

The independent forecasts imply a physically implausible stock build. **VAR is ~3.4×
closer to the historical norm.** Accuracy and coherence are different objectives, and
the model that wins one loses the other.

### Feature importance

Permutation importance (model-agnostic; unlike tree impurity importance it does not
inflate high-cardinality features), XGBoost as probe:

| Attribute | Share | | Lag | Share |
|---|---|---|---|---|
| **Production** | **0.51** | | t-2 | 0.41 |
| Ending Stocks | 0.17 | | t-3 | 0.36 |
| Domestic Consumption | 0.12 | | t-1 | 0.23 |
| Exports | 0.09 | | time index | **0.00** |
| Imports | 0.07 | | | |
| Beginning Stocks | 0.05 | | | |

- **Production lags dominate** — about half the total importance across all targets.
  This is a supply-driven market.
- **`time index` contributes essentially zero** — the extrapolation ceiling from another
  angle. A tree cannot use a raw time index to project beyond its training range.
- **`t-2` and `t-3` outrank `t-1`**, which is counter-intuitive. Plausibly the older
  pair encodes recent slope, but this is one probe on 20 short series and the gaps are
  small relative to noise. Treat as an observation to re-test, not a mechanism.

---

## Predictions

Per-series point forecasts to MY2025 with 80% intervals are in §8 of the notebook, each
produced by the model that won its own backtest. Directionally: continued growth in US
and Australian production, exports and consumption; EU remains a large net importer.

Read them with three caveats:

1. **Intervals are wide, and that is the honest signal.** Never quote the point estimate
   alone.
2. **The per-series winner is noisy.** With 20 series and 22 models some win by chance.
   The median-across-series ranking above is the trustworthy comparison. Each winner's
   own MAPE is also optimistically biased, having been selected on the backtest it is
   reported on.
3. **Production forecasts assume the boom continues.** Extrapolating the post-2015
   California acreage surge linearly to 2025 is aggressive, and is the main reason the
   balance check fails.

Spain and Italy end at MY2000 because of the EU aggregation, so their +5-year horizon is
MY2005 — a historical backcast, not a 2025 projection.

---

## Limitations and proposed fixes

| # | Limitation | Why it matters | Proposed fix |
|---|---|---|---|
| 1 | Independent forecasts don't satisfy the balance identity | Implied US stock build is 19.5× the historical norm — the joint picture is incoherent even where individual series look fine | Forecast three components and **derive the fourth from the identity**; or apply **hierarchical reconciliation** (MinT/OLS). VAR is the textbook answer but is over-parameterised here — a **Bayesian VAR with Minnesota priors** would shrink it enough to be usable |
| 2 | Very short series — 20–61 annual points | Rules out deep learning at full strength; every parameter estimate is noisy and model selection itself is unstable | Pool countries into a **global/panel model** (one model across all series with country features), the standard fix. Or extend history with **FAOSTAT** and national statistics |
| 3 | Annual granularity | No seasonality, no within-year dynamics, ~10 validation points per series | Add **monthly trade data** (UN Comtrade, USDA export sales) for the trade components |
| 4 | ML models were fed levels, not differences | Guarantees the extrapolation ceiling — the tree models were structurally unable to win | **Re-run the bake-off on differenced / log-growth targets** and cumulate back. Highest-value remaining experiment and the fairest test of whether ML can compete |
| 5 | No exogenous drivers | Almond output is driven by bearing acreage, water allocation, drought, frost and tariffs — none present | **ARIMAX/SARIMAX regressors**: USDA/NASS bearing acreage, PDSI drought index, water allocations, tariff dummies (2018 India/China episode) |
| 6 | Structural break at MY2001 handled only by masking | Other regime changes may lurk undetected | **Chow / Bai-Perron break tests** per series, then segment or add break dummies |
| 7 | Approximate forecast intervals | Non-ARIMA intervals use a normal approximation from in-sample first differences, understating tail risk | **Conformal prediction** or block bootstrap for distribution-free coverage |
| 8 | Interpolated missing years | 1963 (all) and 1977–1980 (US) linearly filled, understating uncertainty | **State-space / Kalman** formulation, which handles gaps natively |
| 9 | Winner selected on the same backtest it is reported on | Headline MAPE per winner is optimistically biased | **Nested/outer holdout** — select on an inner split, report on an untouched period |
| 10 | Single random seed | Neural and tree results shift with initialisation; some ranking gaps are noise | Repeat across **multiple seeds**, report mean ± sd |
| 11 | Hyperparameters largely untuned | The ML tier got sensible defaults, not a search — its ranking is a floor, not a ceiling | Nested **`TimeSeriesSplit`** search inside each training window |
| 12 | Forecasts assume no supply shock | A drought, frost or tariff event breaks every model here simultaneously | Present **scenarios** (boom / plateau / drought-shock); cap growth using known bearing-acreage plantings, observable years ahead |

---

## Running it

```bash
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn xgboost lightgbm plotly
```

PyTorch is optional — without it the bake-off runs without LSTM/GRU rather than failing:

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

> **Windows note.** Installing the *latest* torch can abort with `WinError 206` when
> site-packages sits under a long path (common with the Microsoft Store Python): torch's
> bundled licence tree exceeds the 260-character limit, leaving a half-extracted package
> with no `RECORD` file that pip cannot uninstall. Pinning `torch==2.5.1`, whose licence
> tree is shallower, avoids this.

```bash
jupyter notebook time_series_analysis_prediction_model_of_supply_of_almonds.ipynb
```

The notebook loads `almonds_raw.csv` from the working directory and falls back to
mounting Google Drive on Colab. Full run takes ~20 minutes, dominated by the bake-off.
