# Business Case Analysis — Promotion Effectiveness at a Fashion Retail Chain

---

## B1. Problem Formulation

### B1(a) — ML Problem Formulation

**Target variable:** `items_sold` — the number of items sold at a given store in a given month under a given promotion.

**Candidate input features:**
- Store attributes: `store_size` (small/medium/large), `location_type` (urban/semi-urban/rural), `competition_density` (integer score)
- Promotion attributes: `promotion_type` (Flat Discount, BOGO, Free Gift with Purchase, Category-Specific Offer, Loyalty Points Bonus)
- Temporal features: `month`, `is_weekend`, `is_festival`, `year`, `day_of_week`, `is_month_end`
- Store identifier: `store_id` (to capture store-level fixed effects)
- Derived historical features: rolling average items sold per store, per promotion type, per location cluster

**Type of ML problem:** This is a **supervised regression** problem. The target variable (`items_sold`) is a continuous numerical quantity, and we have labelled historical data pairing store-promotion-period combinations with their observed sales outcomes. We are not assigning promotions to discrete categories but predicting a numerical magnitude — how many items a store will sell — so that the retailer can compare predicted outcomes across the five promotion options and deploy the one with the highest expected volume.

**Justification:** The problem is regression rather than classification because the business question is "how much will we sell?" not "will this promotion succeed or fail?". Regression gives the marketing team a quantitative basis for comparing options — e.g., BOGO is predicted to yield 293 items vs Loyalty Points at 251 items for Store 12 in March — rather than a binary label that discards the magnitude of the difference.

---

### B1(b) — Why Items Sold Is a Better Target Than Revenue

Revenue is the product of price and quantity: Revenue = Price × Items Sold. Because the promotions under comparison directly manipulate price — Flat Discount reduces the unit price, BOGO halves the effective price per item, Free Gift adds value without reducing the sticker price — revenue conflates the price effect with the volume effect. A promotion that drives high revenue may simply be one that maintains high prices rather than one that genuinely changes customer behaviour.

For example, in this dataset, BOGO achieves the highest mean items_sold (293.4) while Flat Discount achieves 282.8. However, a Flat Discount by definition reduces per-item revenue. If the target were total revenue, Flat Discount would appear artificially weak even if it drives strong volume. Using items_sold isolates the demand response — the genuine behavioural signal the model should learn.

**Broader principle:** This illustrates the importance of choosing a target variable that directly measures the outcome of interest rather than a composite metric that is partly determined by the inputs under the decision-maker's control. When the intervention (the promotion choice) affects both the numerator and denominator of a ratio, or is embedded in the target metric itself, the resulting signal is confounded and the model learns a distorted relationship. Choosing a clean, causally downstream target — demand volume — produces a more interpretable, deployable, and honest model.

---

### B1(c) — Alternative to a Single Global Model

A single global model pooling all 50 stores implicitly assumes that promotion response patterns are homogeneous across all store types once observable features are controlled for. This assumption is unlikely to hold: the data already shows that urban stores average 301.5 items sold vs 235.9 for rural stores, and different promotions will interact with these structural differences in complex ways.

**Proposed strategy: store-cluster segmentation model**

Rather than one global model or 50 individually trained models (each with insufficient history), we recommend clustering the 50 stores into groups based on their structural characteristics — `location_type`, `store_size`, and `competition_density` — and training one model per cluster. Each cluster model learns promotion response patterns from stores that are genuinely comparable, reducing noise from irrelevant cross-store heterogeneity while pooling enough data for robust generalisation.

For instance, a cluster of large urban stores with high competition density will likely show stronger response to BOGO and Flat Discount (price-sensitive, high-footfall environments) while a cluster of small rural stores may respond better to Loyalty Points (relationship-driven, repeat-customer base). A cluster model captures these distinct response curves without needing a single model to handle all contexts simultaneously. This approach also makes retraining more efficient — only the affected cluster model needs updating when one segment's behaviour changes.

---

## B2. Data and EDA Strategy

### B2(a) — Joining the Four Tables

The four source tables are: `transactions`, `store_attributes`, `promotion_details`, and `calendar`.

**Join logic:**

1. **Aggregate `transactions` to store-month grain:** For each `(store_id, month)` combination, compute `sum(items_sold)` as the target and any other relevant aggregates (transaction count, average basket size). This produces one row per store per month — the grain at which recommendations will be made.

2. **Join `store_attributes` on `store_id` (many-to-one):** Each store-month row inherits the store's static attributes: `store_size`, `location_type`, `competition_density`, and footfall. These are assumed to be stable month-to-month; if they change over time, a slowly changing dimension approach is needed.

3. **Join `promotion_details` on `(store_id, month)` (one-to-one at modelling grain):** Each row gets the `promotion_type` deployed that store-month. This is the key decision variable.

4. **Join `calendar` on `month` or `date` (many-to-one):** Each row gets the month's festival flags, weekend day count, and any other temporal markers from the calendar table.

**Grain of the final modelling dataset:** One row = one store × one calendar month × one promotion type. This is the unit at which the model makes predictions and the retailer makes deployment decisions.

**Key aggregations:** `items_sold` summed within store-month; `is_festival` as a binary flag set to 1 if any day in the month is a festival; `weekend_days` as the count of weekend days in the month.

---

### B2(b) — EDA Before Modelling

**1. Promotion performance distribution by promotion type**
Plot box plots of `items_sold` grouped by `promotion_type`. In this dataset, BOGO achieves the highest mean (293.4 items), followed by Flat Discount (282.8), Free Gift (269.1), Category Offer (256.3), and Loyalty Points (251.5). The analysis would reveal whether these differences are statistically significant or whether the variance within each promotion type is large enough to overlap. If distributions overlap heavily, promotion type alone is not a strong predictor and interaction effects with store type will matter more. This finding directly informs how much weight to place on `promotion_type` as a standalone feature vs as an interaction term.

**2. Items sold by location type and store size**
Plot grouped bar charts of mean `items_sold` by `location_type` and `store_size` separately, then a heatmap of their interaction. Urban stores average 301.5 items vs 263.0 semi-urban and 235.9 rural; large stores average 313.5 vs 276.7 medium and 232.4 small. These structural gaps are larger than most promotion effects, confirming that store characteristics are essential features. The heatmap would reveal whether small urban stores behave differently from large rural ones — informing the cluster modelling strategy in B1(c).

**3. Festival and weekend uplift analysis**
Plot mean `items_sold` for festival vs non-festival days (343.6 vs 263.3 — a 30.5% uplift) and weekend vs weekday (302.8 vs 260.8 — a 16.1% uplift). This quantifies the magnitude of temporal demand spikes. If festival uplift is this large, it must be captured as a feature — and more importantly, the model must not be evaluated on a test set that contains an unusually high or low proportion of festivals compared to the training set. This finding informs both feature engineering (include `is_festival` prominently) and evaluation design (check festival distribution across train and test periods).

**4. Promotion × location interaction heatmap**
Create a heatmap of mean `items_sold` with `promotion_type` on one axis and `location_type` on the other. If BOGO performs strongly in urban stores but weakly in rural ones, the interaction term `promotion_type × location_type` should be added as an explicit feature or the cluster-based modelling approach from B1(c) should be adopted. This is the single most important EDA chart for validating the modelling strategy — a flat heatmap (no interaction) supports a global model; a strongly varying heatmap supports cluster or store-specific models.

---

### B2(c) — Handling the 80% No-Promotion Imbalance

In this regression context, the imbalance means the model is trained predominantly on the baseline (no-promotion) demand pattern. With only 20% of transactions occurring under a promotion, the model has far fewer examples of promoted behaviour and will fit the no-promotion relationship much more tightly.

**Effects on the model:** Predicted items_sold during promotional periods will be biased toward the no-promotion baseline. Feature importance scores for `promotion_type` will be understated relative to their true causal effect because the model sees so few promoted examples. The model may also struggle to distinguish between the five promotion types if each appears in only 4% of total records.

**Steps to address this:**

1. **Oversample promoted records at training time:** Use stratified sampling to oversample the 20% promoted rows, or equivalently apply higher sample weights to promoted rows in the loss function, so that promoted and non-promoted examples contribute more equally to model fitting.

2. **Separate uplift modelling:** Train a baseline model to predict items_sold under no-promotion conditions, and a separate uplift model to predict the incremental effect of each promotion type over the baseline. At deployment, the recommendation is `argmax(baseline + uplift_i)` over the five promotion types. This explicitly isolates the promotion signal from the background demand level.

3. **Feature engineering for context:** Add a binary `has_promotion` flag and explicit interaction terms between `promotion_type` and `location_type`, `store_size`, and `is_festival` to ensure the model can learn that promotion effects are conditional on store context — not just that promotions exist.

---

## B3. Model Evaluation and Deployment

### B3(a) — Train-Test Split Setup and Evaluation Metrics

**Split setup:**
With three years of monthly data across 50 stores (approximately 1,800 store-month observations), we use a **temporal split**: train on years 1 and 2 (months 1–24, approximately 1,200 rows), test on year 3 (months 25–36, approximately 600 rows). Within the training period we use **walk-forward cross-validation**: each fold trains on all data up to month T and validates on month T+1, advancing one month at a time. This simulates the actual deployment setting — predicting one month ahead — and gives a more honest estimate of generalisation than k-fold cross-validation, which would mix past and future data within each fold.

**Why a random split is inappropriate:**
A random split allows the model to train on future months and be tested on past months, constituting temporal data leakage. Evaluation scores from a random split will be inflated and will not reflect the model's ability to predict months it has never seen. In production, the model always predicts into the future from a historical training window — the evaluation must replicate this constraint exactly.

**Evaluation metrics:**

- **RMSE (Root Mean Squared Error):** Measures prediction error in units of items sold, penalising large mispredictions disproportionately. Useful for detecting outlier-level failures — e.g., predicting 200 items when 400 sell during a festival month. In this dataset, a baseline RMSE of ~27 items on a mean of 272 items represents about 10% relative error, which is a useful reference.

- **MAE (Mean Absolute Error):** Average absolute error with equal weight to all errors. More interpretable for the business team: "on average the model is off by 21 items per store-month." Preferred for communicating performance in business reviews.

- **MAPE (Mean Absolute Percentage Error):** Normalises error by actual volume, making it comparable across stores with very different sales levels. A 21-item error means something very different for a large urban store selling 400 items vs a small rural store selling 150. MAPE makes these comparable.

- **Promotion ranking accuracy:** For each store-month in the test set, check whether the model's top-ranked promotion (highest predicted items_sold) matches the historically best-performing promotion. This directly measures the quality of the recommendation — the actual business output — rather than prediction accuracy in isolation.

---

### B3(b) — Using Feature Importance to Explain Different Recommendations

The model recommends Loyalty Points Bonus for Store 12 in December and Flat Discount for Store 12 in March. To investigate and communicate this to the marketing team:

**Step 1 — Global feature importance as context.**
The Random Forest identifies `is_festival` (17.3% importance) and `store_size` (16.8% for small stores) as the top two drivers of `items_sold` across all stores. This establishes that temporal demand context and structural store capacity are the dominant predictors — more influential than promotion type on its own.

**Step 2 — SHAP values for per-prediction explanation.**
Apply SHAP (SHapley Additive exPlanations) to decompose the model's prediction for each of the two store-month scenarios into individual feature contributions. For Store 12 in December, the SHAP analysis would likely show that `is_festival = 1` contributes a large positive uplift to Loyalty Points predictions — historically, loyalty reward mechanics perform strongly during peak festive periods because customers accumulate points for post-season redemption. For Store 12 in March, `is_festival = 0` removes this advantage, and the lower baseline demand period makes price-sensitive mechanics like Flat Discount relatively more effective at driving volume from footfall that needs an incentive to convert.

**Step 3 — Present a feature contribution table to the marketing team:**

| Feature | Value (Dec) | Impact on Loyalty Points | Value (Mar) | Impact on Flat Discount |
|---|---|---|---|---|
| is_festival | 1 | Strong positive | 0 | Neutral |
| month | 12 (peak) | Positive | 3 (low season) | Neutral |
| is_weekend | context-dependent | Moderate | context-dependent | Moderate |
| competition_density | store's value | Context | store's value | Context |

This table makes the model's reasoning transparent, allows the marketing team to validate or override based on domain knowledge (e.g., if a new competitor opens near Store 12 in December), and builds trust in the system by showing that the model's logic is aligned with retail intuition.

---

### B3(c) — End-to-End Deployment Process

**1. Saving the model**
After final training, serialise the complete scikit-learn Pipeline object (which includes both the ColumnTransformer preprocessor and the fitted model) using `joblib.dump(pipeline, 'promotion_model_v1.pkl')`. Store the artefact in a versioned model registry — this can be as simple as a dated folder in cloud storage (e.g., `s3://models/promotion/2024-12/`) or a dedicated system such as MLflow Model Registry. Alongside the model file, save: the training data schema (feature names and dtypes), the feature engineering steps applied before the pipeline (date extraction, is_month_end), and the training data date range. These are needed to validate incoming data and reproduce the pipeline exactly at inference time.

**2. Preparing and feeding monthly data**
At the start of each month, an automated pipeline executes the following steps:
- Pull the latest store attributes, competition density scores, and calendar data for the upcoming month from source systems
- For each of the 50 stores × 5 promotion options, construct one feature row per combination (250 rows total)
- Apply the same feature engineering steps used during training: parse dates, extract year/month/day_of_week/is_month_end, apply is_festival and is_weekend flags from the calendar
- Load the saved pipeline with `joblib.load()` and call `pipeline.predict()` on the 250-row inference dataset — the fitted preprocessor inside the pipeline handles encoding and scaling automatically without being re-fitted
- For each store, select the promotion type with the highest predicted items_sold as the recommendation
- Output a recommendation table (store_id → promotion_type) to the marketing dashboard

The preprocessor must never be re-fitted on new data — only the model artefact from training is used. Re-fitting would cause the scaling and encoding to shift, breaking consistency with the training distribution.

**3. Monitoring for model degradation**
Deploy the following checks on a monthly basis after recommendations are made and actuals are observed:

- **Actual vs predicted tracking:** After each month closes, record actual items_sold against predictions for every store. Compute rolling RMSE and MAE over a 3-month window. If rolling RMSE exceeds 1.3× the baseline training RMSE (approximately 35 items given a training RMSE of ~27), flag for review.

- **Feature distribution drift:** Monitor the distribution of key input features — `competition_density`, promotion type frequencies, store-level footfall — each month using Population Stability Index (PSI). A PSI above 0.2 on any feature indicates significant drift from the training distribution and warrants investigation before the next inference cycle.

- **Prediction distribution drift:** Compare the distribution of predicted items_sold each month to the distribution observed during training using a KS test. A significant shift suggests the model is operating in a regime it was not trained on.

- **Retraining schedule:** Retrain the model quarterly using an expanding window — add new months to the training data without discarding historical data. Trigger an out-of-cycle retrain if: rolling RMSE exceeds the threshold for two consecutive months; a major structural change occurs (new store format, new promotion type added, significant pricing strategy change); or feature drift monitoring detects PSI > 0.2 on more than two features simultaneously.
