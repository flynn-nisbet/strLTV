The dashboard is powered by two analytically distinct datasets that are produced by the same underlying pipeline but serve different purposes. Both datasets cover the same three call centers (Charlotte, Durban, Jamaica) and the same plan categories (Fixed, Bundled, Tiered), and both run every call through an identical enrichment and usage estimation process before diverging into their respective analytical outputs. The separation is intentional: one dataset is old enough to have complete survival observations, while the other is recent enough to be actionable for current LTV measurement.

---

### Survival Dataset — Late 2024 Cohort (`survival_call_level.csv`)

**Date range:** September 1, 2024 – December 1, 2024

This cohort was chosen specifically because orders placed in this window have had enough time to accumulate a full 12 months of observable post-sale tenure by the time the pipeline runs. The primary output is a set of binary survival flags (M1 through M12) indicating whether each activated customer was still on plan at each 30-day milestone, plus an M6→M12 completion rate that measures how many customers who made it to six months ultimately reached a full year. These survival rates are the historical benchmark — they tell us, broken down by every combination of agent, product, and customer dimension, what fraction of customers in each segment actually stayed.

The survival flags are computed as: a customer "survives" month N if they activated and their observed tenure (capped at the current date for still-active customers) is at least N × 30 days. Customers who never activated receive null survival values and are excluded from rate calculations.

### LTV Dataset — 2025 Cohort (`ltv_call_level.csv`)

**Date range:** March 1, 2025 – July 1, 2025

This cohort represents recent sales where the business question is not "what happened historically" but "what is each order actually worth." Because these orders are only months old, survival flags are capped at 6 months rather than 12 — the back half of the survival curve simply hasn't had time to materialize. The core output here is a derived LTV per order, computed as the sum of two components: an upfront bounty (paid at order, activation, or first bill depending on the partner) and a survival-weighted trailing revenue component derived from actual predicted kWh consumption multiplied by the partner's per-thousand-kWh mil rate. The LTV is compared against GCV (Gross Contract Value) on every order to surface the gap between what was contracted and what the order is modeled to actually generate.

---

Shared Pipeline: Rentcast Enrichment and Usage Estimation

Both datasets run through identical enrichment and modeling steps before producing their respective outputs. This is the most computationally intensive part of the pipeline.

### Step 1: Call Population Pull

The pipeline begins by querying the call population for the relevant date window from the core data warehouse — pulling orders, post-sale lifecycle data, agent call records, and plan metadata. For each qualifying call, the customer's captured service address is retrieved from the address event log, taking the most recent address event per call.

### Step 2: Rentcast API Enrichment

Each call's service address is sent to a hosted Rentcast API endpoint (`/address`) via a parallelized thread pool (10 concurrent workers) to retrieve property-level characteristics for the home. The API returns a JSON payload matched to the closest known property record. The pipeline always takes the first result, which corresponds to the zero-distance exact match where available.

From the raw JSON response, the following fields are extracted and flattened:

- **Structural:** property type, bedrooms, bathrooms, square footage, lot size, year built, floor count
- **Systems:** heating type, cooling type, garage type, fireplace type, exterior type, foundation type, roof type, architecture type
- **Occupancy:** owner-occupied flag
- **Financial:** latest tax assessment year, land value, total assessed value, improvement value

Calls that fail the API call (timeout, non-200 response, or no matching property) return null Rentcast features and are carried through the pipeline with imputed values downstream.

### Step 3: Month Scaffolding

For each call, the pipeline generates a 12-row scaffold of future monthly periods starting from the call date, one row per month (month 1 through 12). Each row includes the usage start and end dates for that month, the month name and number (used as a seasonal feature), and the number of days falling in the start and end calendar months (to handle partial-month boundary effects accurately). The Rentcast property features are then joined onto this scaffold so that every call has 12 rows, each representing one potential month of energy consumption.

### Step 4: Weather Feature Enrichment

Temperature and weather data is joined to each month row based on the property's location and the relevant usage period. The weather feature set includes daily temperature distributions (bucketed into Celsius bands: below 10°, 10–15°, 15–20°, 20–25°, 25–30°, 30–35°, above 35°), high and low temps, precipitation, humidity, wind speed, cloud cover, UV index, and counts of snow, rain, dry, and sunny days. Both the current period and the prior period's weather features are included, since prior-month conditions affect usage patterns for heating and cooling systems with thermal inertia.

### Step 5: Call-Level Survey Feature Enrichment

Where available, agent-captured call data supplements the property features. For movers (customers switching to a new address), demographic inputs such as household size, number of bedrooms, whether anyone works from home, and the type of additional appliances are incorporated. For switchers (customers already at the address), self-reported usage history is pulled in: whether their usage is consistent or fluctuating, and their reported usage ranges (last month, seasonal highs, seasonal lows). These inputs directly influence the model's predictions because they encode information the property record cannot — occupancy patterns and appliance load that vary by household even within identical homes.

Calls with no survey data at all (a meaningful fraction of the population, particularly among movers) fall into a null-imputation path where the model relies entirely on the property and weather features.

### Step 6: Feature Engineering and Preprocessing

Before scoring, the pipeline runs a multi-step preprocessing sequence:

1. **Numeric conversion** — ensures all numerical variables are typed correctly, coercing non-numeric strings to null.
2. **Feature engineering** — derives additional features from existing ones, including interaction terms and transformed representations of raw inputs.
3. **Data cleaning** — resolves known data quality issues such as implausible values or inconsistent categorical encodings.
4. **Temperature-based consumption prediction** — generates preliminary consumption estimates from temperature indices as an additional input feature, giving the model a physics-informed prior before it applies learned corrections.

### Step 7: Usage Model Scoring

The preprocessed feature set is passed to the `compass-consumption-model`, a LightGBM regressor versioned in MLflow on Databricks. The model is loaded at the latest registered version and scores each of the 12 monthly rows per call, producing a predicted kWh value for each future month. These predictions are returned as `residual_m1` through `residual_m12` — the expected monthly energy consumption for each call under each future month's conditions.

The model was trained on historical energy usage data with the same feature set and uses gradient boosting over decision trees, which handles the mixed categorical/numerical feature space and non-linear interactions between weather, property characteristics, and usage patterns without requiring explicit interaction term specification.

---

### Post-Scoring Divergence

After usage model scoring, the two pipelines diverge.

**For the survival dataset**, the predicted usage values are aggregated across the 12 months per call to produce a single average monthly kWh figure, which is then bucketed into usage bands (0–500, 500–1000, 1000–1200, 1200–2000, 2000+ kWh) and used as a customer dimension for survival segmentation. Consistency is measured by whether the modal usage band covers at least 8 of the 12 predicted months. The survival flags themselves are computed from post-sale tenure data, not from the usage model — the usage model's contribution here is purely the customer usage segmentation dimensions.

**For the LTV dataset**, the per-month kWh predictions feed directly into the financial calculation. Each month's predicted kWh is multiplied by the partner's mil rate (dollars per 1,000 kWh) and then multiplied by the survival flag for that month (1 if the customer was still active, 0 if not), summing to a survival-weighted trailing revenue figure over the 6-month observation window. This is added to the upfront bounty (where earned) to produce the derived LTV per order. The mil rate varies by partner and in some cases by plan type — for example, TXU pays differently on tiered plans and low-deposit plans versus standard plans. Partners that pay flat per-activation fees with no usage component (Reliant, Direct Energy, Green Mountain, Cirro, Discount Power) carry a 0 mil rate, meaning their trailing revenue is zero and LTV equals the upfront bounty alone.

---

### Agent Quartile Dimensions

Both datasets include two agent-level quartile rankings computed from within the cohort window.

**Conversion quartile** ranks agents by their net conversion rate (orders placed divided by total calls handled). Q1 represents the top 25% of converters. This is a volume/sales efficiency metric.

**Survival quartile** ranks agents by the average number of days their customers stayed on plan, measured across all activated orders. Q1 represents the agents whose customers lasted the longest on average. This is a quality/retention metric, and its independence from conversion quartile makes the cross-tabulation of the two meaningful — an agent can be a high converter whose customers churn quickly, or a modest converter whose customers stay for a long time.

---

### Key Design Choices

**Why two separate cohorts rather than one?** A single cohort cannot simultaneously have enough historical depth for 12-month survival analysis and be recent enough for actionable LTV measurement. Orders need to be roughly 12–15 months old to have settled survival curves; orders need to be recent to reflect current agent behavior, product mix, and customer demographics. The two-cohort design lets each dataset do what it does best without compromising the other.

**Why cap the LTV dataset at 6 months?** Orders from March–July 2025 have at most 6 months of observable tenure as of the pipeline run date. Including months 7–12 would produce survival flags that are structurally zero for almost all orders regardless of actual attrition, which would systematically understate LTV and make the trailing revenue calculation misleading. The 6-month cap ensures every flag in the LTV dataset reflects a period that the cohort has actually had time to live through.

**Why use predicted kWh rather than actual metered usage?** Actual metered usage for a given customer is not available at the time of sale or even in the early months of a plan. The usage model provides a consistent, feature-driven estimate of consumption that can be applied to all orders in the same way, making the LTV calculation reproducible and comparable across segments. For mature orders where actual usage data eventually becomes available, the predictions can be validated against actuals — that comparison is itself a diagnostic for model accuracy by segment.
