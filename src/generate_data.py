"""
generate_data.py
-----------------
Generates a realistic synthetic telecom customer-churn dataset.

Why synthetic data?
Public churn datasets (e.g. Kaggle's Telco Customer Churn) are widely
recycled across student projects, which makes plagiarism/originality
checks and portfolio differentiation harder. This script instead
*programmatically generates* a dataset with the same statistical
structure and feature relationships you'd see in a real telecom churn
problem (tenure, contract type, monthly charges, support usage, etc.),
with churn probability driven by a realistic underlying model plus
noise. This keeps the project fully reproducible end-to-end (anyone
cloning the repo can regenerate the exact same data) while still
requiring genuine EDA, feature engineering, and modeling work.

Run:
    python src/generate_data.py
Output:
    data/customer_churn.csv
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_CUSTOMERS = 3000


def generate_customers(n=N_CUSTOMERS, seed=RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    customer_id = [f"CUST-{100000 + i}" for i in range(n)]

    gender = rng.choice(["Male", "Female"], size=n)
    senior_citizen = rng.choice([0, 1], size=n, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], size=n, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], size=n, p=[0.30, 0.70])

    # Tenure in months - skewed toward newer + very long-term customers
    tenure = np.clip(rng.exponential(scale=24, size=n), 0, 72).round().astype(int)

    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"], size=n, p=[0.55, 0.24, 0.21]
    )
    paperless_billing = rng.choice(["Yes", "No"], size=n, p=[0.59, 0.41])
    payment_method = rng.choice(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        size=n,
        p=[0.34, 0.23, 0.22, 0.21],
    )

    internet_service = rng.choice(
        ["DSL", "Fiber optic", "No"], size=n, p=[0.34, 0.44, 0.22]
    )

    def dependent_service(base_p_yes=0.5):
        out = []
        for svc in internet_service:
            if svc == "No":
                out.append("No internet service")
            else:
                out.append(rng.choice(["Yes", "No"], p=[base_p_yes, 1 - base_p_yes]))
        return np.array(out)

    online_security = dependent_service(0.35)
    online_backup = dependent_service(0.40)
    device_protection = dependent_service(0.40)
    tech_support = dependent_service(0.32)
    streaming_tv = dependent_service(0.45)
    streaming_movies = dependent_service(0.45)

    phone_service = rng.choice(["Yes", "No"], size=n, p=[0.90, 0.10])
    multiple_lines = np.array(
        [
            "No phone service" if p == "No" else rng.choice(["Yes", "No"], p=[0.42, 0.58])
            for p in phone_service
        ]
    )

    # Monthly charges: base + add-ons, with fiber costing more
    base = np.where(internet_service == "Fiber optic", 70, np.where(internet_service == "DSL", 45, 20))
    addon_cost = (
        (online_security == "Yes").astype(int) * 5
        + (online_backup == "Yes").astype(int) * 5
        + (device_protection == "Yes").astype(int) * 4
        + (tech_support == "Yes").astype(int) * 5
        + (streaming_tv == "Yes").astype(int) * 8
        + (streaming_movies == "Yes").astype(int) * 8
        + (phone_service == "Yes").astype(int) * 8
    )
    noise = rng.normal(0, 4, size=n)
    monthly_charges = np.clip(base + addon_cost + noise, 18, 130).round(2)

    total_charges = np.clip(monthly_charges * tenure + rng.normal(0, 20, size=n), 0, None).round(2)

    # ---- Churn probability model (logistic combination of risk factors) ----
    contract_risk = np.where(contract == "Month-to-month", 1.0, np.where(contract == "One year", 0.35, 0.1))
    tenure_risk = np.clip(1 - tenure / 60, 0, 1)
    fiber_risk = (internet_service == "Fiber optic").astype(float) * 0.35
    no_tech_support_risk = (tech_support == "No").astype(float) * 0.25
    echeck_risk = (payment_method == "Electronic check").astype(float) * 0.25
    price_risk = np.clip((monthly_charges - 60) / 70, -0.3, 0.6)
    senior_risk = senior_citizen.astype(float) * 0.15
    paperless_risk = (paperless_billing == "Yes").astype(float) * 0.1
    partner_protective = (partner == "Yes").astype(float) * -0.15
    dependents_protective = (dependents == "Yes").astype(float) * -0.1

    risk_score = (
        1.8 * contract_risk
        + 1.6 * tenure_risk
        + fiber_risk
        + no_tech_support_risk
        + echeck_risk
        + price_risk
        + senior_risk
        + paperless_risk
        + partner_protective
        + dependents_protective
        + rng.normal(0, 0.4, size=n)
    )

    # Calibrate the intercept so the overall churn rate lands close to a
    # realistic ~26-27% (in line with published telecom churn benchmarks),
    # via a quick bisection search rather than a hand-picked constant.
    target_rate = 0.265
    lo, hi = -6.0, 2.0
    for _ in range(40):
        mid = (lo + hi) / 2
        rate = (1 / (1 + np.exp(-(risk_score + mid)))).mean()
        if rate > target_rate:
            hi = mid
        else:
            lo = mid
    intercept = (lo + hi) / 2

    logit = risk_score + intercept
    churn_prob = 1 / (1 + np.exp(-logit))
    churn = (rng.uniform(0, 1, size=n) < churn_prob).astype(int)
    churn_label = np.where(churn == 1, "Yes", "No")

    df = pd.DataFrame(
        {
            "customerID": customer_id,
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Churn": churn_label,
        }
    )
    return df


if __name__ == "__main__":
    df = generate_customers()
    df.to_csv("data/customer_churn.csv", index=False)
    print(f"Generated {len(df)} rows -> data/customer_churn.csv")
    print(f"Churn rate: {(df['Churn'] == 'Yes').mean():.2%}")
