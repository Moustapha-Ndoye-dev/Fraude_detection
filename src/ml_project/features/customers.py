from __future__ import annotations

import pandas as pd


SPEND_COLUMNS = [
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds",
]

PURCHASE_COLUMNS = [
    "NumDealsPurchases",
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases",
]

CAMPAIGN_COLUMNS = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5", "Response"]


def add_customer_features(df: pd.DataFrame, reference_year: int = 2026) -> pd.DataFrame:
    out = df.copy()
    out["Dt_Customer"] = pd.to_datetime(out["Dt_Customer"], dayfirst=True, errors="coerce")
    out["Age"] = reference_year - out["Year_Birth"]
    out["Customer_Tenure_Days"] = (out["Dt_Customer"].max() - out["Dt_Customer"]).dt.days
    out["Children_Total"] = out["Kidhome"] + out["Teenhome"]
    out["Total_Spend"] = out[SPEND_COLUMNS].sum(axis=1)
    out["Total_Purchases"] = out[PURCHASE_COLUMNS].sum(axis=1)
    out["Campaigns_Accepted"] = out[CAMPAIGN_COLUMNS].sum(axis=1)
    out["Spend_Per_Purchase"] = out["Total_Spend"] / out["Total_Purchases"].replace(0, 1)
    return out


def customer_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    featured = add_customer_features(df)
    return featured.drop(columns=["ID", "Dt_Customer"], errors="ignore")
