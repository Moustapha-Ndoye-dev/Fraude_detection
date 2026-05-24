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


def customer_segment_profile(segments: pd.DataFrame) -> pd.DataFrame:
    return (
        segments.groupby("segment")
        .agg(
            clients=("ID", "size"),
            revenu_median=("Income", "median"),
            depense_moyenne=("Total_Spend", "mean"),
            achats_moyens=("Total_Purchases", "mean"),
            achats_web_moyens=("NumWebPurchases", "mean"),
            achats_promo_moyens=("NumDealsPurchases", "mean"),
            recence_moyenne=("Recency", "mean"),
            reponse_campagne=("Response", "mean"),
        )
        .reset_index()
        .sort_values("segment")
    )


def label_customer_segments(profile: pd.DataFrame) -> pd.DataFrame:
    labeled = profile.copy()
    labeled["label_metier"] = ""
    used_indexes: set[int | str] = set()

    def assign_label(label: str, index: int | str) -> None:
        labeled.loc[index, "label_metier"] = label
        used_indexes.add(index)

    premium_idx = labeled["depense_moyenne"].idxmax()
    assign_label("Clients premium", premium_idx)

    dormant_candidates = labeled.loc[~labeled.index.isin(used_indexes)]
    assign_label("Clients dormants", dormant_candidates["recence_moyenne"].idxmax())

    digital_candidates = labeled.loc[~labeled.index.isin(used_indexes)]
    web_promo_score = digital_candidates["achats_web_moyens"].rank() + digital_candidates["achats_promo_moyens"].rank()
    assign_label("Digitaux et promotions", web_promo_score.idxmax())

    labeled.loc[labeled["label_metier"] == "", "label_metier"] = "Clients economes"
    return labeled


def segment_by_label(profile: pd.DataFrame, label: str) -> pd.Series:
    return profile.loc[profile["label_metier"] == label].iloc[0]
