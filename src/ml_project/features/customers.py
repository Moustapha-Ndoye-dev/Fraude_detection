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

BUSINESS_SEGMENT_LABELS = (
    "Clients premium",
    "Clients dormants",
    "Digitaux et promotions",
    "Clients economes",
)


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
    if segments.empty or "segment" not in segments.columns:
        return pd.DataFrame()

    valid_segments = segments.dropna(subset=["segment"]).copy()
    if valid_segments.empty:
        return pd.DataFrame()

    return (
        valid_segments.groupby("segment")
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


def _default_segment_row(label: str = "") -> pd.Series:
    return pd.Series(
        {
            "segment": 0,
            "clients": 0,
            "revenu_median": 0.0,
            "depense_moyenne": 0.0,
            "achats_moyens": 0.0,
            "achats_web_moyens": 0.0,
            "achats_promo_moyens": 0.0,
            "recence_moyenne": 0.0,
            "reponse_campagne": 0.0,
            "label_metier": label,
        }
    )


def _safe_first_row(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return _default_segment_row()
    return df.iloc[0]


def _row_with_max(profile: pd.DataFrame, column: str) -> pd.Series:
    if profile.empty:
        return _default_segment_row()
    if column not in profile.columns:
        return _safe_first_row(profile)

    values = pd.to_numeric(profile[column], errors="coerce")
    if values.notna().any():
        return profile.loc[values.idxmax()]
    return _safe_first_row(profile)


def _resolve_segment_row(profile: pd.DataFrame, label: str) -> pd.Series:
    if profile.empty:
        return _default_segment_row(label)

    if "label_metier" in profile.columns:
        matches = profile.loc[profile["label_metier"] == label]
        if not matches.empty:
            row = _safe_first_row(matches)
            row = row.copy()
            row["label_metier"] = label
            return row

    if label == "Clients premium":
        row = _row_with_max(profile, "depense_moyenne")
    elif label == "Clients dormants":
        row = _row_with_max(profile, "recence_moyenne")
    elif label == "Digitaux et promotions":
        score = profile["achats_web_moyens"].fillna(0) + profile["achats_promo_moyens"].fillna(0)
        row = profile.loc[score.idxmax()] if not profile.empty else _default_segment_row(label)
    elif label == "Clients economes":
        row = _row_with_max(profile, "clients")
    else:
        row = _safe_first_row(profile)

    row = row.copy()
    row["label_metier"] = label
    return row


def label_customer_segments(profile: pd.DataFrame) -> pd.DataFrame:
    labeled = profile.copy().reset_index(drop=True)
    if labeled.empty:
        labeled["label_metier"] = pd.Series(dtype=str)
        return labeled

    labeled["label_metier"] = "Clients economes"
    used_indexes: set[int] = set()

    def pick_row(column: str) -> int | None:
        candidates = labeled.loc[~labeled.index.isin(used_indexes)]
        if candidates.empty:
            return None
        values = candidates[column]
        if values.notna().any():
            return int(values.idxmax())
        return int(candidates.index[0])

    def pick_digital_row() -> int | None:
        candidates = labeled.loc[~labeled.index.isin(used_indexes)]
        if candidates.empty:
            return None
        score = candidates["achats_web_moyens"].fillna(0) + candidates["achats_promo_moyens"].fillna(0)
        return int(score.idxmax())

    assignments = [
        ("Clients premium", lambda: pick_row("depense_moyenne")),
        ("Clients dormants", lambda: pick_row("recence_moyenne")),
        ("Digitaux et promotions", pick_digital_row),
    ]

    for label, picker in assignments:
        row_index = picker()
        if row_index is None:
            continue
        labeled.loc[row_index, "label_metier"] = label
        used_indexes.add(row_index)

    return labeled


def build_segment_lookup(profile: pd.DataFrame, *, already_labeled: bool = False) -> dict[str, pd.Series]:
    labeled = profile if already_labeled else label_customer_segments(profile)
    lookup: dict[str, pd.Series] = {}

    if not labeled.empty and "label_metier" in labeled.columns:
        for label, group in labeled.groupby("label_metier", sort=False):
            label_text = str(label).strip()
            if label_text:
                row = _safe_first_row(group).copy()
                row["label_metier"] = label_text
                lookup[label_text] = row

    for label in BUSINESS_SEGMENT_LABELS:
        if label not in lookup:
            lookup[label] = _resolve_segment_row(labeled, label)

    return lookup


def segment_by_label(profile: pd.DataFrame, label: str) -> pd.Series:
    lookup = build_segment_lookup(profile, already_labeled="label_metier" in profile.columns)
    return lookup.get(label, _default_segment_row(label))
