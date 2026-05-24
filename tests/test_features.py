from __future__ import annotations

import pandas as pd

from ml_project.features.customers import add_customer_features
from ml_project.features.fraud import add_fraud_features


def test_add_fraud_features_creates_expected_columns():
    df = pd.DataFrame(
        [
            {
                "type": "TRANSFER",
                "amount": 100.0,
                "nameOrig": "C1",
                "oldbalanceOrg": 100.0,
                "newbalanceOrig": 0.0,
                "nameDest": "C2",
                "oldbalanceDest": 0.0,
                "newbalanceDest": 100.0,
            }
        ]
    )
    out = add_fraud_features(df)
    assert out.loc[0, "emptied_origin"] == 1
    assert "log_amount" in out.columns


def test_add_customer_features_creates_total_spend():
    df = pd.DataFrame(
        [
            {
                "Year_Birth": 1990,
                "Kidhome": 1,
                "Teenhome": 0,
                "Dt_Customer": "01/01/2020",
                "MntWines": 10,
                "MntFruits": 5,
                "MntMeatProducts": 20,
                "MntFishProducts": 0,
                "MntSweetProducts": 0,
                "MntGoldProds": 2,
                "NumDealsPurchases": 1,
                "NumWebPurchases": 1,
                "NumCatalogPurchases": 0,
                "NumStorePurchases": 2,
                "AcceptedCmp1": 0,
                "AcceptedCmp2": 0,
                "AcceptedCmp3": 1,
                "AcceptedCmp4": 0,
                "AcceptedCmp5": 0,
                "Response": 1,
            }
        ]
    )
    out = add_customer_features(df)
    assert out.loc[0, "Total_Spend"] == 37
    assert out.loc[0, "Campaigns_Accepted"] == 2
