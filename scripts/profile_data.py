import csv
import math
from collections import Counter


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    idx = (len(values) - 1) * p / 100
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - idx) + values[hi] * (idx - lo)


def round_dict(digits=2, **items):
    return {key: round(value, digits) if value is not None else None for key, value in items.items()}


def profile_fraud(path="detection_fraude.csv"):
    num_cols = [
        "step",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
    ]
    stats = {c: {"n": 0, "sum": 0.0, "min": None, "max": None} for c in num_cols}
    missing = Counter()
    type_counts = Counter()
    fraud_by_type = Counter()
    rows = fraud = flagged = flagged_fraud = 0
    amounts = []
    fraud_amounts = []
    normal_amounts = []
    full_balance_fraud = 0
    fraud_transfer_cashout = 0
    fraud_other = 0
    orig_prefix = Counter()
    dest_prefix = Counter()
    unique_orig = set()
    unique_dest = set()

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        fields = reader.fieldnames or []
        for row in reader:
            rows += 1
            for key, value in row.items():
                if value is None or value == "":
                    missing[key] += 1

            transaction_type = row["type"]
            type_counts[transaction_type] += 1
            is_fraud = int(row["isFraud"])
            is_flagged = int(row["isFlaggedFraud"])
            fraud += is_fraud
            flagged += is_flagged
            flagged_fraud += 1 if is_fraud and is_flagged else 0
            if is_fraud:
                fraud_by_type[transaction_type] += 1

            amount = float(row["amount"])
            amounts.append(amount)
            if is_fraud:
                fraud_amounts.append(amount)
            else:
                normal_amounts.append(amount)

            if (
                is_fraud
                and abs(amount - float(row["oldbalanceOrg"])) < 0.01
                and abs(float(row["newbalanceOrig"])) < 0.01
            ):
                full_balance_fraud += 1
            if is_fraud and transaction_type in ("TRANSFER", "CASH_OUT"):
                fraud_transfer_cashout += 1
            if is_fraud and transaction_type not in ("TRANSFER", "CASH_OUT"):
                fraud_other += 1

            if row["nameOrig"]:
                orig_prefix[row["nameOrig"][0]] += 1
                unique_orig.add(row["nameOrig"])
            if row["nameDest"]:
                dest_prefix[row["nameDest"][0]] += 1
                unique_dest.add(row["nameDest"])

            for column in num_cols:
                value = float(row[column])
                col_stats = stats[column]
                col_stats["n"] += 1
                col_stats["sum"] += value
                col_stats["min"] = value if col_stats["min"] is None or value < col_stats["min"] else col_stats["min"]
                col_stats["max"] = value if col_stats["max"] is None or value > col_stats["max"] else col_stats["max"]

    print("FRAUD_DATASET")
    print("rows", rows)
    print("cols", len(fields), fields)
    print("missing", dict(missing))
    print("fraud", fraud, "normal", rows - fraud, "fraud_rate_pct", round(fraud / rows * 100, 4))
    print("flagged", flagged, "flagged_fraud", flagged_fraud)
    print("type_counts", dict(type_counts))
    print("fraud_by_type", dict(fraud_by_type))
    print("fraud_rate_by_type_pct", {k: round(fraud_by_type[k] / v * 100, 4) for k, v in type_counts.items()})
    print(
        "amount_quantiles",
        round_dict(
            min=percentile(amounts, 0),
            p50=percentile(amounts, 50),
            p90=percentile(amounts, 90),
            p99=percentile(amounts, 99),
            max=percentile(amounts, 100),
        ),
    )
    print(
        "fraud_amount_quantiles",
        round_dict(
            min=percentile(fraud_amounts, 0),
            p50=percentile(fraud_amounts, 50),
            p90=percentile(fraud_amounts, 90),
            p99=percentile(fraud_amounts, 99),
            max=percentile(fraud_amounts, 100),
        ),
    )
    print(
        "normal_amount_quantiles",
        round_dict(
            min=percentile(normal_amounts, 0),
            p50=percentile(normal_amounts, 50),
            p90=percentile(normal_amounts, 90),
            p99=percentile(normal_amounts, 99),
            max=percentile(normal_amounts, 100),
        ),
    )
    print(
        "num_stats",
        {
            column: {
                "min": round(col_stats["min"], 2),
                "mean": round(col_stats["sum"] / col_stats["n"], 2),
                "max": round(col_stats["max"], 2),
            }
            for column, col_stats in stats.items()
        },
    )
    print("full_balance_fraud", full_balance_fraud)
    print("fraud_transfer_cashout", fraud_transfer_cashout, "fraud_other", fraud_other)
    print("unique_orig", len(unique_orig), "unique_dest", len(unique_dest))
    print("orig_prefix", dict(orig_prefix), "dest_prefix", dict(dest_prefix))


def profile_cluster(path="data_cluster.csv"):
    numeric_columns = [
        "Year_Birth",
        "Income",
        "Kidhome",
        "Teenhome",
        "Recency",
        "MntWines",
        "MntFruits",
        "MntMeatProducts",
        "MntFishProducts",
        "MntSweetProducts",
        "MntGoldProds",
        "NumDealsPurchases",
        "NumWebPurchases",
        "NumCatalogPurchases",
        "NumStorePurchases",
        "NumWebVisitsMonth",
        "AcceptedCmp3",
        "AcceptedCmp4",
        "AcceptedCmp5",
        "AcceptedCmp1",
        "AcceptedCmp2",
        "Complain",
        "Response",
    ]
    stats = {c: {"n": 0, "sum": 0.0, "min": None, "max": None, "values": []} for c in numeric_columns}
    missing = Counter()
    education = Counter()
    marital = Counter()
    rows = 0
    total_spend = []
    total_purchases = []
    campaign_accepts = []
    age_values = []

    spend_columns = [
        "MntWines",
        "MntFruits",
        "MntMeatProducts",
        "MntFishProducts",
        "MntSweetProducts",
        "MntGoldProds",
    ]
    purchase_columns = [
        "NumDealsPurchases",
        "NumWebPurchases",
        "NumCatalogPurchases",
        "NumStorePurchases",
    ]
    campaign_columns = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5", "Response"]

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        fields = reader.fieldnames or []
        for row in reader:
            rows += 1
            for key, value in row.items():
                if value is None or value == "":
                    missing[key] += 1
            education[row["Education"]] += 1
            marital[row["Marital_Status"]] += 1

            for column in numeric_columns:
                value = row[column]
                if value == "":
                    continue
                value = float(value)
                col_stats = stats[column]
                col_stats["n"] += 1
                col_stats["sum"] += value
                col_stats["values"].append(value)
                col_stats["min"] = value if col_stats["min"] is None or value < col_stats["min"] else col_stats["min"]
                col_stats["max"] = value if col_stats["max"] is None or value > col_stats["max"] else col_stats["max"]

            total_spend.append(sum(float(row[column]) for column in spend_columns))
            total_purchases.append(sum(float(row[column]) for column in purchase_columns))
            campaign_accepts.append(sum(float(row[column]) for column in campaign_columns))
            age_values.append(2026 - float(row["Year_Birth"]))

    def summarize(values):
        return round_dict(
            min=percentile(values, 0),
            p50=percentile(values, 50),
            mean=sum(values) / len(values) if values else None,
            p90=percentile(values, 90),
            p99=percentile(values, 99),
            max=percentile(values, 100),
        )

    print("CLUSTER_DATASET")
    print("rows", rows)
    print("cols", len(fields), fields)
    print("missing", dict(missing))
    print("education", dict(education))
    print("marital", dict(marital))
    print("income", summarize(stats["Income"]["values"]))
    print("age_2026", summarize(age_values))
    print("recency", summarize(stats["Recency"]["values"]))
    print("total_spend", summarize(total_spend))
    print("total_purchases", summarize(total_purchases))
    print("campaign_accepts", summarize(campaign_accepts))
    print(
        "selected_num_stats",
        {
            column: {
                "min": round(col_stats["min"], 2) if col_stats["min"] is not None else None,
                "mean": round(col_stats["sum"] / col_stats["n"], 2) if col_stats["n"] else None,
                "max": round(col_stats["max"], 2) if col_stats["max"] is not None else None,
            }
            for column, col_stats in stats.items()
            if column in ("Income", "Kidhome", "Teenhome", "Recency", "Response", "Complain")
        },
    )


if __name__ == "__main__":
    profile_fraud()
    print()
    profile_cluster()
