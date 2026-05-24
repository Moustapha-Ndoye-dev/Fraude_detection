from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ml_project.features.customers import (
    BUSINESS_SEGMENT_LABELS,
    build_segment_lookup,
    customer_segment_profile,
    label_customer_segments,
    segment_by_label,
)


def test_label_customer_segments_assigns_all_business_labels():
    profile = pd.DataFrame(
        [
            {"segment": 0, "clients": 100, "depense_moyenne": 100, "recence_moyenne": 10, "achats_web_moyens": 1, "achats_promo_moyens": 1},
            {"segment": 1, "clients": 50, "depense_moyenne": 500, "recence_moyenne": 20, "achats_web_moyens": 2, "achats_promo_moyens": 2},
            {"segment": 2, "clients": 80, "depense_moyenne": 200, "recence_moyenne": 90, "achats_web_moyens": 1, "achats_promo_moyens": 1},
            {"segment": 3, "clients": 60, "depense_moyenne": 150, "recence_moyenne": 15, "achats_web_moyens": 8, "achats_promo_moyens": 7},
        ]
    )
    labeled = label_customer_segments(profile)
    assert set(labeled["label_metier"]) == set(BUSINESS_SEGMENT_LABELS)


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_label_customer_segments_handles_fewer_than_four_segments(n: int):
    profile = pd.DataFrame(
        [
            {"segment": 0, "clients": 100, "depense_moyenne": 100, "recence_moyenne": 10, "achats_web_moyens": 1, "achats_promo_moyens": 1},
            {"segment": 1, "clients": 50, "depense_moyenne": 500, "recence_moyenne": 20, "achats_web_moyens": 2, "achats_promo_moyens": 2},
            {"segment": 2, "clients": 80, "depense_moyenne": 200, "recence_moyenne": 90, "achats_web_moyens": 1, "achats_promo_moyens": 1},
            {"segment": 3, "clients": 60, "depense_moyenne": 150, "recence_moyenne": 15, "achats_web_moyens": 8, "achats_promo_moyens": 7},
        ]
    ).head(n)
    labeled = label_customer_segments(profile)
    assert len(labeled) == n
    assert labeled["label_metier"].notna().all()


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_build_segment_lookup_always_exposes_business_labels(n: int):
    profile = pd.DataFrame(
        [
            {"segment": 0, "clients": 100, "depense_moyenne": 100, "recence_moyenne": 10, "achats_web_moyens": 1, "achats_promo_moyens": 1},
            {"segment": 1, "clients": 50, "depense_moyenne": 500, "recence_moyenne": 20, "achats_web_moyens": 2, "achats_promo_moyens": 2},
            {"segment": 2, "clients": 80, "depense_moyenne": 200, "recence_moyenne": 90, "achats_web_moyens": 1, "achats_promo_moyens": 1},
            {"segment": 3, "clients": 60, "depense_moyenne": 150, "recence_moyenne": 15, "achats_web_moyens": 8, "achats_promo_moyens": 7},
        ]
    ).head(n)
    lookup = build_segment_lookup(profile)
    assert set(lookup) == set(BUSINESS_SEGMENT_LABELS)
    for label in BUSINESS_SEGMENT_LABELS:
        assert lookup[label]["clients"] >= 0


def test_segment_by_label_handles_empty_profile():
    row = segment_by_label(pd.DataFrame(), "Clients dormants")
    assert row["clients"] == 0
    assert row["label_metier"] == "Clients dormants"


def test_segment_by_label_falls_back_when_label_missing():
    profile = pd.DataFrame(
        [
            {"segment": 0, "clients": 100, "depense_moyenne": 500, "recence_moyenne": 10, "achats_web_moyens": 1, "achats_promo_moyens": 1, "label_metier": "Clients premium"},
        ]
    )
    dormant = segment_by_label(profile, "Clients dormants")
    assert dormant["segment"] == 0


def test_customer_segment_profile_ignores_missing_segment_values():
    segments = pd.DataFrame(
        {
            "segment": [pd.NA, pd.NA],
            "ID": [1, 2],
            "Income": [1, 2],
            "Total_Spend": [1, 2],
            "Total_Purchases": [1, 1],
            "NumWebPurchases": [1, 1],
            "NumDealsPurchases": [1, 1],
            "Recency": [1, 1],
            "Response": [0, 0],
        }
    )
    assert customer_segment_profile(segments).empty


def test_build_segment_lookup_with_real_customer_segments_file():
    root = Path(__file__).resolve().parents[1]
    segments_path = root / "data" / "processed" / "customer_segments.csv"
    if not segments_path.exists():
        pytest.skip("customer_segments.csv absent")

    segments = pd.read_csv(segments_path)
    profile = label_customer_segments(customer_segment_profile(segments))
    lookup = build_segment_lookup(profile, already_labeled=True)

    assert set(lookup) == set(BUSINESS_SEGMENT_LABELS)
    for label in BUSINESS_SEGMENT_LABELS:
        row = lookup[label]
        assert row["clients"] > 0
        assert row["depense_moyenne"] >= 0


def test_build_segment_lookup_without_explicit_dormant_label():
    profile = pd.DataFrame(
        [
            {
                "segment": 0,
                "clients": 100,
                "depense_moyenne": 500,
                "recence_moyenne": 80,
                "achats_web_moyens": 1,
                "achats_promo_moyens": 1,
                "reponse_campagne": 0.2,
                "label_metier": "Clients premium",
            },
            {
                "segment": 1,
                "clients": 200,
                "depense_moyenne": 100,
                "recence_moyenne": 10,
                "achats_web_moyens": 5,
                "achats_promo_moyens": 4,
                "reponse_campagne": 0.1,
                "label_metier": "Digitaux et promotions",
            },
        ]
    )
    lookup = build_segment_lookup(profile, already_labeled=True)
    dormant = lookup["Clients dormants"]
    assert dormant["recence_moyenne"] == 80
    assert lookup["Clients economes"]["clients"] > 0
