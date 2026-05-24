from __future__ import annotations

import pandas as pd

from ml_project.features.customers import customer_segment_profile, label_customer_segments


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
    assert set(labeled["label_metier"]) == {
        "Clients premium",
        "Clients dormants",
        "Digitaux et promotions",
        "Clients economes",
    }
