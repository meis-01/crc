from __future__ import annotations

import re

CANONICAL_LEADS = (
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
)
PRECORDIAL_LEADS = ("V1", "V2", "V3", "V4", "V5", "V6")

_ALIASES = {
    "I": "I",
    "DI": "I",
    "LEADI": "I",
    "II": "II",
    "DII": "II",
    "LEADII": "II",
    "III": "III",
    "DIII": "III",
    "LEADIII": "III",
    "AVR": "aVR",
    "AVL": "aVL",
    "AVF": "aVF",
    **{f"V{i}": f"V{i}" for i in range(1, 7)},
}


def normalize_lead_name(name: str) -> str:
    """Normalize common ECG lead spellings without guessing unknown names."""
    key = re.sub(r"[^A-Za-z0-9]", "", str(name)).upper()
    if key not in _ALIASES:
        raise ValueError(f"Unknown ECG lead name: {name!r}")
    return _ALIASES[key]


def normalize_target_lead(name: str) -> str:
    lead = normalize_lead_name(name)
    if lead not in CANONICAL_LEADS:
        raise ValueError(f"Target lead {name!r} is not a standard 12-lead ECG lead")
    return lead


LIMB_LEAD_RELATIONSHIPS = {
    "Einthoven": "II = I + III",
    "III": "III = II - I",
    "aVR": "aVR = -(I + II) / 2",
    "aVL": "aVL = I - II / 2",
    "aVF": "aVF = II - I / 2",
}

