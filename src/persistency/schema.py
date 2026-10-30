"""Categorical input contract restricted to baseline-named measurements."""

import pandas as pd

# Options are the source dictionary, not target-dependent feature selection.
FEATURES = {
    "Gluco_Record_Prior_Ntm": ("Glucocorticoid record before treatment", ["N", "Y"]),
    "Frag_Frac_Prior_Ntm": ("Fragility fracture before treatment", ["N", "Y"]),
    "Risk_Segment_Prior_Ntm": ("Risk segment before treatment", ["VLR_LR", "HR_VHR"]),
    "Tscore_Bucket_Prior_Ntm": ("T-score bucket before treatment", [">-2.5", "<=-2.5"]),
    "Age_Bucket": ("Source age bucket", ["<55", "55-65", "65-75", ">75"]),
    "Ntm_Speciality": (
        "Treating specialty",
        [
            "GENERAL PRACTITIONER",
            "RHEUMATOLOGY",
            "ENDOCRINOLOGY",
            "ONCOLOGY",
            "Unknown",
            "OTHER",
        ],
    ),
}
LABELS = {
    "Non-Persistent": "Non-persistent source label",
    "Persistent": "Persistent source label",
}


def validate_records(records):
    """Reject malformed categories; blank values are explicit missing observations."""
    if not isinstance(records, list) or not 1 <= len(records) <= 500:
        raise ValueError("Provide a list of 1 to 500 records.")
    cleaned = []
    for number, record in enumerate(records, 1):
        if not isinstance(record, dict):
            raise ValueError(f"Row {number}: each record must be an object.")
        missing = set(FEATURES) - set(record)
        if missing:
            raise ValueError(
                f"Row {number}: missing columns: {', '.join(sorted(missing))}"
            )
        row = {}
        for field, (_, options) in FEATURES.items():
            value = record[field]
            if value is None or value == "":
                row[field] = "MISSING"
            elif not isinstance(value, str) or value not in options:
                raise ValueError(f"Row {number}: invalid category for {field}.")
            else:
                row[field] = value
        if all(value == "MISSING" for value in row.values()):
            raise ValueError(f"Row {number}: provide at least one observed value.")
        cleaned.append(row)
    return pd.DataFrame(cleaned, columns=list(FEATURES))
