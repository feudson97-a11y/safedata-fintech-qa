"""
Reproducible, fully synthetic Fintech transactions for exercising the pipeline.

* Realism: repeat customers, log-normal spend per merchant category, Luhn-valid
  card numbers across several schemes, heterogeneous phone/card formatting.
* Hidden PII: ``Transaction_Notes`` embeds emails, phones, cards and names mid-sentence.
* Known-bad data: a controlled share of rows receive one injected defect, recorded
  in a ground-truth manifest so detection recall can be measured objectively.

No real person, card or phone number is represented.
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np
import pandas as pd

from safedata.domain.schema import DATE_FORMAT, LINEAGE_COLUMN

# --------------------------------------------------------------------------- #
# Reference data                                                              #
# --------------------------------------------------------------------------- #
FIRST_NAMES: tuple[str, ...] = (
    "James",
    "Olivia",
    "Liam",
    "Emma",
    "Noah",
    "Ava",
    "Ethan",
    "Sophia",
    "Mason",
    "Isabella",
    "Lucas",
    "Mia",
    "Logan",
    "Amelia",
    "Elijah",
    "Harper",
    "Aiden",
    "Evelyn",
    "Carter",
    "Abigail",
    "Daniel",
    "Emily",
    "Matthew",
    "Ella",
    "Henry",
    "Scarlett",
    "Owen",
    "Chloe",
    "Sebastian",
    "Zoe",
    "Priya",
    "Arjun",
    "Wei",
    "Mei",
    "Hiroshi",
    "Yuki",
    "Carlos",
    "Sofia",
    "Mateo",
    "Valentina",
    "Omar",
    "Fatima",
    "Kwame",
    "Amara",
    "Dmitri",
    "Anya",
)
LAST_NAMES: tuple[str, ...] = (
    "Smith",
    "Johnson",
    "Williams",
    "Brown",
    "Garcia",
    "Miller",
    "Davis",
    "Rodriguez",
    "Martinez",
    "Hernandez",
    "Lopez",
    "Gonzalez",
    "Wilson",
    "Anderson",
    "Thomas",
    "Taylor",
    "Moore",
    "Jackson",
    "Martin",
    "Lee",
    "Thompson",
    "Harris",
    "Clark",
    "Lewis",
    "Robinson",
    "Walker",
    "Allen",
    "Wright",
    "Patel",
    "Nguyen",
    "Kim",
    "Chen",
    "Okafor",
    "Rossi",
    "Schmidt",
    "O'Brien",
    "Kowalski",
    "Haddad",
    "Mensah",
    "Tanaka",
    "Silva",
    "Novak",
)
EMAIL_DOMAINS: tuple[str, ...] = (
    "gmail.com",
    "yahoo.com",
    "outlook.com",
    "icloud.com",
    "protonmail.com",
    "fintechmail.io",
)

# category -> (log-normal mu, log-normal sigma, sampling weight)
MERCHANT_PROFILES: dict[str, tuple[float, float, float]] = {
    "Groceries": (3.9, 0.50, 0.20),
    "Restaurants": (3.4, 0.60, 0.16),
    "Fuel": (3.8, 0.30, 0.10),
    "E-Commerce": (4.1, 0.80, 0.15),
    "Utilities": (4.6, 0.40, 0.07),
    "Healthcare": (4.8, 0.70, 0.06),
    "Entertainment": (3.5, 0.60, 0.08),
    "Electronics": (5.3, 0.80, 0.05),
    "Travel": (5.8, 0.70, 0.05),
    "Peer-to-Peer Transfer": (4.5, 1.00, 0.08),
}

# (IIN prefix, PAN length, sampling weight)
CARD_SCHEMES: tuple[tuple[str, int, float], ...] = (
    ("4", 16, 0.50),  # Visa
    ("51", 16, 0.06),
    ("52", 16, 0.06),
    ("53", 16, 0.06),
    ("54", 16, 0.06),
    ("55", 16, 0.06),  # Mastercard
    ("34", 15, 0.07),
    ("37", 15, 0.07),  # Amex
    ("6011", 16, 0.06),  # Discover
)

DATE_START = pd.Timestamp("2025-01-01 00:00:00")
DATE_END = pd.Timestamp("2026-09-30 23:59:59")

PII_NOTE_TEMPLATES: tuple[str, ...] = (
    "Customer called from {email} asking about a charge.",
    "{name} disputed this transaction by phone at {phone}.",
    "Card {card} reported lost; replacement card shipped to {first}.",
    "Chargeback paperwork emailed to {email} for review.",
    "{honorific} {last} requested a callback on {phone} regarding a duplicate charge.",
    "Verified identity of {name} before releasing the hold.",
    "Account holder {name} ({email}) flagged for enhanced KYC review.",
    "Fraud team reached {first} at {phone}; customer confirmed card {card} is in their possession.",
    "Joint account holder {other_name} added as authorized user.",
    "Customer asked to update contact number to {phone} and email to {email}.",
)
CLEAN_NOTE_TEMPLATES: tuple[str, ...] = (
    "Recurring subscription payment.",
    "Standard point-of-sale purchase.",
    "Merchant settlement batch processed.",
    "No customer contact required.",
    "Contactless payment approved.",
    "Card-not-present transaction; 3DS authentication passed.",
    "",
)
HONORIFICS: tuple[str, ...] = ("Mr.", "Ms.", "Mrs.", "Dr.")


class DefectType(StrEnum):
    AMOUNT_NON_NUMERIC = "amount_non_numeric"
    AMOUNT_NEGATIVE = "amount_negative"
    AMOUNT_OUTLIER = "amount_outlier"
    DATE_UNPARSEABLE = "date_unparseable"
    DATE_FUTURE = "date_future"
    MERCHANT_CATEGORY_INVALID = "merchant_category_invalid"
    EMAIL_MALFORMED = "email_malformed"
    CARD_LUHN_FAILURE = "card_luhn_failure"
    TRANSACTION_ID_MISSING = "transaction_id_missing"
    TRANSACTION_ID_DUPLICATE = "transaction_id_duplicate"


DEFECT_TYPES: tuple[DefectType, ...] = tuple(DefectType)


# --------------------------------------------------------------------------- #
# Primitive generators                                                        #
# --------------------------------------------------------------------------- #
def luhn_check_digit(partial_pan: str) -> str:
    """Return the Luhn check digit that makes ``partial_pan + digit`` valid."""
    total = 0
    # Walk right-to-left; the digit adjacent to the (future) check digit is doubled.
    for i, char in enumerate(reversed(partial_pan)):
        digit = int(char)
        if i % 2 == 0:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return str((10 - total % 10) % 10)


def generate_card_number(rng: np.random.Generator) -> str:
    """Generate a random, Luhn-valid PAN for a weighted card scheme."""
    weights = np.array([w for _, _, w in CARD_SCHEMES])
    prefix, length, _ = CARD_SCHEMES[rng.choice(len(CARD_SCHEMES), p=weights / weights.sum())]
    body_len = length - len(prefix) - 1
    body = "".join(rng.integers(0, 10, size=body_len).astype(str))
    partial = prefix + body
    return partial + luhn_check_digit(partial)


def format_card_number(pan: str, rng: np.random.Generator) -> str:
    """Render a PAN in one of the layouts commonly seen in source systems."""
    style = rng.integers(0, 3)
    if style == 0:
        return pan
    groups = [pan[0:4], pan[4:10], pan[10:]] if len(pan) == 15 else [pan[i : i + 4] for i in range(0, 16, 4)]
    return ("-" if style == 1 else " ").join(groups)


def generate_phone_number(rng: np.random.Generator) -> str:
    """Generate a NANP-shaped phone number in a randomly chosen format."""
    area = int(rng.integers(201, 990))
    exchange = int(rng.integers(200, 1000))
    line = int(rng.integers(0, 10_000))
    fmt = rng.integers(0, 5)
    if fmt == 0:
        return f"({area}) {exchange}-{line:04d}"
    if fmt == 1:
        return f"{area}-{exchange}-{line:04d}"
    if fmt == 2:
        return f"{area}.{exchange}.{line:04d}"
    if fmt == 3:
        return f"+1 {area} {exchange} {line:04d}"
    return f"{area}{exchange}{line:04d}"


def generate_email(first: str, last: str, rng: np.random.Generator) -> str:
    """Derive a plausible email address from a customer's name."""
    first_lc, last_lc = first.lower(), last.lower().replace("'", "")
    nn = int(rng.integers(1, 99))
    patterns = (
        f"{first_lc}.{last_lc}",
        f"{first_lc[0]}{last_lc}",
        f"{first_lc}{last_lc}{nn}",
        f"{last_lc}.{first_lc}{nn}",
        f"{first_lc}_{last_lc}",
    )
    local = patterns[rng.integers(0, 5)]
    return f"{local}@{EMAIL_DOMAINS[rng.integers(0, len(EMAIL_DOMAINS))]}"


# --------------------------------------------------------------------------- #
# Dataset assembly                                                            #
# --------------------------------------------------------------------------- #
def build_customer_pool(n_customers: int, rng: np.random.Generator) -> pd.DataFrame:
    """Create a pool of synthetic customers; transactions sample from it."""
    records = []
    for _ in range(n_customers):
        first = FIRST_NAMES[rng.integers(0, len(FIRST_NAMES))]
        last = LAST_NAMES[rng.integers(0, len(LAST_NAMES))]
        pan = generate_card_number(rng)
        records.append(
            {
                "first": first,
                "last": last,
                "Customer_Name": f"{first} {last}",
                "Email_Address": generate_email(first, last, rng),
                "Phone_Number": generate_phone_number(rng),
                "Credit_Card_Number": format_card_number(pan, rng),
            }
        )
    return pd.DataFrame.from_records(records)


def build_note(row: pd.Series, other_name: str, rng: np.random.Generator, pii_note_rate: float) -> str:
    """Compose a support-agent note; a share of notes embed raw PII mid-sentence."""
    if rng.random() >= pii_note_rate:
        return CLEAN_NOTE_TEMPLATES[rng.integers(0, len(CLEAN_NOTE_TEMPLATES))]
    template = PII_NOTE_TEMPLATES[rng.integers(0, len(PII_NOTE_TEMPLATES))]
    return template.format(
        name=row["Customer_Name"],
        first=row["first"],
        last=row["last"],
        honorific=HONORIFICS[rng.integers(0, len(HONORIFICS))],
        email=row["Email_Address"],
        phone=row["Phone_Number"],
        card=row["Credit_Card_Number"],
        other_name=other_name,
    )


def inject_defects(df: pd.DataFrame, rng: np.random.Generator, defect_rate: float) -> pd.DataFrame:
    """Mutate a random subset of rows with one defect each; return a manifest."""
    n_defects = round(len(df) * defect_rate)
    if n_defects == 0:
        return pd.DataFrame(columns=[LINEAGE_COLUMN, "Defect_Type"])

    rows = rng.choice(len(df), size=n_defects, replace=False)
    clean_rows = np.setdiff1d(np.arange(len(df)), rows)
    manifest: list[dict[str, object]] = []

    for i, row_idx in enumerate(rows):
        defect = DEFECT_TYPES[i % len(DEFECT_TYPES)]
        category = df.at[row_idx, "Merchant_Category"]

        match defect:
            case DefectType.AMOUNT_NON_NUMERIC:
                df.at[row_idx, "Transaction_Amount"] = ("N/A", "", "12.3.4", "USD 45.00")[rng.integers(0, 4)]
            case DefectType.AMOUNT_NEGATIVE:
                df.at[row_idx, "Transaction_Amount"] = "-" + df.at[row_idx, "Transaction_Amount"]
            case DefectType.AMOUNT_OUTLIER:
                mu, sigma, _ = MERCHANT_PROFILES[category]
                df.at[row_idx, "Transaction_Amount"] = f"{np.exp(mu + 6 * sigma) * rng.uniform(1.0, 3.0):.2f}"
            case DefectType.DATE_UNPARSEABLE:
                bad_dates = ("2025-13-45 10:00:00", "not a date", "", "31/12/2025")
                df.at[row_idx, "Transaction_Date"] = bad_dates[rng.integers(0, 4)]
            case DefectType.DATE_FUTURE:
                offset = pd.Timedelta(seconds=int(rng.integers(0, 365 * 86_400)))
                df.at[row_idx, "Transaction_Date"] = (pd.Timestamp("2027-06-01") + offset).strftime(DATE_FORMAT)
            case DefectType.MERCHANT_CATEGORY_INVALID:
                df.at[row_idx, "Merchant_Category"] = ("", "UNKNOWN_MCC", "groceries ")[rng.integers(0, 3)]
            case DefectType.EMAIL_MALFORMED:
                df.at[row_idx, "Email_Address"] = df.at[row_idx, "Email_Address"].replace("@", ".")
            case DefectType.CARD_LUHN_FAILURE:
                card = df.at[row_idx, "Credit_Card_Number"]
                # Changing only the check digit is guaranteed to break the Luhn checksum.
                df.at[row_idx, "Credit_Card_Number"] = card[:-1] + str((int(card[-1]) + 1) % 10)
            case DefectType.TRANSACTION_ID_MISSING:
                df.at[row_idx, "Transaction_ID"] = ""
            case DefectType.TRANSACTION_ID_DUPLICATE:
                df.at[row_idx, "Transaction_ID"] = df.at[int(rng.choice(clean_rows)), "Transaction_ID"]

        manifest.append({LINEAGE_COLUMN: int(row_idx), "Defect_Type": str(defect)})

    return pd.DataFrame(manifest).sort_values(LINEAGE_COLUMN).reset_index(drop=True)


def generate_dataset(
    n_rows: int = 10_000,
    seed: int = 42,
    defect_rate: float = 0.02,
    pii_note_rate: float = 0.45,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the synthetic transaction table.

    Returns:
        (transactions, defect_manifest). Every column of ``transactions`` is a
        string, exactly as it would arrive from a CSV extract.
    """
    if n_rows <= 0:
        raise ValueError("n_rows must be positive")
    if not 0.0 <= defect_rate < 0.5:
        raise ValueError("defect_rate must be in [0, 0.5)")

    rng = np.random.default_rng(seed)
    customers = build_customer_pool(max(50, n_rows // 5), rng)
    picks = customers.iloc[rng.integers(0, len(customers), size=n_rows)].reset_index(drop=True)

    # Merchant category and spend drawn from category-specific log-normals.
    categories = list(MERCHANT_PROFILES)
    weights = np.array([MERCHANT_PROFILES[c][2] for c in categories])
    cat_idx = rng.choice(len(categories), size=n_rows, p=weights / weights.sum())
    mus = np.array([MERCHANT_PROFILES[c][0] for c in categories])[cat_idx]
    sigmas = np.array([MERCHANT_PROFILES[c][1] for c in categories])[cat_idx]
    amounts = np.maximum(np.round(rng.lognormal(mus, sigmas), 2), 0.50)

    # Chronologically ordered timestamps -> sequential transaction IDs.
    span = int((DATE_END - DATE_START).total_seconds())
    timestamps = np.sort(DATE_START.value // 10**9 + rng.integers(0, span, size=n_rows))
    dates = pd.to_datetime(timestamps, unit="s").strftime(DATE_FORMAT)

    other_names = customers["Customer_Name"].to_numpy()
    notes = [
        build_note(row, other_names[rng.integers(0, len(other_names))], rng, pii_note_rate)
        for _, row in picks.iterrows()
    ]

    df = pd.DataFrame(
        {
            "Transaction_ID": [f"TXN-{i:08d}" for i in range(1, n_rows + 1)],
            "Customer_Name": picks["Customer_Name"],
            "Email_Address": picks["Email_Address"],
            "Phone_Number": picks["Phone_Number"],
            "Credit_Card_Number": picks["Credit_Card_Number"],
            "Transaction_Amount": [f"{a:.2f}" for a in amounts],
            "Transaction_Date": list(dates),
            "Merchant_Category": [categories[i] for i in cat_idx],
            "Transaction_Notes": notes,
        },
        dtype=object,
    )
    manifest = inject_defects(df, rng, defect_rate)
    return df, manifest
