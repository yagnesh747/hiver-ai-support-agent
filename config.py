"""
Central path constants for the AppleSupport AI Support Agent.

Note: individual src/ modules currently define their own path constants
at the top of each file (kept intentionally explicit and self-contained
so each script is easy to read and run in isolation). This file exists
as a single reference point for the project's directory layout.
"""

RAW_DATA_PATH = "data/raw/twcs.csv"
PROCESSED_DIR = "data/processed"
GOLDEN_PATH = "data/golden/apple_support_golden_200_annotation.csv"
DEV_LABELED_PATH = "data/processed/apple_support_dev_20000_labeled.csv"
CLEAN_DATA_PATH = "data/processed/apple_support_clean.csv"
OUTPUTS_DIR = "outputs"

BRAND = "AppleSupport"

INTENTS = [
    "IOS_SOFTWARE",
    "BATTERY",
    "DEVICE_PERFORMANCE",
    "APPS",
    "APPLE_ID_ACCOUNT",
    "ICLOUD_BACKUP",
    "CONNECTIVITY",
    "AUDIO_MEDIA",
    "HARDWARE_CHARGING",
    "OTHER",
]

ACTIONS = ["ANSWER", "TROUBLESHOOT", "ESCALATE"]
