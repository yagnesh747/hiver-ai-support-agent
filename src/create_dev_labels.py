import re
import pandas as pd

INPUT_PATH = "data/processed/apple_support_dev_20000.csv"
OUTPUT_PATH = "data/processed/apple_support_dev_20000_labeled.csv"


def has_any(text, keywords):
    return any(re.search(r"\b" + re.escape(word) + r"\b", text) for word in keywords)


def assign_intent(text):
    text = str(text).lower()

    # 1. Apple ID / Account
    if has_any(text, [
        "apple id", "appleid", "password", "login",
        "logged in", "sign in", "account", "verification code"
    ]):
        return "APPLE_ID_ACCOUNT"

    # 2. iCloud / Backup
    if has_any(text, [
        "icloud", "backup", "back up", "backing up",
        "restore from backup", "sync", "synchronizing"
    ]):
        return "ICLOUD_BACKUP"

    # 3. Battery
    if has_any(text, [
        "battery", "battery life", "battery health",
        "draining", "drain", "dies quickly", "battery dies"
    ]):
        return "BATTERY"

    # 4. Connectivity
    if has_any(text, [
        "wifi", "wi-fi", "bluetooth", "cellular",
        "network", "signal", "no service", "internet",
        "hotspot", "airdrop"
    ]):
        return "CONNECTIVITY"

    # 5. Audio / Media
    if has_any(text, [
        "speaker", "speakers", "headphone", "headphones",
        "audio", "sound", "music", "podcast", "podcasts",
        "volume", "microphone", "video", "itunes"
    ]):
        return "AUDIO_MEDIA"

    # 6. Hardware / Charging
    if has_any(text, [
        "charging", "charger", "charge",
        "charging cable", "screen", "touchscreen",
        "camera", "home button", "touch id",
        "broken", "cracked", "physical damage"
    ]):
        return "HARDWARE_CHARGING"

    # 7. Apps
    if has_any(text, [
        "app store", "application", "netflix",
        "instagram", "facebook", "twitter",
        "whatsapp", "safari", "calendar",
        "mail app"
    ]):
        return "APPS"

    # 8. iOS / Software
    if has_any(text, [
        "ios", "ios 11", "ios 12", "ios 13",
        "ios 14", "ios 15", "ios 16", "ios 17",
        "ios 18", "update", "updated", "upgrade",
        "software", "macos", "operating system"
    ]):
        return "IOS_SOFTWARE"

    # 9. Device Performance
    if has_any(text, [
        "slow", "slower", "lag", "lagging",
        "freezing", "freeze", "crash", "crashing",
        "restarting", "restart", "performance",
        "overheating", "overheats"
    ]):
        return "DEVICE_PERFORMANCE"

    # 10. Other
    return "OTHER"


def assign_action(intent, text):
    text = str(text).lower()

    # Account-specific financial/purchase issues
    if has_any(text, [
        "refund", "charged", "double charged",
        "double billed", "billing", "payment",
        "purchase dispute", "fraud"
    ]):
        return "ESCALATE"

    if intent == "OTHER":
        return "ANSWER"

    return "TROUBLESHOOT"


def main():
    print("=" * 60)
    print("Creating Development Training Labels")
    print("=" * 60)

    print("\nLoading development dataset...")

    df = pd.read_csv(INPUT_PATH)

    print(f"Rows loaded: {len(df):,}")

    # Assign intent
    df["intent"] = df["text"].apply(assign_intent)

    # Assign expected action
    df["expected_action"] = df.apply(
        lambda row: assign_action(row["intent"], row["text"]),
        axis=1
    )

    # Save
    df.to_csv(OUTPUT_PATH, index=False)

    print("\nDevelopment labels created.")

    print("\nIntent distribution:")
    print(df["intent"].value_counts())

    print("\nAction distribution:")
    print(df["expected_action"].value_counts())

    print(f"\nSaved to:")
    print(OUTPUT_PATH)

    print("\nComplete.")


if __name__ == "__main__":
    main()