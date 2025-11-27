import pandas as pd

# --------------------------------------
# 1. Load RAW packets
# --------------------------------------
raw = pd.read_parquet("data/raw/combined_dataset.parquet")
raw = raw.rename(columns={
    "Time": "time",
    "Source": "source",
    "Destination": "destination",
    "Protocol": "protocol",
    "Length": "length",
    "Info": "info"
})
raw["time"] = pd.to_datetime(raw["time"])
raw = raw.sort_values("time").reset_index(drop=True)

# --------------------------------------
# 2. Load FEATURES
# --------------------------------------
features = pd.read_parquet("data/processed/features_clean.parquet")
features = features.reset_index(drop=True)
features["window_id"] = features.index

# --------------------------------------
# 3. Load LABELS
# --------------------------------------
labels = pd.read_parquet("data/processed/labeled_features.parquet")
labels = labels.reset_index(drop=True)
labels["window_id"] = labels.index

# Remove duplicate engineered columns
duplicate_cols = [
    "packet_count", "byte_count", "avg_packet_len",
    "std_packet_len", "iat_mean", "iat_std",
    "tls_client_hello", "tls_server_hello"
]
labels = labels.drop(columns=[c for c in duplicate_cols if c in labels])

# Combine features + labels
combined = pd.merge(features, labels, on="window_id", how="inner")


# --------------------------------------
# 4. Assign window_id to RAW packets
# --------------------------------------
total_time = (raw["time"].max() - raw["time"].min()).total_seconds()
W = len(features)
window_seconds = total_time / W

raw["time_offset"] = (raw["time"] - raw["time"].min()).dt.total_seconds()
raw["window_id"] = (raw["time_offset"] // window_seconds).astype(int)
raw["window_id"] = raw["window_id"].clip(0, W - 1)

# --------------------------------------
# 5. For each window, pick one source/destination pair
# --------------------------------------
meta = raw.groupby("window_id").agg({
    "source": "first",
    "destination": "first",
    "protocol": "first",
    "domain": "first"
}).reset_index()

# --------------------------------------
# 6. Merge with combined dataset
# --------------------------------------
merged = pd.merge(combined, meta, on="window_id", how="left")

merged.to_parquet("data/processed/labeled_with_metadata.parquet", index=False)

print("✅ Final dataset saved: data/processed/labeled_with_metadata.parquet")
print("Shape:", merged.shape)
print("Columns:", merged.columns.tolist())
