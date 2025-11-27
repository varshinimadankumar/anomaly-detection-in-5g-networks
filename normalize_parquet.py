import pandas as pd

def normalize_packet_parquet():
    df = pd.read_parquet("data/raw/combined_dataset.parquet")

    df = df.rename(columns={
        "Time": "timestamp",
        "Source": "src",
        "Destination": "dst",
        "Protocol": "protocol",
        "Length": "length",
        "Info": "info",
        "Domain": "domain"
    })

    # convert timestamp
    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    except:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

    df["length"] = pd.to_numeric(df["length"], errors="coerce").fillna(0)

    df.to_parquet("data/processed/normalized.parquet", index=False)
    print("Saved: data/processed/normalized.parquet")

if __name__ == "__main__":
    normalize_packet_parquet()
