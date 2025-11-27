import pandas as pd
import numpy as np

def aggregate_window(win: pd.DataFrame):
    res = {
        "packet_count": len(win),
        "byte_count": win["length"].sum(),
        "avg_packet_len": win["length"].mean(),
        "std_packet_len": win["length"].std(),
        "iat_mean": 0,
        "iat_std": 0
    }

    info = win["info"].astype(str)
    res["tls_client_hello"] = info.str.contains("Client Hello").sum()
    res["tls_server_hello"] = info.str.contains("Server Hello").sum()

    # IAT
    ts = win.index.astype("int64") // 1_000_000
    if len(ts) > 1:
        iat = np.diff(ts)
        res["iat_mean"] = iat.mean()
        res["iat_std"] = iat.std()

    return res

def generate_windows(df, window=5, step=2):
    df = df.sort_values("timestamp")
    df = df.set_index("timestamp")

    start = df.index.min()
    end = df.index.max()

    rows = []
    t = start
    while t + pd.Timedelta(seconds=window) <= end:
        win = df[t : t + pd.Timedelta(seconds=window)]
        if len(win) > 0:
            row = aggregate_window(win)
            row["window_start"] = t
            row["window_end"] = t + pd.Timedelta(seconds=window)
            rows.append(row)
        t += pd.Timedelta(seconds=step)

    return pd.DataFrame(rows)
