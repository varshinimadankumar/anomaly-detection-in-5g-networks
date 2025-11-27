import pandas as pd
import numpy as np

FEATURES = [
    "packet_count", 
    "byte_count",
    "avg_packet_len", 
    "std_packet_len",
    "iat_mean", 
    "iat_std",
    "tls_client_hello", 
    "tls_server_hello"
]

def build_feature_matrix(df):
    X = df.copy()

    # Fix std for single packet windows
    X["std_packet_len"] = X["std_packet_len"].fillna(0)

    # Replace all other NaN or infinities with 0
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    return X[FEATURES]
