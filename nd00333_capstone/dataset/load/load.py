import glob
import pandas as pd


DTYPE = {
    "Flow Duration": "int64",
    "TotLen Fwd Pkts": "int64",
    "TotLen Bwd Pkts": "int64",
    "Fwd Pkt Len Std": "int64",
    "Bwd Pkt Len Max": "int64",
    "Bwd Pkt Len Std": "int64",
    "Flow Byts/s": "int64",
    "Flow Pkts/s": "int64",
    "Flow IAT Max": "int64",
    "Bwd IAT Min": "int64",
    "Bwd Header Len": "int64",
    "Pkt Len Max": "int64",
    "Pkt Len Std": "int64",
    "RST Flag Cnt": "int64",
    "PSH Flag Cnt": "int64",
    "ECE Flag Cnt": "int64",
    "Init Fwd Win Byts": "int64",
    "Init Bwd Win Byts": "int64",
    "Fwd Seg Size Min": "int64",
    "Label": "object",
}

LABEL_TO_INDEX = {
    "Benign": 0,
    "Bot": 1,
    "Brute Force -Web": 2,
    "Brute Force -XSS": 3,
    "DDOS attack-HOIC": 4,
    "DDoS attacks-LOIC-HTTP": 5,
    "DoS attacks-GoldenEye": 6,
    "DoS attacks-Hulk": 7,
    "DoS attacks-SlowHTTPTest": 8,
    "DoS attacks-Slowloris": 9,
    "FTP-BruteForce": 10,
    "Infilteration": 11,
    "SQL Injection": 12,
    "SSH-Bruteforce": 13,
}


def get_df_from_csv(csv, usecols=None, dtype=None):
    return pd.read_csv(csv, usecols=usecols, dtype=dtype)


def get_df_from_directory(directory, usecols=None, dtype=None):
    return pd.concat(
        get_df_from_csv(csv, usecols=usecols, dtype=dtype)
        for csv in sorted(glob.glob(f"{directory}/*.csv"))
    ).reset_index(drop=True)
