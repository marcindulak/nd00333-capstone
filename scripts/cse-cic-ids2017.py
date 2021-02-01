"""
Preprocess the https://www.unb.ca/cic/datasets/ids-2017.html dataset
"""

from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

from nd00333.dataset.clean import clean
from nd00333.dataset.load import load

FEATURES_2017_TO_2018 = {
    "Flow ID": "Flow ID",
    "Source IP": "Src IP",
    "Source Port": "Src Port",
    "Destination IP": "Dst IP",
    "Destination Port": "Dst Port",
    "Protocol": "Protocol",
    "Timestamp": "Timestamp",
    "Flow Duration": "Flow Duration",
    "Total Fwd Packets": "Tot Fwd Pkts",
    "Total Backward Packets": "Tot Bwd Pkts",
    "Total Length of Fwd Packets": "TotLen Fwd Pkts",
    "Total Length of Bwd Packets": "TotLen Bwd Pkts",
    "Fwd Packet Length Max": "Fwd Pkt Len Max",
    "Fwd Packet Length Min": "Fwd Pkt Len Min",
    "Fwd Packet Length Mean": "Fwd Pkt Len Mean",
    "Fwd Packet Length Std": "Fwd Pkt Len Std",
    "Bwd Packet Length Max": "Bwd Pkt Len Max",
    "Bwd Packet Length Min": "Bwd Pkt Len Min",
    "Bwd Packet Length Mean": "Bwd Pkt Len Mean",
    "Bwd Packet Length Std": "Bwd Pkt Len Std",
    "Flow Bytes/s": "Flow Byts/s",
    "Flow Packets/s": "Flow Pkts/s",
    "Flow IAT Mean": "Flow IAT Mean",
    "Flow IAT Std": "Flow IAT Std",
    "Flow IAT Max": "Flow IAT Max",
    "Flow IAT Min": "Flow IAT Min",
    "Fwd IAT Total": "Fwd IAT Tot",
    "Fwd IAT Mean": "Fwd IAT Mean",
    "Fwd IAT Std": "Fwd IAT Std",
    "Fwd IAT Max": "Fwd IAT Max",
    "Fwd IAT Min": "Fwd IAT Min",
    "Bwd IAT Total": "Bwd IAT Tot",
    "Bwd IAT Mean": "Bwd IAT Mean",
    "Bwd IAT Std": "Bwd IAT Std",
    "Bwd IAT Max": "Bwd IAT Max",
    "Bwd IAT Min": "Bwd IAT Min",
    "Fwd PSH Flags": "Fwd PSH Flags",
    "Bwd PSH Flags": "Bwd PSH Flags",
    "Fwd URG Flags": "Fwd URG Flags",
    "Bwd URG Flags": "Bwd URG Flags",
    "Fwd Header Length": "Fwd Header Len",
    "Bwd Header Length": "Bwd Header Len",
    "Fwd Packets/s": "Fwd Pkts/s",
    "Bwd Packets/s": "Bwd Pkts/s",
    "Min Packet Length": "Pkt Len Min",
    "Max Packet Length": "Pkt Len Max",
    "Packet Length Mean": "Pkt Len Mean",
    "Packet Length Std": "Pkt Len Std",
    "Packet Length Variance": "Pkt Len Var",
    "FIN Flag Count": "FIN Flag Cnt",
    "SYN Flag Count": "SYN Flag Cnt",
    "RST Flag Count": "RST Flag Cnt",
    "PSH Flag Count": "PSH Flag Cnt",
    "ACK Flag Count": "ACK Flag Cnt",
    "URG Flag Count": "URG Flag Cnt",
    "CWE Flag Count": "CWE Flag Cnt",
    "ECE Flag Count": "ECE Flag Cnt",
    "Down/Up Ratio": "Down/Up Ratio",
    "Average Packet Size": "Pkt Size Avg",
    "Avg Fwd Segment Size": "Fwd Seg Size Avg",
    "Avg Bwd Segment Size": "Bwd Seg Size Avg",
    "Fwd Avg Bytes/Bulk": "Fwd Byts/b Avg",
    "Fwd Avg Packets/Bulk": "Fwd Pkts/b Avg",
    "Fwd Avg Bulk Rate": "Fwd Blk Rate Avg",
    "Bwd Avg Bytes/Bulk": "Bwd Byts/b Avg",
    "Bwd Avg Packets/Bulk": "Bwd Pkts/b Avg",
    "Bwd Avg Bulk Rate": "Bwd Blk Rate Avg",
    "Subflow Fwd Packets": "Subflow Fwd Pkts",
    "Subflow Fwd Bytes": "Subflow Fwd Byts",
    "Subflow Bwd Packets": "Subflow Bwd Pkts",
    "Subflow Bwd Bytes": "Subflow Bwd Byts",
    "Init_Win_bytes_forward": "Init Fwd Win Byts",
    "Init_Win_bytes_backward": "Init Bwd Win Byts",
    "act_data_pkt_fwd": "Fwd Act Data Pkts",
    "min_seg_size_forward": "Fwd Seg Size Min",
    "Active Mean": "Active Mean",
    "Active Std": "Active Std",
    "Active Max": "Active Max",
    "Active Min": "Active Min",
    "Idle Mean": "Idle Mean",
    "Idle Std": "Idle Std",
    "Idle Max": "Idle Max",
    "Idle Min": "Idle Min",
    "Label": "Label",
}

LABELS_2017_TO_2018 = {
    "BENIGN": "Benign",
    # https://stackoverflow.com/questions/50891292/python-replacement-not-allowed-with-overlapping-keys-and-values
    # "Bot": "Bot",
    "DDoS": np.nan,
    "DoS GoldenEye": "DoS attacks-GoldenEye",
    "DoS Hulk": "DoS attacks-Hulk",
    "DoS Slowhttptest": "DoS attacks-SlowHTTPTest",
    "DoS slowloris": "DoS attacks-Slowloris",
    "FTP-Patator": "FTP-BruteForce",
    "Heartbleed": np.nan,
    "Infiltration": "Infilteration",
    "PortScan": np.nan,
    "SSH-Patator": "SSH-Bruteforce",
    "Web Attack  Brute Force": "Brute Force -Web",
    "Web Attack  Sql Injection": "SQL Injection",
    "Web Attack  XSS": "Brute Force -XSS",
}

FEATURES_2018_TO_2017 = {value: key for key, value in FEATURES_2017_TO_2018.items()}

dataset_name = "cse-cic-ids2017"

DTYPE_MAP = {"int64": np.dtype(int), "object": np.dtype(str)}
DATA_TYPES = {
    FEATURES_2018_TO_2017[feature]: DTYPE_MAP[dtype]
    for feature, dtype in load.DTYPE.items()
}

dataset_name_clean = Path(dataset_name + "-clean")
dataset_name_clean.mkdir(parents=False, exist_ok=True)
for dataset_file in sorted(glob(f"{dataset_name}/*.csv")):
    file_name = dataset_file.split("/")[-1]
    print("#" * 80)
    print("New datafile:", dataset_file)
    print("#" * 80)
    df = pd.read_csv(
        dataset_file,
        usecols=DATA_TYPES.keys(),
        skipinitialspace=True,
        encoding="latin1",
    )
    # Replace 2017 columns names with the corresponding 2018 values
    df.rename(columns=FEATURES_2017_TO_2018, inplace=True)
    df = clean.get_clean_df(df, verbose=0)
    # Remove non-ascii characters from the Label column
    df["Label"] = df["Label"].str.encode("ascii", "ignore").str.decode("ascii")
    # Replace (or remove inexistent in 2018) labels
    df.replace({"Label": LABELS_2017_TO_2018}, inplace=True)
    df.dropna(axis=0, inplace=True)
    df.to_csv(f"{dataset_name_clean}/{file_name}", index=False)
