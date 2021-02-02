import json

from azureml.core import Webservice


def main(service):
    data = {
        "data": [
            {
                "Flow Duration": 3761843,
                "TotLen Fwd Pkts": 1441,
                "TotLen Bwd Pkts": 1731,
                "Fwd Pkt Len Std": 191,
                "Bwd Pkt Len Max": 1179,
                "Bwd Pkt Len Std": 405,
                "Flow Byts/s": 843,
                "Flow Pkts/s": 6,
                "Flow IAT Max": 953181,
                "Bwd IAT Min": 124510,
                "Bwd Header Len": 172,
                "Pkt Len Max": 1179,
                "Pkt Len Std": 279,
                "RST Flag Cnt": 1,
                "PSH Flag Cnt": 1,
                "ECE Flag Cnt": 1,
                "Init Fwd Win Byts": 8192,
                "Init Bwd Win Byts": 62644,
                "Fwd Seg Size Min": 20,
            }
        ]
    }
    input_data = json.dumps(data)

    output_data = service.run(input_data)
    predictions = json.loads(output_data).get("result")
    assert predictions == ["Benign"]


if __name__ == "__main__":
    main()
