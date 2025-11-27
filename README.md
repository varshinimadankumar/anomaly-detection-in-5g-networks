🚀 5G Network Anomaly Detection System

GNN • Autoencoder • OCSVM • Random Forest • FastAPI • Streamlit Dashboard

This project is an end-to-end anomaly detection pipeline for 5G network traffic.
It combines deep learning, graph neural networks, and classical machine learning to detect malicious or abnormal network behavior in real time.

The system includes:

⚡ FastAPI inference server

📊 Streamlit analytics dashboard

🧠 Graph Neural Network (GNN) model

🔗 Ensemble fusion model (AE + OCSVM + RF + GNN)

📁 Support for CSV & Parquet network traffic files

📌 Features
🔹 Multi-Model Anomaly Detection

Autoencoder (AE) – reconstruction error based anomaly score

OCSVM – boundary anomaly detection

Random Forest – classical ML scoring

GNN – graph-aware anomaly reasoning

Ensemble Fusion – combines all scores for final ranking

🔹 Real-Time Inference API

Accepts batch traffic windows

Handles 1 or multiple rows

Returns model scores + fused score

Exposed via FastAPI (/predict)

🔹 Interactive Visualization Dashboard

Upload CSV/Parquet

Choose number of rows to scan

Visualize trends, anomalies, suspicious hosts

Drill-down on any traffic window

Highlight Top-50 suspicious observations
📁 Project Structure
5g-anomaly-detection/
│
├── src/
│   ├── models/
│   │   ├── gnn.py
│   │   ├── autoencoder.py
│   │   ├── ocsvm.py
│   │   ├── random_forest.py
│   │   ├── ensemble.py
│   │
│   ├── inference/
│   │   ├── predict_utils.py
│   │   ├── server.py
│   │
│   ├── dashboard/
│   │   ├── app.py
│   │
│   ├── training/
│       ├── train_gnn.py
│       ├── train_ae.py
│       ├── train_ocsvm.py
│       ├── train_rf.py
│
├── saved_models/
│   ├── gnn_model.pth
│   ├── ae_model.pth
│   ├── ocsvm.pkl
│   ├── rf.pkl
│
├── requirements.txt
└── README.md

⚙️ Installation
1️⃣ Clone the repository
git clone https://github.com/your-username/5g-anomaly-detection.git
cd 5g-anomaly-detection

2️⃣ Install dependencies
pip install -r requirements.txt


Recommended Python version: 3.9 or 3.10

🚀 Run the FastAPI Server
uvicorn src.inference.server:app --reload --port 8000

Open API docs:

👉 http://localhost:8000/docs

🎛 Run the Streamlit Dashboard
streamlit run src/dashboard/app.py


Dashboard opens at:

👉 http://localhost:8501/

📡 API Usage
Endpoint
POST /predict

Request Body
{
  "data": [
    {
      "packet_count": 120,
      "byte_count": 45000,
      "avg_packet_len": 300,
      "std_packet_len": 12,
      "iat_mean": 0.10,
      "iat_std": 0.05,
      "tls_client_hello": 0,
      "tls_server_hello": 0,
      "source": "10.1.1.1",
      "destination": "10.1.1.2",
      "protocol": "TCP",
      "domain": "example.com"
    }
  ]
}

Response Example
[
  {
    "ae_score": -2.14,
    "ocsvm_score": -4.75,
    "rf_score": 0.0,
    "gnn_score": 0.437,
    "fused_score": -1.07
  }
]

📊 Dashboard Highlights

✔ Fused anomaly score timeline
✔ Comparison of AE / OCSVM / RF / GNN
✔ Top-50 most suspicious network windows
✔ Suspicious source IP detection
✔ Protocol-wise anomaly distribution
✔ Single-window deep inspection

The dashboard allows selecting how many rows to analyze, preventing unnecessary scanning of entire datasets.

🧠 Training the GNN Model
python src/training/train_gnn.py


Model output is saved in:

saved_models/gnn_model.pth

🧪 Testing the API

Curl example:

curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d @sample.json

🤝 Contributing

Pull requests are welcome.
For major changes, open an issue first to discuss the proposal.

🛡 License

This project is licensed under the MIT License.

⭐ Acknowledgements

This project integrates modern ML techniques for cybersecurity research, combining GNNs with classical detection algorithms for robust anomaly detection in 5G traffic.
