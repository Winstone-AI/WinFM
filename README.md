
# 🏙️ Winstone Multimodal Foundation Model (WinFM)

**Proof-of-Concept Evaluation and Backtest for Real Estate Investment**

---
This project presents a proof-of-concept (POC) validation of Winstone’s WinFM using comprehensive real estate transaction data from London (2010–2024), aggregated at the Lower Layer Super Output Area (LSOA) level. We compare WinFM with other models including LLM, temporal models and traditional spatial-temporal models. WinFM achieves a technical breakthrough by unifying language understanding with spatial-temporal modeling in an efficient, scalable architecture—delivering the first effective multimodal solution for real estate forecasting. 
## 🛠️ How to Use

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/winstone
cd winstone
```

### 2️⃣ Install Dependencies

It is recommended to use Python 3.8+ and install the following libraries:

```bash
pip install pandas numpy scikit-learn pyarrow
```

Alternatively, use a `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Run Evaluation & Backtest

Run the following command to compute model prediction metrics and investment backtest results:

```bash
python evaluate.py --results_dir results
```

This will output:

- **Prediction Metrics**: RMSE, SMAPE, MASE, R²
- **Investment Backtest**: Total Return, CAGR, Success Rate

---

## 📊 Model Performance

### 📈 Price Prediction Metrics

| Model      | RMSE ↓   | SMAPE ↓  | MASE ↓   | R² ↑     |
|------------|-----------|-----------|-----------|-----------|
| GPT-4o     | 1801.2    | 14.21%    | 1.065     | 0.606     |
| Qwen       | 1709.9    | 13.77%    | 1.009     | 0.645     |
| Gemini-2   | 1596.0    | 13.47%    | 0.954     | 0.690     |
| RNN        | 1453.1    | 12.05%    | 0.905     | 0.743     |
| LSTM       | 1387.3    | 11.94%    | 0.887     | 0.766     |
| ST-RNN     | 1369.6    | 11.78%    | 0.876     | 0.772     |
| ST-LSTM    | 1352.6    | 11.69%    | 0.867     | 0.778     |
| Win-LSTR   | 1310.9    | 11.00%    | 0.822     | 0.791     |
| Win-LSTL   | 1301.0    | 10.92%    | 0.815     | 0.794     |
| **WinFM**  | **1258.9**| **10.56%**| **0.787** | **0.807** |

---

### 💸 Investment Backtest Results

| Model      | Total Return ↑ | CAGR ↑     | Success Rate ↑ |
|------------|----------------|-------------|----------------|
| Market Avg | 3.14%          | 1.56%       | 52.04%         |
| GPT-4o     | 5.63%          | 1.84%       | 52.28%         |
| Qwen       | 5.32%          | 1.74%       | 52.52%         |
| Gemini-2   | 11.34%         | 3.64%       | 58.76%         |
| RNN        | 10.63%         | 3.43%       | 59.09%         |
| LSTM       | 12.33%         | 3.95%       | 60.28%         |
| ST-RNN     | 12.32%         | 3.95%       | 61.08%         |
| ST-LSTM    | 12.96%         | 4.15%       | 62.07%         |
| Win-LSTR   | 14.14%         | 4.51%       | 61.50%         |
| Win-LSTL   | 14.86%         | 4.73%       | 62.37%         |
| **WinFM**  | **18.35%**     | **5.78%**   | **67.53%**     |

