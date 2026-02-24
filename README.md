# Stock Price Prediction Using Machine Learning

A comprehensive machine learning project demonstrating end-to-end ML pipeline development for financial time series prediction. This project explores stock price forecasting using traditional ML algorithms (Linear Regression, Random Forest, XGBoost) with detailed analysis of model limitations and practical trading applications.

## 🎯 Project Overview

This project investigates whether machine learning models can predict stock price movements and generate profitable trading strategies. Through systematic experimentation and analysis, I discovered critical limitations in price-based prediction approaches and identified solutions for more robust financial ML systems.

**Key Finding:** Tree-based models (Random Forest, XGBoost) fail when market prices exceed training data range due to inability to extrapolate, achieving only 41-43% directional accuracy in a bullish test period.

---

## 📊 Current Status

**✅ Version 1 Complete:**

- ✅ Data collection and exploratory analysis
- ✅ Feature engineering with 50+ technical indicators
- ✅ Baseline model implementation and comparison
- ✅ Financial metrics and backtesting framework
- ✅ Root cause analysis of model failures

**🚧 Version 2 (Planned):**

- ⏳ Return-based prediction implementation
- ⏳ LSTM deep learning model
- ⏳ Model robustness improvements

---

## 🔑 Key Findings & Lessons Learned

### **1. The "Training Ceiling" Problem**

**Discovery:** Random Forest and XGBoost models exhibited a critical failure mode where they couldn't predict prices above the training data range.

- **Training data:** Prices ranged $117-$181 (Oct 2020 - Feb 2024)
- **Test data:** Prices reached $163-$257 (Feb 2024 - Dec 2024)
- **Result:** Models capped predictions at ~$195, leading to 41-43% directional accuracy (worse than random 50%)

**Root Cause:** Tree-based models cannot extrapolate beyond their training range, unlike linear models which can extend learned relationships.

**Evidence:**

- Models predicted DOWN 75-86% of time in a bull market (60% UP days)
- Average predicted change: -$0.57 to -$23 vs actual +$0.33
- Trading strategies underperformed buy-and-hold by $2,300-$3,300 (-59% of potential gains)

### **2. Model Performance Comparison**

| Model                 | Directional Accuracy | Days Predicted UP | Trading Return | vs Buy-and-Hold |
| --------------------- | -------------------- | ----------------- | -------------- | --------------- |
| **Linear Regression** | 43.40%               | 54/212 (25.5%)    | +5.84%         | -$3,270         |
| **Random Forest**     | 43.40%               | 30/212 (14.2%)    | +14.81%        | -$2,373         |
| **XGBoost**           | 41.98%               | 31/212 (14.6%)    | +15.56%        | -$2,298         |
| **Buy & Hold**        | -                    | 212/212 (100%)    | **+38.53%**    | **baseline**    |

**Insight:** By staying in cash 75-86% of the time, ML strategies missed most gains in the bullish test period.

### **3. Feature Importance Analysis**

**Dominant Feature:** Current `Close` price accounted for 97.5% of model importance

- Other 25+ technical indicators: Only 2.5% combined importance
- **Implication:** For next-day prediction, recent price dominates; momentum indicators add minimal value

**Technical indicators tested:**

- Moving averages (5, 10, 20, 50, 200-day)
- RSI, MACD, Stochastic Oscillator
- Bollinger Bands, ATR, CCI, Williams %R
- Volume indicators and On-Balance Volume

### **4. Trading Strategy Results**

**Simulation:** $10,000 initial capital over 212 trading days (Feb-Dec 2024)

**Final Values:**

- Buy & Hold: **$13,853** (+38.5%)
- Best ML (XGBoost): $11,556 (+15.6%)
- Worst ML (Linear Reg): $10,584 (+5.8%)

**Sharpe Ratios:**

- Buy & Hold: 1.702 (Good - strong risk-adjusted returns)
- Random Forest: 1.263 (Good - conservative strategy pays off)
- XGBoost: 1.159 (Good - similar to RF)
- Linear Regression: 0.435 (Poor - high risk for low return)

**Maximum Drawdowns:**

- Random Forest: -5.80% (best - most defensive)
- XGBoost: -6.84% (second best)
- Linear Regression: -9.45%
- Buy & Hold: -11.75% (worst - full market exposure)

**Interpretation:**

- ML strategies reduced drawdown risk by staying in cash
- However, they sacrificed 60-80% of potential gains
- Random Forest achieved best risk-adjusted performance among ML models
- But all ML strategies significantly underperformed simple buy-and-hold

---

## 🛠️ Technical Stack

**Languages & Libraries:**

- Python 3.10+
- **Data Processing:** pandas, numpy
- **Data Source:** yfinance (Yahoo Finance API)
- **Technical Analysis:** ta (Technical Analysis Library)
- **Machine Learning:** scikit-learn, XGBoost
- **Visualization:** matplotlib, seaborn
- **Development:** Jupyter Notebook

---

## 📁 Project Structure

```
stock-prediction/
├── notebooks/
│   ├── 01_data_collection.ipynb              # Data acquisition & EDA
│   ├── 02_feature_engineering.ipynb          # Technical indicators
│   ├── 03_baseline_models.ipynb              # Model training & comparison
│   └── 04_financial_metrics_backtesting.ipynb # Trading strategy evaluation
├── data/
│   ├── AAPL_historical.csv                   # Raw price data
│   └── AAPL_engineered_features.csv          # Processed features
├── models/
│   ├── linear_regression.pkl                 # Trained models
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   └── scaler.pkl                            # Feature scaler
├── results/
│   ├── model_comparison.csv                  # Baseline metrics
│   └── trading_strategy_comparison.csv       # Financial performance
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🚀 Getting Started

### **Prerequisites**

- Python 3.8 or higher
- pip package manager

### **Installation**

1. **Clone the repository**

```bash
git clone https://github.com/YOUR-USERNAME/stock-prediction.git
cd stock-prediction
```

2. **Create virtual environment**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Run notebooks**

```bash
jupyter notebook
```

5. **Execute notebooks in order:**
   - `01_data_collection.ipynb` - Download stock data
   - `02_feature_engineering.ipynb` - Create features
   - `03_baseline_models.ipynb` - Train models
   - `04_financial_metrics_backtesting.ipynb` - Evaluate strategies

---

## 📈 Methodology

### **1. Data Collection**

- Stock: Apple Inc. (AAPL)
- Period: Oct 2020 - Dec 2024 (1,058 trading days)
- Source: Yahoo Finance via yfinance API
- Features: OHLCV (Open, High, Low, Close, Volume)

### **2. Feature Engineering (26 features)**

**Price-based:**

- Lagged prices (1, 2, 3, 5 days)
- Price changes and percentage changes
- High-low spreads, open-close differences

**Moving Averages:**

- Simple: 5, 10, 20, 50-day
- Exponential: 12, 26-day (for MACD)
- Crossovers and ratios

**Momentum Indicators:**

- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Stochastic Oscillator, CCI, Williams %R

**Volatility:**

- ATR (Average True Range)
- Bollinger Bands position

**Volume:**

- Volume ratios and changes
- On-Balance Volume (OBV)

### **3. Model Training**

**Train/Test Split:**

- Training: 846 samples (Oct 2020 - Feb 2024) - 80%
- Testing: 212 samples (Feb 2024 - Dec 2024) - 20%
- **No shuffling** (preserves temporal order)

**Models Trained:**

1. Linear Regression (baseline, can extrapolate)
2. Random Forest (100 trees, max_depth=10)
3. XGBoost (100 estimators, learning_rate=0.1)

**Evaluation Metrics:**

- RMSE, MAE, R² (price prediction accuracy)
- Directional Accuracy (UP/DOWN prediction)
- Sharpe Ratio (risk-adjusted returns)
- Maximum Drawdown (worst loss from peak)

### **4. Trading Strategy**

- **Rule:** Buy when model predicts UP, hold cash when predicts DOWN
- **Capital:** $10,000 initial investment
- **Period:** 212 trading days (Feb-Dec 2024)
- **Benchmark:** Buy-and-hold strategy

---

## 💡 Implications & Solutions

### **Why Price-Based Prediction Failed**

**Problem:** When Apple's stock price exceeded $180 (training maximum) and reached $260 in the test period, tree-based models couldn't adapt.

**Why It Matters:**

- Financial markets regularly reach new all-time highs
- Models trained on historical prices become obsolete
- This is a fundamental limitation of tree-based regression on non-stationary data

### **Proposed Solutions**

**1. Return-Based Prediction (Recommended)**

- Predict percentage returns instead of absolute prices
- Returns are more stationary than prices
- Works across any price level

```python
# Instead of: target = next_day_price ($180)
# Use: target = (next_day_price - current_price) / current_price (1.5%)
```

**2. Frequent Model Retraining**

- Retrain models monthly as new price levels are reached
- Use rolling window (e.g., most recent 2 years)
- Computationally expensive but maintains accuracy

**3. Ensemble with Linear Models**

- Combine tree models (good at patterns) with linear models (can extrapolate)
- Weight based on recent performance

**4. Regime Detection**

- Detect when market enters new price territory
- Trigger retraining or switch to alternative models
- Use statistical tests for distribution shifts

---

## 🎓 Skills Demonstrated

### **Technical Skills**

- Time series analysis and forecasting
- Feature engineering for financial data
- Machine learning model development and comparison
- Model diagnostics and failure analysis
- Trading strategy backtesting
- Data visualization and communication

### **Domain Knowledge**

- Technical analysis indicators and their applications
- Financial metrics (Sharpe ratio, drawdown, directional accuracy)
- Trading strategy development
- Market behavior and price dynamics
- Risk management concepts

### **Software Engineering**

- Version control with Git/GitHub
- Project organization and documentation
- Reproducible research with Jupyter notebooks
- Virtual environment management
- Code modularity and reusability

### **Problem-Solving**

- Root cause analysis of model failures
- Diagnostic methodology development
- Critical evaluation of results
- Solution proposal and validation

---

## 📚 What I Learned

1. **Feature Importance ≠ Predictive Power**
   - Close price dominated feature importance (97.5%)
   - But high importance doesn't mean better predictions
   - Need to consider feature stability across different market conditions

2. **Extrapolation Limitations**
   - Tree-based models can only predict values within training range
   - Linear models can extrapolate but may be less accurate within range
   - Model choice depends on whether data will stay within training distribution

3. **Evaluation Metrics Matter**
   - Low RMSE doesn't guarantee profitable trading
   - Directional accuracy is more important for strategy development
   - Financial metrics (Sharpe, drawdown) reveal practical viability

4. **Market Non-Stationarity**
   - Financial markets evolve and reach new price levels
   - Models must adapt to changing distributions
   - Stationary features (returns) are more robust than non-stationary (prices)

5. **Negative Results Have Value**
   - Discovering what doesn't work is as valuable as finding what does
   - Thorough diagnosis leads to better solutions
   - Documentation of failures prevents repeating mistakes

---

## 🔮 Future Work

### **Short-term (Version 2)**

- [ ] Implement return-based prediction
- [ ] Retrain all models with percentage returns as target
- [ ] Compare price-based vs return-based performance
- [ ] Expected improvement: >50% directional accuracy

### **Medium-term**

- [ ] Add LSTM/GRU models for sequence learning
- [ ] Implement walk-forward validation
- [ ] Test on multiple stocks (MSFT, GOOGL, TSLA)
- [ ] Add sentiment analysis from news/social media

### **Long-term**

- [ ] Deploy real-time prediction system
- [ ] Implement reinforcement learning trading agent
- [ ] Portfolio optimization across multiple assets
- [ ] Paper trading with live data

---

## ⚠️ Disclaimer

This project is for **educational purposes only**. Stock price prediction is inherently uncertain, and past performance does not guarantee future results.

**Do not use these models for actual trading without:**

- Proper risk management
- Understanding of financial markets
- Professional financial advice
- Thorough backtesting with transaction costs
- Regulatory compliance

The models developed here have demonstrated poor performance (41-43% accuracy) and should **not** be used for real trading decisions.

---

## 📫 Contact

**Your Name**

- GitHub: [@YOUR-USERNAME](https://github.com/YOUR-USERNAME)
- LinkedIn: [Your LinkedIn Profile](https://linkedin.com/in/your-profile)
- Email: your.email@example.com

---

## 🙏 Acknowledgments

- Data provided by Yahoo Finance via yfinance library
- Technical indicators from the ta (Technical Analysis) library
- Inspired by quantitative finance and algorithmic trading research
- Special thanks to the open-source community

---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).

---

_Last Updated: December 2024_

_This project demonstrates machine learning engineering skills, financial domain knowledge, and critical analysis capabilities relevant to roles in data science, quantitative finance, and ML engineering._
