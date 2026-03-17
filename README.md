# Flux_AI
Hybrid time-series forecasting pipeline (NeuralProphet + XGBoost Residual Boosting) designed for volatile income streams. Features automated expense coverage analysis, 80% confidence interval bands, and persistent JSON-based state management.
Smart financial forecasting for the modern gig worker.
Flux AI is a personal financial strategist built for the unpredictable. If you're a freelancer, delivery partner, or gig-economy professional, your income doesn't come in a steady paycheck—it fluctuates. Flux uses a hybrid machine-learning approach to predict those ups and downs, helping you build a safety net (The Vault) before the "lean" days even arrive.

🧠 How the "Brain" Works
Most AI models struggle with the "chaos" of daily life. Flux handles this by using two different AI "personalities" working together:

The Trend-Spotter (NeuralProphet): It looks at the big picture—identifying your weekly patterns, long-term trends, and even accounting for Indian national holidays that affect your earnings.

The Pattern-Matcher (XGBoost): It looks at the "mistakes" of the first model. It learns from your 7-day rolling averages and weekend behavior to catch the sharp spikes or dips that standard models miss.

The Result: A highly accurate, hybrid forecast that is significantly more reliable than a simple average.

🏦 The "Vault" System
Flux doesn't just tell you the future; it helps you prepare for it. The engine calculates your Daily Survival Threshold (Fuel + Loans + Emergencies) and acts as your automated CFO:

During Surplus: When the AI predicts a "good month," it automatically suggests moving 40% of the extra cash into your virtual Vault.

During Deficits: If a shortfall is predicted, Flux "releases" funds from your Vault to cover your essentials, keeping your stress levels low.

Persistent Memory: Every time you run the engine, it remembers your Vault balance, tracking your financial resilience over time.

🛠️ Built for Performance
Hybrid Residual Boosting: Combines PyTorch-based time-series forecasting with Gradient Boosting.

Self-Evaluating: The engine includes a built-in backtesting suite that measures its own error rate (MAPE) so you know exactly how much to trust the prediction.

Visual Dashboard: Generates a clear, two-panel chart showing your income trajectory against your survival line and your Vault’s growth history.

🚀 Getting Started
Flux is optimized for Google Colab. Just upload your historical CSV (at least 30 days of data), and the engine handles the feature engineering, model training, and financial analysis in one click.

Required CSV Columns:
Net_Income: What you actually earned.

Fuel_or_Expense: What it cost you to work.

Loan_Repayment: Your daily debt obligations.

Emergency_Expense: The "unforeseen" costs.
