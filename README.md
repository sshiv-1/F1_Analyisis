# 🏎️ F1 Qualifying Prediction Model

This Python project uses [FastF1](https://theoehrly.github.io/Fast-F1/) and machine learning (Gradient Boosting Regressor) to predict **Q3 qualifying lap times** for Formula 1 races.  
It combines historical qualifying data, driver and team performance factors, and statistical modeling to simulate rankings for upcoming races.

---

## 📌 Features
- **Automatic F1 Data Fetching** from the official F1 API using FastF1.
- **Data Cleaning & Preprocessing**:
  - Converts lap times to seconds.
  - Handles missing values with median imputation.
- **Machine Learning**:
  - Trains a Gradient Boosting Regressor on past qualifying results.
  - Evaluates performance using MAE and R² score.
- **Performance Factors**:
  - Applies custom 2025-specific team and driver performance multipliers.
  - Adds small random variation for realism.
- **Race Prediction**:
  - Generates predicted Q3 times and rankings for the **2025 Dutch GP**.

---

## 📂 Project Structure
├── cache/ # FastF1 cache for faster repeated runs
├── f1_predict.py # Main script (this file)
├── README.md # Documentation (this file)
