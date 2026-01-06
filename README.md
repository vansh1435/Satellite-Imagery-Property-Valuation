🛰️ Satellite-Imagery-Based Property Valuation

Multimodal Machine Learning for Real Estate Price Prediction

🌟 Overview

This project explores whether satellite imagery can complement traditional tabular housing data for predicting property prices.

We build an end-to-end machine learning pipeline combining:

📊 Structured tabular features (area, rooms, quality, location)

🛰️ Satellite images capturing neighborhood context

🔗 Multimodal fusion models

👁️ Explainability using Grad-CAM

The project emphasizes experimental rigor and honest evaluation, rather than forcing performance gains.

🎯 Problem Statement

Traditional real-estate valuation models rely heavily on structured features such as:

Living area

Number of bedrooms & bathrooms

Construction quality

Geographic coordinates

However, these features fail to capture neighborhood-level context, including:

🌳 Green spaces

🌊 Water bodies

🛣️ Road connectivity

🏙️ Urban density & layout

💡 Research Question

Can satellite imagery improve property price prediction when combined with tabular data?

🧠 Project Approach

We implement three complementary models:

┌───────────────────────────────────────────┐
│           Property Location (lat, lon)    │
└──────────────────────┬────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
        ▼                             ▼
📋 Tabular Features            🛰️ Satellite Images
(beds, baths, sqft, etc.)       (urban context)
        │                             │
        ▼                             ▼
  XGBoost / MLP                 CNN (ResNet)
        │                             │
        └──────────────┬──────────────┘
                       ▼
                🔗 Multimodal Fusion
                       ▼
                 💰 Price Prediction

📂 Repository Structure
satellite-property-valuation/
│
├── data/
│   ├── raw/                  # Original train/test CSVs
│   ├── processed/            # Cleaned & filtered datasets
│   └── images/               # Satellite images (not committed)
│
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_xgboost_tabular.ipynb
│   ├── 03_image_only_model.ipynb
│   ├── 04_multimodal_fusion.ipynb
│   ├── 05_grad_cam.ipynb
│
├── src/
│   ├── data_fetcher.py       # Satellite image downloader
│   └── grad_cam.py           # Grad-CAM implementation
│
├── outputs/
│   └── predictions.csv       # Final test predictions
│
├── requirements.txt
├── README.md
└── .gitignore

🧪 Models Implemented
1️⃣ Tabular Model (XGBoost) — Baseline

Uses only structured housing features

Robust, fast, and highly accurate

✅ Best performance
✅ Strong baseline
🥇 Winner

2️⃣ Image-Only Model (CNN)

Satellite images → ResNet embeddings → regression

Captures neighborhood patterns

⚠️ Weak standalone signal
⚠️ Noisy predictions
🔴 Underperforms

3️⃣ Multimodal Fusion (Tabular + Images)

Early fusion of CNN image embeddings + tabular features

❓ Explores complementary signal
❌ Did not outperform tabular model
🟡 Insightful but limited by data

📊 Key Results (Summary)
Model	RMSE	R²	Performance
Tabular (XGBoost)	⭐ Lowest	⭐ Highest	🥇 Best
Image-Only CNN	⚠️ High	⚠️ Negative	🔴 Noisy
Multimodal Fusion	⬇️ Worse	⬇️ Lower	🟡 Did not improve
