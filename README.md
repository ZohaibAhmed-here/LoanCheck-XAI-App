# LoanCheck XAI App
> **A high-precision decision-support system for credit risk assessment, featuring real-time Explainable AI (XAI) transparency.**

## Project Overview
**LoanCheck XAI** is a next-generation intelligence system designed to bridge the gap between complex machine learning and banking transparency. While traditional credit scoring models often act as "black boxes," this application provides a dual-layered approach: **High-Precision Risk Assessment** combined with **Granular Explainability.**

Developed as part of advanced research in Data Science, this system ensures that every financial decision is not only accurate but also **auditable and interpretable.**

---

## System Preview
![App Dashboard](deployment/main.png) 
*Note: The interface features a real-time SHAP Waterfall plot to explain model predictions.*

---

## Key Features
* **Predictive Intelligence:** Leverages optimized Gradient Boosting to evaluate loan applications across multiple financial dimensions (CIBIL, Assets, Income).
* **SHAP Visual Intelligence:** Maps exactly how specific financial attributes (like CIBIL Score or Education) influenced the final approval or rejection.
* **Real-Time Modeling:** A dynamic Streamlit interface allowing for "What-If" credit analysis by adjusting applicant parameters on the fly.
* **Research-Backed:** Documentation of the full model training and evaluation process is available in the `Experiment Notebook` directory.

---

## Research Benchmarks
Our champion model was selected after rigorous testing against several ensemble methods:

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Gradient Boosting (Champion)** | **98.36%** | **0.98** | **0.98** | **0.98** |
| XGBoost | 97.45% | 0.97 | 0.97 | 0.97 |
| Random Forest | 96.12% | 0.96 | 0.96 | 0.96 |

---

## Repository Structure
* `app.py`: The main Streamlit application script.
* `gb_loan_model.pkl`: The serialized high-performance Gradient Boosting model.
* `Dataset/`: Contains the raw financial data used for training.
* `Experiment Notebook/`: Comprehensive Jupyter Notebook showing data cleaning, EDA, and model selection.
* `Deployment/`: UI screenshots and visual assets for documentation.

---

## 🛠️ Installation & Usage
To run the dashboard locally:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/ZohaibAhmed-here/LoanCheck-XAI-App.git](https://github.com/ZohaibAhmed-here/LoanCheck-XAI-App.git)