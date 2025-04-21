# Megaline_Plan_Selection
Megaline is a mobile carrier trying to develop a model to analyze subscribers' behavior to recommend one of Megaline's newer plans: Smart or Ultra.   Project success will be in developing a model with the highest possible accuracy (0.75) that will pick the right plan.

📊 Megaline Plan Selection



📌 Description
This project analyzes user behavior to support mobile plan selection for a telecommunications company, Megaline. Using a dataset of user usage patterns and demographics, the notebook applies data preprocessing, exploratory analysis, and modeling to identify patterns that can inform the company’s mobile plan offerings.

💡 Features
Data cleaning and feature engineering

Exploratory Data Analysis (EDA) with visualizations

Classification models to predict user plan choice

Evaluation of model performance (accuracy, precision, recall)

Insights into customer behavior to optimize plan offerings

🧪 Technologies Used
Python 3.8+

Jupyter Notebook

Pandas, NumPy

Matplotlib, Seaborn

Scikit-learn

🚀 How to Use
Clone this repo:

bash
Copy
Edit
git clone https://github.com/yourusername/megaline-plan-selection.git
Open the notebook:

bash
Copy
Edit
jupyter notebook "Megaline_Plan_Selection.ipynb"
Run cells in order to see analysis, modeling, and conclusions.

📌 Project Summary
Megaline, a mobile carrier, aims to recommend one of two newer mobile plans — Smart or Ultra — using machine learning to analyze user behavior. The goal is to build a binary classification model that meets or exceeds 75% accuracy in plan recommendation.

🧹 Data Overview
Each row in the dataset represents a single user, with the following features:

calls — Number of calls

minutes — Total call duration in minutes

messages — Number of text messages

mb_used — Internet traffic used (MB)

is_ultra — Target label (1 = Ultra plan, 0 = Smart plan)

⚙️ Models Tested
The notebook evaluated three classification models:

1. Random Forest Classifier
Best performance with n_estimators = 52

Validation accuracy: 78.6%

Test accuracy: 81.4%

✅ Best-performing model, exceeding the target threshold

2. Decision Tree Classifier
Best depth varied slightly below tested depths (6, 10, 50, 100)

Validation accuracy: 77.9%

Test accuracy: Lower than on validation set (possible underfitting)

3. Logistic Regression
Validation accuracy: 67.8%

Test accuracy: 74.8%

🟡 Better than random chance, but worse than tree-based models

🏁 Conclusion
Random Forest is the recommended model for production deployment.
It achieved the highest test set accuracy (81.4%) and met the business goal of surpassing 75% accuracy in plan prediction. Logistic Regression and 

📄 License
This project is licensed under the MIT License.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Platform](https://img.shields.io/badge/Platform-JupyterLab%20%7C%20Notebook-lightgrey.svg)
![Status](https://img.shields.io/badge/Status-Exploratory-blueviolet.svg)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)
