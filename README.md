# ACIS – End-to-End Insurance Risk Analytics & Predictive Modeling

This repository contains the deliverables for the AlphaCare Insurance Solutions (ACIS) Analytics Challenge. The project covers end-to-end data analysis, predictive modeling, and premium optimization for an insurance portfolio.

Project Overview

The goal of this project is to analyze historical insurance claim data to:

Identify low- and high-risk customer segments

Test differences in risk across provinces, zip codes, and client demographics

Build predictive models for claim probability and severity

Enable dynamic pricing and premium optimization for AlphaCare Insurance Solutions

Data

Source: MachineLearningRating_v3.txt
Period: Feb 2014 – Aug 2015
Rows: 1,000,098
Columns: 52

Data Categories:

Policy information

Client demographics

Location

Vehicle details

Insurance plan and payment information

Claim metrics

Notes:

Cleaned dataset is stored in data/interim/cleaned.csv

Derived features include lossratio and has_claim

Tasks Completed
Task 1: Exploratory Data Analysis

Summary statistics and distributions of TotalPremium, TotalClaims, and loss ratios

Identification of outliers and skewed distributions

Geographic, vehicle, and client-level trends analyzed

Visualizations: barplots, histograms, boxplots

Task 2: Data Version Control (DVC)

Initialized DVC for reproducible workflow

Tracked raw and cleaned datasets without storing them in Git

Configured local remote storage and pushed datasets to remote

Ensured full reproducibility for collaborative analysis

Task 3: Hypothesis Testing

Chi-square tests for claim risk by province and gender

ANOVA for claim severity by zip code

Findings limited by skewed data; no statistically significant group differences observed

Pipeline validated for future analysis

Task 4: Predictive Modeling

Built pipelines for:

Regression models (claim severity): LinearRegression, RandomForestRegressor, XGBRegressor

Classification models (claim probability): RandomForestClassifier, XGBClassifier

SHAP analysis for feature importance

Actual model training skipped due to insufficient positive claims in cleaned dataset

Ready-to-run modeling pipelines for future datasets

Modeling & Analysis

Pipelines created using scikit-learn and XGBoost

Preprocessing via ColumnTransformer (numeric passthrough + OneHotEncoder for categorical features)

Model evaluation metrics: RMSE, R², Accuracy, Precision, Recall, F1-score

SHAP analysis included for interpretability
