# Customer Segmentation Using Unsupervised Learning

## Overview

This project focuses on customer segmentation using unsupervised machine learning techniques. The main objective is to analyze customer financial behaviors and automatically group customers into meaningful segments based on their spending patterns, payment behaviors, credit usage, and transaction characteristics.

Customer segmentation is an important task in data mining and business analytics because it enables organizations to better understand their customers, design personalized marketing strategies, improve customer relationship management, and make data-driven decisions.

In this project, several clustering algorithms are implemented and compared to identify the most suitable approach for discovering hidden patterns within customer data.

---

## Project Objectives

The main goals of this project are:

- Explore and analyze customer financial behavior patterns.
- Identify natural groups of customers without using predefined labels.
- Compare different unsupervised learning algorithms for customer clustering.
- Investigate the impact of feature scaling techniques on clustering performance.
- Optimize clustering parameters to achieve better separation between customer groups.
- Visualize and interpret the discovered customer segments.

---

# Dataset Description

The dataset contains customer-level information related to credit card usage and financial activities.

Each row represents an individual customer, while each feature describes a specific aspect of customer behavior.

### Main Features

| Feature | Description |
|---|---|
| BALANCE | Current account balance |
| BALANCE_FREQUENCY | Frequency of balance updates |
| PURCHASES | Total purchase amount |
| ONEOFF_PURCHASES | Amount of one-time purchases |
| INSTALLMENTS_PURCHASES | Amount of installment purchases |
| CASH_ADVANCE | Amount of cash advances |
| PURCHASES_FREQUENCY | Frequency of purchases |
| PURCHASES_TRX | Number of purchase transactions |
| CASH_ADVANCE_FREQUENCY | Frequency of cash advances |
| CREDIT_LIMIT | Credit limit assigned to customers |
| PAYMENTS | Total payments made |
| MINIMUM_PAYMENTS | Minimum payments performed |
| PRC_FULL_PAYMENT | Percentage of full payments |
| TENURE | Duration of customer membership |

The customer identifier feature is removed because it does not contain meaningful behavioral information.

---

# Data Preprocessing

Before applying clustering algorithms, several preprocessing steps are performed:

### 1. Data Cleaning

- Identification of missing values.
- Removal of incomplete records.
- Removal of non-informative identifier columns.

### 2. Exploratory Data Analysis (EDA)

Several visualization techniques are applied to understand the dataset:

- Feature distributions.
- Boxplots for identifying data variability and possible outliers.
- Scatter plots for analyzing relationships between variables.
- Correlation heatmap for studying feature dependencies.
- Pairwise feature analysis.

### 3. Feature Scaling

Since clustering algorithms rely on distance calculations, feature normalization is performed using:

- **MinMaxScaler**
- **StandardScaler**

The impact of different scaling strategies is evaluated during model comparison.

---

# Clustering Algorithms

Three different unsupervised learning algorithms are implemented:

---

## 1. K-Means Clustering

K-Means is a centroid-based clustering algorithm that divides customers into groups by minimizing the distance between customers and their corresponding cluster centers.

### Purpose:
- Discover groups of customers with similar financial behaviors.
- Identify the optimal number of customer segments.

Different cluster numbers are tested, and the best configuration is selected based on clustering quality metrics.

---

## 2. DBSCAN

DBSCAN is a density-based clustering algorithm capable of identifying dense regions and detecting noisy observations.

### Purpose:
- Investigate whether customer groups can be discovered based on density patterns.
- Identify potential outliers.

Different values of epsilon (`eps`) are tested to find suitable clustering parameters.

---

## 3. Gaussian Mixture Model (GMM)

GMM is a probabilistic clustering approach that assumes the data is generated from a mixture of multiple Gaussian distributions.

### Purpose:
- Model customer groups from a probabilistic perspective.
- Compare probabilistic clustering performance with distance-based methods.

---

# Model Evaluation

The clustering algorithms are evaluated using:

## Silhouette Score

Silhouette Score measures how well-separated the generated clusters are.

A higher value indicates:

- Better cluster compactness.
- Better separation between different groups.

The performance of different algorithms and scaling techniques is compared based on this metric.

---

# Optimization Process

After comparing different clustering approaches:

- Different numbers of clusters are tested.
- The optimal number of customer groups is selected.
- The final clustering configuration is determined based on the best Silhouette Score.

The final model is obtained using optimized K-Means clustering.

---

# Visualization and Analysis

To better understand the clustering results, several visualization techniques are applied:

### PCA Visualization

Principal Component Analysis (PCA) is used to reduce the dimensionality of the dataset and visualize customer groups in a two-dimensional space.

### Cluster Size Analysis

The number of customers in each cluster is visualized to understand the distribution of customer segments.

### Cluster Center Analysis

Cluster centers are analyzed to identify behavioral differences between customer groups.

### Feature Average Heatmap

Average feature values for each cluster are calculated and visualized to compare customer characteristics across different segments.

---

# Results Summary

The experiments show that:

- K-Means provides the most effective clustering performance among the evaluated methods.
- Feature scaling significantly affects clustering quality.
- The optimal number of clusters is determined through performance evaluation.
- The final clustering model successfully separates customers into distinct behavioral groups.

The discovered segments can be further analyzed for applications such as:

- Customer profiling.
- Personalized marketing.
- Customer retention strategies.
- Financial behavior analysis.



# Project Workflow
