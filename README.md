# Alzheimer's Disease Data Analysis

This project focuses on analyzing data related to Alzheimer's disease. It involves cleaning, visualizing, and analyzing the data to identify patterns and relationships between features.

---

## Table of Contents
- [Project Description](#project-description)
- [Dataset](#dataset)
- [How to Run](#how-to-run)
- [Results and Visualizations](#results-and-visualizations)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

---

## Project Description
This project includes the following steps:
1. **Data Cleaning**: Removing duplicate rows and handling missing values.
2. **Data Visualization**: Creating various plots to explore data distributions and feature relationships.
3. **Data Analysis**: Performing statistical analysis to identify patterns in the data.

---

## Dataset
The dataset is loaded from `dataset.csv`. It includes information such as age, BMI, cognitive test scores, and Alzheimer's disease status.

---

## How to Run
To run this project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/zfarzaneh/dataset-analysis.git

## Results and Visualizations

### Plots

<table class="gray-table">
  <tr>
    <td style="background-color: #f0f0f0;">**Class Distribution**</td>
    <td style="background-color: #f0f0f0;">**Correlation Matrix**</td>
  </tr>
  <tr>
    <td><img src="df_plots/Figure_10.png" alt="Class Distribution" style="width:100%;"></td>
    <td><img src="df_plots/Figure_5.png" alt="Correlation Matrix" style="width:100%;"></td>
  </tr>
  <tr style="background-color: #f0f0f0;">
    <td>**Data Distribution**</td>
    <td>**Feature Distributions (Boxplot)**</td>
  </tr>
  <tr>
    <td><img src="df_plots/Figure_11.png" alt="Data Distribution" style="width:100%;"></td>
    <td><img src="df_plots/Figure_8.png" alt="Feature Distributions (Boxplot)" style="width:100%;"></td>
  </tr>
  <tr>
    <td style="background-color: #f0f0f0;">**Q-Q Plots for Dataset Columns**</td>
    <td style="background-color: #f0f0f0;">**Skewness Distribution of All Numerical Features**</td>
  </tr>
  <tr>
    <td><img src="df_plots/Figure_3.png" alt="Q-Q Plots for Dataset Columns" style="width:100%;"></td>
    <td><img src="df_plots/Figure_2.png" alt="Skewness Distribution of All Numerical Features" style="width:100%;"></td>
  </tr>
</table>


**To view more plots, please refer to the [df_plots](df_plots/) folder**
