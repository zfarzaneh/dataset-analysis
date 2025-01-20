# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 16:06:25 2025
@author: zeina

Summary of the Code:

This script is designed for comprehensive data analysis and visualization, specifically tailored for a dataset related to Alzheimer's disease. It performs the following key tasks:

1. **Data Cleaning:**
   - Removes duplicate rows and unnecessary columns.
   - Renames columns for better readability.

2. **Handling Missing Values:**
   - Uses KNN Imputer to fill missing values in the dataset.

3. **Outlier Detection and Removal:**
   - Identifies and removes outliers using the Interquartile Range (IQR) method while maintaining class balance.

4. **Data Visualization:**
   - Plots mean and standard deviation of selected features.
   - Visualizes feature distributions using boxplots, histograms, and Q-Q plots.
   - Generates pairwise relationships between features.
   - Creates correlation matrices and class distribution charts.

5. **Statistical Analysis:**
   - Performs t-tests to compare characteristics between groups (e.g., Alzheimer's vs. non-Alzheimer's).
   - Checks for normality using Shapiro-Wilk, D'Agostino, and Anderson-Darling tests.

6. **Exporting Results:**
   - Exports summary statistics and visualizations to files for further analysis.

Key Features:
- **Modularity:** The code is organized into reusable functions, making it easy to extend or modify.
- **Visualization:** Utilizes `matplotlib` and `seaborn` for creating informative and visually appealing plots.
- **Error Handling:** Includes basic checks to ensure data integrity and handle potential issues gracefully.

Usage:
- Ensure the dataset is loaded correctly and contains the required columns.
- Adjust parameters (e.g., `class_column_name`, `threshold`) as needed for your specific dataset.
- Run the script to generate visualizations, statistical summaries, and cleaned data for further analysis.

Example Workflow:
1. Load the dataset.
2. Clean the data by removing duplicates and unnecessary columns.
3. Handle missing values using KNN Imputer.
4. Detect and remove outliers while preserving class balance.
5. Generate visualizations and statistical summaries.
6. Export results for reporting or further analysis.
"""

#=========================================================================
#       Import Libraries
#=========================================================================
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
from scipy.stats import ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro, normaltest, anderson
import math
from scipy.stats import skew
from matplotlib.table import Table
import warnings
from statsmodels.graphics.gofplots import qqplot
from typing import List, Dict, Any
# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
plt.ion()
#=========================================================================
#        Plotting and Visualizations Functions
#=========================================================================
#Plots the mean and standard deviation of selected columns in the DataFrame
def plot_mean_standardDeviation(df: pd.DataFrame, title: str) -> None:
    """
    Plots the mean and standard deviation of selected columns in the DataFrame.

    This function selects specific columns with the '_point' suffix, calculates their mean and standard deviation,
    and visualizes the results using bar plots. It also exports the summary statistics to a text file.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        title (str): The title of the plot.

    Returns:
        None
    """
    # Select columns with '_point' suffix for numerical analysis
    int_columns: List[str] = [
        'form_personal_info_point', 'MMSE_point', 'CSI-D_point', 'MNA-SF_point',
        'GDS-15_point', 'GAD-7_point', 'IADL_point', 'ADL_point', 'walking_efficiency_point'
    ]

    # Calculate mean and standard deviation for selected columns
    summary_stats: pd.DataFrame = df[int_columns].describe().loc[['mean', 'std']].transpose()
    
    # Plotting mean and standard deviation
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Plot mean values
    axes[0].bar(summary_stats.index, summary_stats['mean'], color='steelblue', alpha=0.8)
    axes[0].set_title('Mean Values', fontsize=14)
    axes[0].set_xlabel('', fontsize=12)
    axes[0].set_ylabel('Mean', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45, labelsize=10)
    axes[0].tick_params(axis='y', labelsize=10)

    # Plot standard deviation values
    axes[1].bar(summary_stats.index, summary_stats['std'], color='tomato', alpha=0.8)
    axes[1].set_title('Standard Deviation', fontsize=14)
    axes[1].set_xlabel('', fontsize=12)
    axes[1].set_ylabel('Standard Deviation', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45, labelsize=10)
    axes[1].tick_params(axis='y', labelsize=10)

    # Adjust layout and show plot
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Space for the main title
    plt.show()
    
    # Print and export summary statistics
    print("Summary statistics (mean, std) for each column:")
    print(summary_stats)
    output_file: str = 'summary_stats.txt'
    summary_stats.to_csv(output_file, sep='\t')
    print(f"Summary statistics exported to {output_file}")

#Plots the mean and standard deviation of numeric features in the DataFrame.    
def plot_mean_standardDeviation_of_features(df: pd.DataFrame, numeric_columns: list) -> None:
    """  
    This function calculates the mean and standard deviation for the specified numeric columns
    and visualizes them using a bar plot. The plot helps in understanding the central tendency
    and variability of the features.
    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        numeric_columns (list): A list of column names (strings) representing numeric features.

    Returns:
        None
    """
    # Calculate mean and standard deviation for the specified numeric columns
    summary_stats: pd.DataFrame = df[numeric_columns].describe().loc[['mean', 'std']].transpose()

    # Plot the mean and standard deviation as a bar plot
    summary_stats.plot(kind='bar', figsize=(12, 12), color=['blue', 'orange'], alpha=0.7)

    # Set plot title and labels
    plt.title('Mean and Standard Deviation of Features', fontsize=14)
    plt.ylabel('Value', fontsize=12)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=35, ha='right', fontsize=8)

    # Display the plot
    plt.show()
    
#Plots the results of t-tests comparing features between two groups (e.g., Alzheimer's vs. non-Alzheimer's).
def plot_t_test_results(df: pd.DataFrame, title: str, class_column: str) -> None:
    """    
    This function performs independent t-tests for specified features between two groups defined by the `class_column`.
    It calculates the mean values for each group and the p-value from the t-test. The results are displayed in a table
    and saved as an image file.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        title (str): The title of the table/plot.
        class_column (str): The column name in the DataFrame that defines the two groups (e.g., 0 and 1).

    Returns:
        None
    """
    # Ensure the labels in the class column are integers
    df[class_column] = df[class_column].astype(int)

    # Features selected for analysis
    features: list = [
        'age', 'form_personal_info_point', 'MMSE_point', 'CSI-D_point', 
        'MNA-SF_point', 'GDS-15_point', 'GAD-7_point', 'IADL_point', 
        'ADL_point', 'walking_efficiency_point', 'smoker', 'marriage'
    ]

    # Store the results of the t-test
    results: list = []

    # Calculate means and p-value for each feature
    for feature in features:
        mean_0: float = df[df[class_column] == 0][feature].mean()  # Mean for group 0
        mean_1: float = df[df[class_column] == 1][feature].mean()  # Mean for group 1
        t_stat, p_value = ttest_ind(
            df[df[class_column] == 0][feature],  # Data for group 0
            df[df[class_column] == 1][feature],  # Data for group 1
            nan_policy='omit'  # Ignore NaN values
        )

        # Append results to the list
        results.append({
            'Feature': feature,
            'Mean (No Alzheimer)': mean_0,
            'Mean (Alzheimer)': mean_1,
            'p-value': p_value
        })

    # Convert the results to a DataFrame
    results_df: pd.DataFrame = pd.DataFrame(results)

    # Format numbers to display with 5 decimal places
    results_df['Mean (No Alzheimer)'] = results_df['Mean (No Alzheimer)'].apply(lambda x: f'{x:.5f}')
    results_df['Mean (Alzheimer)'] = results_df['Mean (Alzheimer)'].apply(lambda x: f'{x:.5f}')
    results_df['p-value'] = results_df['p-value'].apply(lambda x: f'{x:.5f}')

    # Create a plot with specific size for displaying the table
    fig, ax = plt.subplots(figsize=(14, 18))  # Adjusted size for better visibility of rows
    ax.axis('tight')
    ax.axis('off')  # Turn off the axes (we are showing a table, not a graph)

    # Set title for the table with some padding to avoid overlap
    ax.set_title(title, fontsize=14, pad=40, fontweight='bold')

    # Create a table with the comparison data and place it at the center
    tbl = Table(ax, bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)  # Adjust font size for readability
    tbl.auto_set_column_width([0, 1, 2, 3, 4])  # Automatically adjust column widths
    tbl.scale(1.2, 1.2)  # Scale the table to make it more visible

    # Add the rows to the table
    for (i, row) in enumerate(results_df.values):
        for j, value in enumerate(row):
            tbl.add_cell(i, j + 1, width=0.2, height=0.1, text=str(value), loc='center')

    # Add header
    for i, column in enumerate(results_df.columns):
        tbl.add_cell(-1, i + 1, width=0.2, height=0.1, text=column, loc='center', facecolor='#ADD8E6')

    # Add the table to the axes
    ax.add_table(tbl)

    # Adjust the layout to prevent overlap of title and table
    plt.subplots_adjust(top=0.85)  # Adjust the top margin to create space for the title

    # Save the table as an image file
    plt.savefig('results_table_with_title.png', bbox_inches='tight', dpi=300)

    # Show the plot (which contains the table)
    plt.show()

#Plots Q-Q (Quantile-Quantile) plots for all numeric columns in the DataFrame to check for normality.
def plotQQ(df: pd.DataFrame, title: str) -> None:
    """   
    This function generates Q-Q plots for each numeric column in the DataFrame. Q-Q plots are used to visually
    assess whether the data follows a normal distribution. The plots are arranged in a grid with up to 3 plots per row.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        title (str): The title of the overall plot.

    Returns:
        None
    """
    # Identify numeric columns (int64 or float64)
    numeric_columns: list = [col for col in df.columns if df[col].dtype in [np.int64, np.float64]]
    
    # Check if there are any numeric columns
    if not numeric_columns:
        print("No numeric columns found for Q-Q Plot.")
        return

    # Determine the number of rows and columns for the subplot grid
    num_cols: int = len(numeric_columns)
    num_rows: int = (num_cols + 2) // 3  # 3 plots per row
    
    # Create subplots
    fig, axes = plt.subplots(num_rows, 3, figsize=(15, num_rows * 5))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Flatten the axes array for easy indexing
    axes = axes.flatten()
    
    # Generate Q-Q plots for each numeric column
    for i, column in enumerate(numeric_columns):
        qqplot(df[column].dropna(), line='s', ax=axes[i])  # 's' indicates a standardized line
        axes[i].set_title(f'Q-Q Plot: {column}', fontsize=8, fontweight='bold')
        
    # Hide any empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    # Adjust layout to prevent overlap with the title
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Display the plot
    plt.show()
    
#Plots a correlation matrix for the numeric columns in the DataFrame.
def plotCorrelationMatrix(df: pd.DataFrame, title: str) -> None:
    """  
    This function calculates the correlation matrix for all numeric columns in the DataFrame
    and visualizes it as a heatmap. The heatmap uses a color gradient to represent correlation
    values, with annotations displaying the exact correlation coefficients.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        title (str): The title of the correlation matrix plot.

    Returns:
        None
    """
    # Calculate the correlation matrix for numeric columns
    correlation_matrix: pd.DataFrame = df.corr()

    # Create a large figure to accommodate the heatmap
    plt.figure(figsize=(26, 26))

    # Plot the heatmap with annotations and a color gradient
    sns.heatmap(
        correlation_matrix, 
        annot=True,  # Display correlation values on the heatmap
        cmap='coolwarm',  # Color map for the heatmap
        fmt=".2f",  # Format for annotation values (2 decimal places)
        annot_kws={"size": 10},  # Font size for annotations
        cbar_kws={"shrink": 0.8}  # Adjust the size of the color bar
    )

    # Set the title of the plot
    plt.title(title, fontsize=16,fontweight="bold")

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=9)

    # Adjust y-axis label font size
    plt.yticks(fontsize=9)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Adjust top margin to move the title away from the plot
    plt.subplots_adjust(top=0.92)  # Decrease the value to move the title further from the plot

    # Add extra space at the bottom to prevent cutting off labels
    plt.subplots_adjust(bottom=0.35)

    # Display the plot
    plt.show()

#Visualizes the distribution of each column in the DataFrame using histograms with KDE (Kernel Density Estimation).
def plot_dataColumnDistribution(df: pd.DataFrame, title: str) -> None:
    """ 
    This function creates a grid of histograms, one for each column in the DataFrame. Each histogram is accompanied
    by a KDE curve to provide a smooth estimate of the data distribution. The grid is dynamically sized based on the
    number of columns, with up to 4 plots per row.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        title (str): The title of the overall plot.

    Returns:
        None
    """
    # Determine the number of columns and rows for the subplot grid
    num_columns: int = df.shape[1]  # Number of columns in the DataFrame
    num_cols: int = 4  # Number of columns in the grid
    num_rows: int = (num_columns + num_cols - 1) // num_cols  # Calculate number of rows needed

    # Increase the figure size to avoid overlap and make the subplots larger
    plt.figure(figsize=(num_cols * 6, num_rows * 12))  # Adjust size based on the number of columns and rows

    # Hide the frame around the entire figure
    plt.gcf().patch.set_visible(False)  # Remove the frame around the whole figure

    # Set the main title for the entire figure
    plt.suptitle(title, fontsize=14, fontweight='bold')

    # Plot histograms with KDE for each column
    for i, col in enumerate(df.columns):
        ax = plt.subplot(num_rows, num_cols, i + 1)  # Create a subplot for each column
        sns.histplot(df[col].dropna(), kde=True, ax=ax)  # Plot histogram with KDE
        ax.set_title(f'Data Distribution for {col}', fontsize=9, fontweight='bold')  # Set subplot title
        ax.set_xlabel(col, fontsize=8)  # Set x-axis label
        ax.set_ylabel('Frequency', fontsize=8)  # Set y-axis label
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')  # Rotate x-axis labels for better readability
        ax.tick_params(axis='x', labelsize=6)  # Adjust x-axis tick label font size
        ax.tick_params(axis='y', labelsize=6)  # Adjust y-axis tick label font size

    # Adjust layout to prevent overlap and ensure everything fits
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for the main title
    plt.subplots_adjust(left=0.1, right=0.95, hspace=0.85, wspace=0.45)  # Adjust spacing between rows and columns

    # Display the plot
    plt.show()
    
#Visualizes the distribution of values in each row of the DataFrame using histograms with KDE (Kernel Density Estimation).
def plot_dataDistribution(df: pd.DataFrame) -> None:
    """
    
    This function creates a grid of histograms, one for each row in the DataFrame. Each histogram represents the distribution
    of values across columns for that specific row. The grid is dynamically sized based on the number of rows, with up to
    3 plots per row.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.

    Returns:
        None
    """
    # Determine the number of rows and columns for the subplot grid
    num_samples: int = df.shape[0]  # Number of rows in the DataFrame
    num_cols: int = 3  # Number of columns in the grid
    num_rows: int = (num_samples + num_cols - 1) // num_cols  # Calculate number of rows needed

    # Set the figure size based on the number of rows
    plt.figure(figsize=(20, num_rows * 5))

    # Plot histograms with KDE for each row
    for i, (index, row) in enumerate(df.iterrows()):
        plt.subplot(num_rows, num_cols, i + 1)  # Create a subplot for each row
        sns.histplot(row.dropna(), kde=True)  # Plot histogram with KDE
        plt.title(f'Data Distribution for Row {index}', fontsize=10)  # Set subplot title
        plt.xlabel('Columns', fontsize=8)  # Set x-axis label
        plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
        plt.gca().set_xticklabels(row.dropna().index)  # Set x-axis tick labels to column names

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Display the plot
    plt.show()

#Checks for normal distribution of numeric features using Shapiro-Wilk, D'Agostino, and Anderson-Darling tests.
def normalDistribution(df: pd.DataFrame, title: str) -> pd.DataFrame:
    """  
    This function performs three statistical tests (Shapiro-Wilk, D'Agostino, and Anderson-Darling) on each numeric column
    in the DataFrame to assess whether the data follows a normal distribution. The results are displayed in a table and
    saved as an image file.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        title (str): The title of the results table.

    Returns:
        pd.DataFrame: A DataFrame containing the test results for each feature.
    """
    # Print the title and separator
    print(f"\n{title}\n" + "=" * len(title))

    # Dictionary to store results for each feature
    normal_test_results: list = []

    # Perform normality tests for each numeric column
    for column in df.columns:
        if df[column].dtype in [np.int64, np.float64]:  # Only check numeric columns
            # Perform Shapiro-Wilk test
            shapiro_stat, shapiro_p = shapiro(df[column].dropna())
            # Perform D'Agostino's K^2 test
            dagostino_stat, dagostino_p = normaltest(df[column].dropna())
            # Perform Anderson-Darling test
            anderson_result = anderson(df[column].dropna())

            # Collect results
            normal_test_results.append({
                'Feature': column,
                'Shapiro-Wilk p-value': round(shapiro_p, 4),
                'D\'Agostino p-value': round(dagostino_p, 4),
                'Anderson Statistic': round(anderson_result.statistic, 4)
            })

            # Print results for the current column
            print(f"Results for {column}:")
            print(f"  Shapiro-Wilk p-value: {shapiro_p}")
            print(f"  D'Agostino p-value: {dagostino_p}")
            print(f"  Anderson Statistic: {anderson_result.statistic}")
            print("-" * 40)

    # Convert results to a DataFrame for better visualization
    results_df: pd.DataFrame = pd.DataFrame(normal_test_results)

    # Create a plot with specific size for displaying the table
    fig, ax = plt.subplots(figsize=(14, 18))  # Adjusted size for better visibility of rows
    ax.axis('tight')
    ax.axis('off')  # Turn off the axes (we are showing a table, not a graph)

    # Set title for the table with some padding to avoid overlap
    ax.set_title(title, fontsize=14, pad=40,fontweight='bold')

    # Create a table with the comparison data and place it at the center
    tbl = Table(ax, bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)  # Adjust font size for readability
    tbl.auto_set_column_width([0, 1, 2, 3, 4])  # Automatically adjust column widths
    tbl.scale(1.2, 1.2)  # Scale the table to make it more visible

    # Add the rows to the table
    for (i, row) in enumerate(results_df.values):
        for j, value in enumerate(row):
            tbl.add_cell(i, j + 1, width=0.2, height=0.1, text=str(value), loc='center')

    # Add header
    for i, column in enumerate(results_df.columns):
        tbl.add_cell(-1, i + 1, width=0.2, height=0.1, text=column, loc='center', facecolor='#ADD8E6')

    # Add the table to the axes
    ax.add_table(tbl)

    # Adjust the layout to prevent overlap of title and table
    plt.subplots_adjust(top=0.85)  # Adjust the top margin to create space for the title

    # Save the table as an image file
    plt.savefig('results_table_with_title.png', bbox_inches='tight', dpi=300)

    # Show the plot (which contains the table)
    plt.show()

    # Return the results DataFrame
    return results_df

#Plots the distribution of classes in the DataFrame using a bar chart and a pie chart.
def plot_classDistribution(df: pd.DataFrame, title: str, class_column: str) -> None:
    """
    This function visualizes the distribution of classes (e.g., binary classification) in the DataFrame.
    It creates two subplots: a bar chart showing the count of each class and a pie chart showing the
    percentage distribution of each class.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        title (str): The title of the bar chart.
        class_column (str): The column name in the DataFrame that contains the class labels.

    Returns:
        None
    """
    # Check if the specified class column exists in the DataFrame
    if class_column not in df.columns:
        print(f"Error: Column '{class_column}' not found in the DataFrame.")
        return

    # Check if the DataFrame is empty
    if df.empty:
        print("Error: The DataFrame is empty.")
        return

    # Calculate the count of each class
    class_counts = df[class_column].value_counts().sort_index()

    # Define class labels and colors for the plots
    class_labels: list = ['Class 0', 'Class 1']
    colors: list = ['skyblue', 'salmon']  # Consistent colors for both charts

    try:
        # Create a figure with two subplots (bar chart and pie chart)
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))

        # Bar Chart
        bars = ax[0].bar(class_counts.index, class_counts.values, color=colors, alpha=0.85)
        ax[0].set_title(title, fontsize=16, weight='bold')
        ax[0].set_xlabel('Class', fontsize=14)
        ax[0].set_ylabel('Count', fontsize=14)
        ax[0].set_xticks(class_counts.index)
        ax[0].set_xticklabels(class_labels, fontsize=12)
        ax[0].grid(axis='y', linestyle='--', alpha=0.6)

        # Add count labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax[0].text(
                bar.get_x() + bar.get_width() / 2, height + 0.2, f'{int(height)}',
                ha='center', va='bottom', fontsize=12
            )

        # Pie Chart
        ax[1].pie(class_counts, labels=class_labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax[1].set_title('Class Distribution (Pie Chart)', fontsize=16, weight='bold')

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Display the plot
        plt.show()

    except Exception as e:
        # Handle any exceptions that occur during plotting
        print(f"An error occurred: {e}")

#Plots feature distributions using either boxplots or histograms.
def plot_feature_distributions(df: pd.DataFrame, numeric_columns: list, title: str, plot_type: str = 'boxplot') -> None:
    """   
    This function visualizes the distribution of numeric features in the DataFrame using either boxplots or histograms.
    Boxplots are useful for identifying outliers and understanding the spread of the data, while histograms provide
    insights into the frequency distribution of the data.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        numeric_columns (list): A list of column names (strings) representing numeric features.
        title (str): The title of the plot.
        plot_type (str): The type of plot to generate. Options are 'boxplot' or 'histogram'. Default is 'boxplot'.

    Returns:
        None

    Raises:
        ValueError: If an invalid `plot_type` is provided.
    """
    if plot_type == 'boxplot':
        # Create a figure for the boxplot
        plt.figure(figsize=(14, 10))

        # Define properties for the outliers
        flierprops = dict(marker='o', color='red', alpha=0.6, markersize=8, markeredgecolor='black')

        # Create the boxplot with modified flier properties
        sns.boxplot(data=df[numeric_columns], palette="Set2", flierprops=flierprops)

        # Set title and labels
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xticks(rotation=35, fontsize=10)
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Values', fontsize=12)

        # Set y-axis ticks with a step of 10
        y_min, y_max = plt.ylim()  # Get current y-axis limits
        plt.yticks(np.arange(0, y_max + 10, 10), fontsize=10)  # Set y-ticks with a step of 10

        # Add grid
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Adjust layout
        plt.tight_layout()

        # Display the plot
        plt.show()

    elif plot_type == 'histogram':
        # Use Seaborn style for better aesthetics
        sns.set(style="whitegrid")

        # Initialize the color palette
        palette = sns.color_palette("husl", len(numeric_columns))

        # Determine the number of rows and columns for the subplot grid
        num_plots = len(numeric_columns)
        num_cols = 4  # Number of columns in the grid
        num_rows = int(np.ceil(num_plots / num_cols))  # Calculate number of rows needed

        # Create subplots with specified size
        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(16, num_rows * 4))
        axes = axes.flatten()  # Flatten the axes array for easy indexing

        # Iterate over all numeric columns and plot each on a separate subplot
        for i, col in enumerate(numeric_columns):
            ax = axes[i]
            df[col].hist(bins=20, color=palette[i], edgecolor='black', ax=ax)
            ax.set_title(col, fontsize=10, fontweight='bold')
            ax.tick_params(axis='both', which='major', labelsize=10)

        # Remove any unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        # Set the main title for the entire figure
        plt.suptitle(title, fontsize=16, fontweight='bold')

        # Adjust layout to prevent overlap
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Display the plot
        plt.show()

    else:
        # Raise an error if an invalid plot type is provided
        raise ValueError("Invalid plot_type. Use 'boxplot' or 'histogram'.")

#Plots pairwise relationships between numeric features in the DataFrame using a pairplot.
def plot_pairwiseRelationships(df: pd.DataFrame, numeric_columns: list, title: str, class_column: str) -> None:
    """    
    This function creates a pairplot (scatterplot matrix) to visualize the relationships between all pairs of numeric
    features. The points are colored based on the `class_column` to highlight differences between groups. Diagonal plots
    show Kernel Density Estimates (KDE) for each feature.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        numeric_columns (list): A list of column names (strings) representing numeric features.
        title (str): The title of the pairplot.
        class_column (str): The column name in the DataFrame used to color the points (e.g., class labels).

    Returns:
        None
    """
    # Create the pairplot
    pair_grid = sns.pairplot(
        df,
        vars=numeric_columns,  # Use only the specified numeric columns
        hue=class_column,  # Color points based on the class_column
        palette='coolwarm',  # Color palette for the plot
        diag_kind='kde',  # Use KDE for diagonal plots
        plot_kws={'alpha': 0.8, 's': 50, 'edgecolor': 'k'},  # Customize scatterplot settings
        diag_kws={'linewidth': 1.5}  # Customize KDE plot settings
    )

    # Set a title for the entire figure
    pair_grid.fig.suptitle(
        title,
        y=1.08, fontsize=16, fontweight='bold'  # Adjust title position and font size
    )

    # Adjust x and y labels for better readability
    for ax in pair_grid.axes.flat:
        if ax:
            ax.set_xlabel(ax.get_xlabel(), rotation=25, fontsize=6, ha='right')  # Rotate x-axis labels
            ax.set_ylabel(ax.get_ylabel(), rotation=0, fontsize=6, labelpad=45)  # Adjust y-axis labels

            # Adjust tick parameters for better visibility
            ax.tick_params(axis='both', labelsize=6)

    # Adjust legend size and position
    legend = pair_grid._legend
    if legend:
        legend.set_bbox_to_anchor((1, 0.5))  # Shift legend outside the plot area
        legend.set_title(class_column, prop={'size': 10})  # Adjust legend title font size
        for text in legend.get_texts():
            text.set_fontsize(9)  # Adjust legend text font size

    # Adjust figure size and spacing
    pair_grid.fig.set_size_inches(16, 16)  # Set figure size
    pair_grid.fig.tight_layout(rect=[0, 0, 0.93, 1])  # Leave space for the legend
    pair_grid.fig.subplots_adjust(left=0.11, bottom=0.12, wspace=0.14, hspace=0.1)  # Adjust spacing between subplots

    # Display the plot
    plt.show()   
       
#Plots a heatmap of summary statistics for numeric features in the DataFrame.
def plot_statisticalFeatures(df: pd.DataFrame, title: str) -> None:
    """
    This function calculates summary statistics (mean, standard deviation, median, and p-value from a one-sample t-test)
    for each numeric column in the DataFrame. The results are displayed in a heatmap with black labels and table borders
    for better readability.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        title (str): The title of the heatmap.

    Returns:
        None
    """
    # Calculate summary statistics
    summary_stats = df.agg(['mean', 'std', 'median']).transpose()  # Calculate mean, std, and median
    summary_stats['P-Value'] = [stats.ttest_1samp(df[col], 0)[1] for col in df.columns]  # Calculate p-values
    summary_stats.reset_index(inplace=True)  # Reset index to make 'Feature' a column
    summary_stats.rename(columns={'index': 'Feature'}, inplace=True)  # Rename the index column to 'Feature'

    # Convert the results to a DataFrame
    summary_df = pd.DataFrame(summary_stats)

    # Plotting the table
    plt.figure(figsize=(10, 4))  # Set figure size
    sns.set(font_scale=1.2)  # Adjust font scale for better readability

    # Create the heatmap
    ax = sns.heatmap(
        summary_df.set_index('Feature'),  # Use 'Feature' as the index for the heatmap
        annot=True,  # Display values in each cell
        fmt=".2f",  # Format values to 2 decimal places
        cmap='Blues',  # Use a blue color map
        cbar=False,  # Disable the color bar
        linewidths=0.5,  # Add borders to cells
        linecolor='black'  # Set border color to black
    )

    # Set title and labels
    ax.set_title(title, fontsize=16, color='black',fontweight='bold')  # Set title in black
    plt.xticks(color='black', fontsize=12,fontweight='bold')  # Set x-axis labels in black
    plt.yticks(color='black', fontsize=9,fontweight='bold')  # Set y-axis labels in black

    # Save the plot as an image file
    plt.savefig('summary_statistics.png', bbox_inches='tight')

    # Display the plot
    plt.show()
      
#Plots the skewness of all numerical features in a single image with better adjustments for large datasets.
def plot_skewness_all_features(data: pd.DataFrame, title: str) -> None:
    """   
    This function creates a grid of histograms with Kernel Density Estimation (KDE) for each numeric column in the DataFrame.
    The color of each histogram is determined by the skewness of the data:
    - Green (#0da58d) for positive skewness.
    - Salmon for negative skewness.
    - Skyblue for zero skewness.

    Parameters:
        data (pd.DataFrame): The input DataFrame containing the data.
        title (str): The title of the overall plot.

    Returns:
        None
    """
    # Filter numerical columns
    numerical_columns = data.select_dtypes(include=['number']).columns
    num_features = len(numerical_columns)

    # Determine the grid size for subplots
    cols = 4  # Number of columns in the grid
    rows = math.ceil(num_features / cols)  # Calculate required rows

    # Create subplots
    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 5))  # Adjusted figsize
    axes = axes.flatten()  # Flatten to iterate easily

    # Plot histograms with KDE for each numeric column
    for i, column in enumerate(numerical_columns):
        # Extract column data and drop NaN values
        column_data = data[column].dropna()

        # Calculate skewness
        skewness_value = skew(column_data)

        # Select color based on skewness
        if skewness_value > 0:
            color = '#0da58d'  # Positive skewness
        elif skewness_value < 0:
            color = 'salmon'  # Negative skewness
        else:
            color = 'skyblue'  # Zero skewness

        # Plot histogram with KDE
        sns.histplot(
            column_data,
            kde=True,
            bins=20,
            color=color,
            alpha=0.7,
            ax=axes[i],
            line_kws={'color': 'black'}  # Set KDE line color to black
        )
        axes[i].set_title(f"{column}\n(Skewness: {skewness_value:.2f})", fontsize=10)
        axes[i].set_xlabel(column, fontsize=10)
        axes[i].set_ylabel('Frequency', fontsize=10)
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)

    # Remove empty subplots if any
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Add a legend for color explanation in the bottom-right corner
    legend_elements = [
        plt.Line2D([0], [0], color='#0da58d', lw=4, label='Positive Skewness'),
        plt.Line2D([0], [0], color='salmon', lw=4, label='Negative Skewness'),
        plt.Line2D([0], [0], color='skyblue', lw=4, label='Zero Skewness')
    ]

    fig.legend(
        handles=legend_elements,
        loc='lower right',  # Position the legend at the bottom-right
        bbox_to_anchor=(0.95, 0.05),  # Adjust position slightly inside the figure
        fontsize=9,
        frameon=True,
        fancybox=True,
        shadow=True,
        ncol=1  # Display legend items in a single column
    )

    # Add a main title for the entire figure
    fig.suptitle(
        title,
        fontsize=16,
        fontweight='bold',
        y=1.1  # Adjust vertical position of the title
    )

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)  # Add space for legend at the bottom

    # Display the plot
    plt.show()
        
#=========================================================================
#          Identify and remove outliers using IQR
#=========================================================================

def detectAndRemoveOutliersWithClassBalance(df, class_column='haveAlzheimer', threshold=1.5):
    """
    Detect and remove outliers using IQR, ensuring class balance.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        class_column (str): The column representing the class labels.
        threshold (float): Multiplier for the IQR range.
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed, preserving class balance.
    """
    df_cleaned = pd.DataFrame()
    classes = df[class_column].unique()
    
    for cls in classes:
        class_data = df[df[class_column] == cls]
        numeric_columns = class_data.select_dtypes(include=['int64', 'float64']).columns
        
        for column in numeric_columns:
            Q1 = class_data[column].quantile(0.25)
            Q3 = class_data[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            print(f"Class: {cls}, Column: {column}")
            print(f"  Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
            print(f"  Lower Bound: {lower_bound:.2f}, Upper Bound: {upper_bound:.2f}")
            
            # Keep rows within bounds
            class_data = class_data[(class_data[column] >= lower_bound) & (class_data[column] <= upper_bound)]
        
        df_cleaned = pd.concat([df_cleaned, class_data])
    
    return df_cleaned

#=========================================================================
#       Load and Prepare Dataset
#=========================================================================
# Define the column name for the target variable (class labels)
class_column_name = 'haveAlzheimer'

# Load dataset from CSV
df = pd.read_csv('dataset.csv')

# Drop unnecessary columns (e.g., 'idUser' which is not needed for analysis)
df = df.drop('idUser', axis=1)

# Map and rename columns for better readability
column_mapping = {
    'فرم اطلاعات شخصی_point': 'form_personal_info_point',
    'آزمون مختصر وضعیت شناختی-MMSE_point': 'MMSE_point',
    'نسخه فارسی ابزا ر غربالگری دمانس درسطح جامعه (CSI-D)_point': 'CSI-D_point',
    'فرم کوتاه ارزیابی وضعیت تغذیه(MNA-SF)_point': 'MNA-SF_point',
    'فرم GDS-15_point': 'GDS-15_point',
    'فرم GAD-7_point': 'GAD-7_point',
    'فعالیت های روزانه زندگی زندگی با کمک وسایل (IADL)_point': 'IADL_point',
    'فعالیت های روزانه زندگی (ADL)_point': 'ADL_point',
    'مقياس اصلاح شده كارآمدي در راه رفتن_point': 'walking_efficiency_point',
    'فرم اطلاعات شخصی_label': 'form_personal_info_label',
    'آزمون مختصر وضعیت شناختی-MMSE_label': 'MMSE_label',
    'نسخه فارسی ابزا ر غربالگری دمانس درسطح جامعه (CSI-D)_label': 'CSI-D_label',
    'فرم کوتاه ارزیابی وضعیت تغذیه(MNA-SF)_label': 'MNA-SF_label',
    'فرم GDS-15_label': 'GDS-15_label',
    'فرم GAD-7_label': 'GAD-7_label',
    'فعالیت های روزانه زندگی زندگی با کمک وسایل (IADL)_label': 'IADL_label',
    'فعالیت های روزانه زندگی (ADL)_label': 'ADL_label',
    'مقياس اصلاح شده كارآمدي در راه رفتن_label': 'walking_efficiency_label',
    'marriage': 'marriage',
    'smoker': 'smoker',
    'age': 'age',
    'bmi': 'bmi',
    'haveAlzheimer': 'haveAlzheimer'
}

# Rename columns using the mapping dictionary
df.rename(columns=column_mapping, inplace=True)

# Backup original DataFrame for comparison (optional, for debugging or reference)
df_original = df.copy()

#========================================================================= 
#       Data Cleaning
#=========================================================================

# Remove duplicate rows from the dataset
df.drop_duplicates(inplace=True)

# Drop columns with '_label' suffix since they are not needed for analysis
df = df.drop(columns=[col for col in df.columns if '_label' in col])

#=========================================================================
#       Handle Missing Values with KNN Imputer
#=========================================================================

# Separate features (X) and target (y)
X = df.drop([class_column_name], axis=1)  # Features (all columns except the target)
y = df[class_column_name]  # Target variable

# Use KNN Imputer to handle missing values in the features
imputer = KNNImputer(n_neighbors=2)
X_imputed = imputer.fit_transform(X)

# Recreate DataFrame with imputed values
df = pd.DataFrame(X_imputed, columns=df.columns.drop([class_column_name]))
df[class_column_name] = y.values  # Add the target column back to the DataFrame

#========================================================================= 
#       Handle Outliers (Optional)
#=========================================================================

# Detect and remove outliers while preserving class balance
df = detectAndRemoveOutliersWithClassBalance(df, threshold=2.5)

#========================================================================= 
#       Check Normality of Data
#=========================================================================

# Perform normality test on the dataset and store the results before data augmentation
normal_test_results = normalDistribution(df, 'Normal Distribution before Data Augmentation: ')

#========================================================================= 
#       Plotting and Visualizations
#=========================================================================

# Select numeric columns (float64 and int64)
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

# Plot skewness distribution for all numerical features before augmentation
plot_skewness_all_features(df, 'Skewness Distribution of All Numerical Features before Data Augmentation')

# Plot Q-Q plots for dataset columns before augmentation
plotQQ(df, "Q-Q Plots for Dataset Columns before Data Augmentation")

# Plot T-test results to compare characteristics after data augmentation
plot_t_test_results(df, 'T-test to compare characteristics between people with Alzheimer\'s disease and those without before Data Augmentation', class_column=class_column_name)

# Plot correlation matrix before augmentation
plotCorrelationMatrix(df, "Correlation Matrix before Data Augmentation")

# Plot summary statistics before augmentation
plot_statisticalFeatures(df, 'Summary Statistics before Data Augmentation')


# Plot mean and standard deviation for each column before data augmentation
plot_mean_standardDeviation(df, 'Summary statistics (mean, std) for each column before Data Augmentation')

# Plot distribution for each column before augmentation using boxplots
plot_feature_distributions(df, numeric_columns, 'Feature Distributions (Boxplot) before Data Augmentation', plot_type='boxplot')

# Plot feature distributions using histograms before augmentation
plot_feature_distributions(df, numeric_columns, 'Feature Distributions (Histograms)', plot_type='histogram')


# Plot class distribution using bar and pie charts before augmentation
plot_classDistribution(df, 'Class Distribution (Bar Chart) before Data Augmentation', class_column=class_column_name)

# Plot data column distribution before augmentation
plot_dataColumnDistribution(df, 'Data Distribution before Data Augmentation')




# Plot pairwise relationships between features before augmentation
plot_pairwiseRelationships(df, numeric_columns, 'Pairwise Relationships (Initial Data) before Data Augmentation', class_column=class_column_name)

# Print summary statistics of the dataset
print(df.describe().transpose())
        
