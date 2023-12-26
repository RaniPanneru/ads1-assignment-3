# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 05:53:51 2023

@author: panne
"""

# Importing necessary libraries
from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram

def read_and_cleanse_data(filename):
    # Read and cleanse the data from a CSV file

    df = pd.read_csv(filename, skiprows=4)

    # Drop unnecessary columns
    cols_to_drop = ['Country Code', 'Indicator Code', 'Unnamed: 66']
    df = df.drop(cols_to_drop, axis=1)

    # Rename columns
    df = df.rename(columns={'Country Name': 'Country'})

    # Reshape the dataframe by melting
    df = df.melt(id_vars=['Country', 'Indicator Name'],
                 var_name='Year', value_name='Value')

    # Convert columns to appropriate types
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

    # Separate dataframes with years and countries as columns
    df_years = df.pivot_table(
        index=['Country', 'Indicator Name'], columns='Year', values='Value')
    df_countries = df.pivot_table(
        index=['Year', 'Indicator Name'], columns='Country', values='Value')

    # Clean the data
    df_years = df_years.dropna(how='all', axis=1)
    df_countries = df_countries.dropna(how='all', axis=1)

    return df_years, df_countries

def subset_data(df_years, countries, indicators):
    # Subset data based on selected countries and indicators

    years = list(range(1980, 2018))
    if isinstance(indicators, str):
        indicators = [indicators]

    df = df_years.loc[(countries, indicators), years]
    df = df.transpose()
    return df

def normalize_data(df):
    # Normalize the dataframe using StandardScaler

    scaler = StandardScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df_normalized

def kmeans_clustering(df, num_clusters):
    # Perform k-means clustering on the dataframe

    kmeans = KMeans(n_clusters=num_clusters, n_init=5, random_state=42)
    cluster_labels = kmeans.fit_predict(df)
    return cluster_labels

def scatter_plot(df, cluster_labels, cluster_centers):
    # Visualize the cluster centers and data points

    plt.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(df.iloc[:, 0], df.iloc[:, 1], c=cluster_labels, cmap='rainbow')
    ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=200, marker='h', c='black')
    ax.set_xlabel(df.columns[0], fontsize=12)
    ax.set_ylabel(df.columns[1], fontsize=12)
    ax.set_title("K-Means Clustering Results", fontsize=14)
    ax.grid(True)
    plt.colorbar(scatter)
    plt.show()

def cluster_summary(cluster_labels):
    # Print a summary of the number of data points in each cluster

    cluster_counts = np.bincount(cluster_labels)
    for i, count in enumerate(cluster_counts):
        print(f"Cluster {i+1}: {count} data points")

def filter_and_pivot_urbanization_data(filename, countries, indicators, start_year, end_year):
    # Read, filter, and pivot urbanization data based on specified parameters

    urbanization_data = pd.read_csv(filename, skiprows=4)

    cols_to_drop = ['Country Code', 'Indicator Code', 'Unnamed: 66']
    urbanization_data = urbanization_data.drop(cols_to_drop, axis=1)

    urbanization_data = urbanization_data.rename(columns={'Country Name': 'Country'})

    urbanization_data = urbanization_data[urbanization_data['Country'].isin(countries) &
                                          urbanization_data['Indicator Name'].isin(indicators)]

    urbanization_data = urbanization_data.melt(id_vars=['Country', 'Indicator Name'],
                                               var_name='Year', value_name='Value')

    urbanization_data['Year'] = pd.to_numeric(urbanization_data['Year'], errors='coerce')
    urbanization_data['Value'] = pd.to_numeric(urbanization_data['Value'], errors='coerce')

    urbanization_data = urbanization_data.pivot_table(index=['Country', 'Indicator Name'],
                                                      columns='Year', values='Value')

    urbanization_data = urbanization_data.loc[:, start_year:end_year]

    return urbanization_data

def exponential_growth_function(x, a, b):
    # Exponential growth function for curve fitting

    return a * np.exp(b * x)

def calculate_error_ranges(xdata, ydata, popt, pcov, alpha=0.05):
    # Calculate error ranges for curve fitting

    n = len(ydata)
    m = len(popt)
    df = max(0, n - m)
    tval = -1 * stats.t.ppf(alpha / 2, df)
    residuals = ydata - exponential_growth_function(xdata, *popt)
    stdev = np.sqrt(np.sum(residuals**2) / df)
    ci = tval * stdev * np.sqrt(1 + np.diag(pcov))
    return ci

def visualize_growth_rate(data, countries):
    # Visualize the growth rate for each country using a line plot

    fig, ax = plt.subplots()
    for i in range(data.shape[0]):
        ax.plot(np.arange(data.shape[1]), data.iloc[i],
                label=data.index.get_level_values('Country')[i])
    ax.set_xlabel('Year')
    ax.set_ylabel('Urbanization Growth Rate')
    ax.set_title(', '.join(indicators_of_interest))
    ax.legend(loc='best')
    plt.show()

def visualize_dendrogram(df_normalized):
    # Visualize hierarchical clustering dendrogram

    linkage_matrix = linkage(df_normalized, 'ward')

    plt.figure(figsize=(12, 8))
    dendrogram(linkage_matrix, labels=df_normalized.index, orientation='top', leaf_rotation=90, leaf_font_size=8)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Countries')
    plt.ylabel('Distance')
    plt.show()

def visualize_corr_histogram(df):
    # Visualize histogram of correlation values

    corr = df.corr()

    plt.figure(figsize=(10, 6))
    sns.histplot(corr.values.flatten(), bins=30, kde=True)
    plt.title('Histogram of Correlation Values')
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Frequency')
    plt.show()

def predict_future_values(urbanization_data, countries, indicators, start_year, end_year):
    # Predict future values and visualize growth rate

    data = filter_and_pivot_urbanization_data(urbanization_data, countries, indicators, start_year, end_year)

    growth_rate = np.zeros(data.shape)
    for i in range(data.shape[0]):
        popt, pcov = curve_fit(exponential_growth_function, np.arange(data.shape[1]), data.iloc[i])
        ci = calculate_error_ranges(np.arange(data.shape[1]), data.iloc[i], popt, pcov)
        growth_rate[i] = popt[1]

    visualize_growth_rate(data, countries)

if __name__ == '__main__':
    # Reading and cleansing the data
    df_years, df_countries = read_and_cleanse_data("C:\\Users\\panne\\Downloads\\worlddata.csv")

    # Subsetting the data for the indicators of interest and the selected countries
    indicators_of_interest = ['Urban population (% of total population)', 'Urban population growth (annual %)']
    selected_countries = ['India', 'United States', 'China', 'United Kingdom']
    df_subset = subset_data(df_years, selected_countries, indicators_of_interest)

    # Normalizing the data using standardization
    df_normalized = normalize_data(df_subset)

    # Performing k-means clustering
    num_clusters = 3
    cluster_labels = kmeans_clustering(df_normalized, num_clusters)
    cluster_centers = KMeans(n_clusters=num_clusters, random_state=42).fit(df_normalized).cluster_centers_

    # Visualizing the clustering results as a scatter plot
    scatter_plot(df_normalized, cluster_labels, cluster_centers)

    # Predicting future values and visualizing growth rate
    predict_future_values("C:\\Users\\panne\\Downloads\\worlddata.csv", selected_countries, ["Urban population (% of total population)"], 1960, 2019)

    # Printing a summary of the number of data points in each cluster
    cluster_summary(cluster_labels)

    # Visualizing hierarchical clustering dendrogram
    visualize_dendrogram(df_normalized)

    # Visualizing histogram of correlation values
    visualize_corr_histogram(df_subset)
