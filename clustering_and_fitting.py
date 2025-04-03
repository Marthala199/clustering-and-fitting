import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression

def plot_relational_plot(df):
    sns.set_style("darkgrid")
    plt.figure(figsize=(12, 7))
    
    scatter = sns.scatterplot(
        data=df, x='BALANCE', y='PURCHASES', 
        alpha=0.7, edgecolor='black', 
        palette='coolwarm', hue=df['PURCHASES'], s=100
    )
    
    plt.title('Balance vs Purchases', fontsize=16, fontweight='bold', color='darkblue')
    plt.xlabel('Balance ($)', fontsize=14, fontweight='bold', color='black')
    plt.ylabel('Purchases ($)', fontsize=14, fontweight='bold', color='black')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('relational_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_categorical_plot(df):
    sns.set_style("darkgrid")
    plt.figure(figsize=(12, 7))
    
    sns.histplot(df['BALANCE'], bins=50, kde=True, color='royalblue', edgecolor='black', alpha=0.7)
    
    plt.title('Distribution of Balance', fontsize=16, fontweight='bold', color='darkblue')
    plt.xlabel('Balance ($)', fontsize=14, fontweight='bold', color='black')
    plt.ylabel('Frequency', fontsize=14, fontweight='bold', color='black')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig('categorical_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_statistical_plot(df):
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    heatmap = sns.heatmap(
        corr, annot=True, fmt=".2f", cmap='coolwarm_r', 
        linewidths=0.5, linecolor='black', square=True, 
        cbar_kws={'shrink': 0.8}
    )
    
    plt.title('Correlation Heatmap', fontsize=16, fontweight='bold', color='darkblue')
    plt.savefig('statistical_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

def preprocessing(df):
    df = df.drop(columns=['CUST_ID'], errors='ignore')
    df.fillna(df.mean(), inplace=True)
    return df

def perform_clustering(df, col1, col2):
    features = [col1, col2, 'CASH_ADVANCE', 'CREDIT_LIMIT', 'PAYMENTS']
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    inertias = []
    silhouette_scores = []
    K_range = range(2, 8)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        if k > 1:
            silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    
    plt.figure(figsize=(12, 7))
    plt.plot(K_range, inertias, marker='o', linestyle='-', color='royalblue', linewidth=2)
    plt.title('Elbow Method for Optimal Clusters', fontsize=16, fontweight='bold', color='darkblue')
    plt.xlabel('Number of Clusters (K)', fontsize=14, fontweight='bold')
    plt.ylabel('Inertia', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('elbow_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    optimal_k = 3
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    return labels, (df[col1], df[col2])

def plot_clustered_data(labels, data):
    x, y = data
    plt.figure(figsize=(12, 7))
    plt.scatter(x, y, c=labels, cmap='viridis', alpha=0.7, edgecolors='black')
    plt.xlabel('Balance ($)', fontsize=14, fontweight='bold')
    plt.ylabel('Purchases ($)', fontsize=14, fontweight='bold')
    plt.title('Customer Clusters', fontsize=16, fontweight='bold', color='darkblue')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('clustering.png', dpi=300, bbox_inches='tight')
    plt.show()

def perform_fitting(df, col1, col2):
    X = df[[col1]]
    y = df[col2]
    model = LinearRegression()
    model.fit(X, y)
    x_range = np.linspace(X.min().values[0], X.max().values[0], 100).reshape(-1, 1)
    y_pred = model.predict(x_range)
    return (X.values.flatten(), y.values), x_range.flatten(), y_pred

def plot_fitted_data(data, x, y):
    X, y_true = data
    plt.figure(figsize=(12, 7))
    plt.scatter(X, y_true, alpha=0.7, color='royalblue', edgecolors='black', s=80)
    plt.plot(x, y, color='red', linewidth=3, linestyle='-')
    plt.xlabel('Balance ($)', fontsize=14, fontweight='bold')
    plt.ylabel('Payments ($)', fontsize=14, fontweight='bold')
    plt.title('Linear Regression Fit', fontsize=16, fontweight='bold', color='darkblue')
    plt.grid(True, linestyle='-', alpha=0.6)
    plt.savefig('fitting.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    df = pd.read_csv('data.csv')
    df = preprocessing(df)
    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)
    
    labels, clustering_data = perform_clustering(df, 'BALANCE', 'PURCHASES')
    plot_clustered_data(labels, clustering_data)
    
    fitting_data, x_range, y_pred = perform_fitting(df, 'BALANCE', 'PAYMENTS')
    plot_fitted_data(fitting_data, x_range, y_pred)
    
if __name__ == "__main__":
    main()

