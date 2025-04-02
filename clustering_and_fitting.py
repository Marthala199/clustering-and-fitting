import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression

def plot_relational_plot(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x='BALANCE', y='PURCHASES', alpha=0.6, ax=ax)
    ax.set_title('Balance vs Purchases')
    plt.savefig('relational_plot.png')
    plt.show()
    return

def plot_categorical_plot(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['BALANCE'], bins=50, kde=True, ax=ax)
    ax.set_title('Distribution of Balance')
    plt.savefig('categorical_plot.png')
    plt.show()
    return

def plot_statistical_plot(df):
    fig, ax = plt.subplots(figsize=(12, 8))
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Heatmap')
    plt.savefig('statistical_plot.png')
    plt.show()
    return

def statistical_analysis(df, col: str):
    mean = df[col].mean()
    stddev = df[col].std()
    skew = df[col].skew()
    excess_kurtosis = df[col].kurtosis()
    return mean, stddev, skew, excess_kurtosis

def preprocessing(df):
    df = df.drop('CUST_ID', axis=1)
    df.fillna(df.mean(), inplace=True)
    print("Data Overview:")
    print(df.describe())
    print("\nFirst few rows:")
    print(df.head())
    print("\nCorrelation Matrix:")
    print(df.corr())
    return df

def writing(moments, col):
    print(f'\nFor the attribute {col}:')
    print(f'Mean = {moments[0]:.2f}, '
          f'Standard Deviation = {moments[1]:.2f}, '
          f'Skewness = {moments[2]:.2f}, and '
          f'Excess Kurtosis = {moments[3]:.2f}.')
    skew_direction = "right" if moments[2] > 0 else "left" if moments[2] < 0 else "no"
    kurtosis_type = "leptokurtic" if moments[3] > 0 else "platykurtic" if moments[3] < 0 else "mesokurtic"
    print(f'The data was {skew_direction} skewed and {kurtosis_type}.')
    return

def perform_clustering(df, col1, col2):
    features = [col1, col2, 'CASH_ADVANCE', 'CREDIT_LIMIT', 'PAYMENTS']
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    inertias = []
    silhouette_scores = []
    K_range = range(2, 8)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        if k > 1:
            score = silhouette_score(X_scaled, kmeans.labels_)
            silhouette_scores.append(score)
    
    def plot_elbow_method():
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(K_range, inertias, marker='o')
        ax.set_xlabel('Number of Clusters')
        ax.set_ylabel('Inertia')
        ax.set_title('Elbow Method')
        plt.savefig('elbow_plot.png')
        plt.show()
    
    plot_elbow_method()
    
    optimal_k = 3  # Determined from elbow plot
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    kmeans.fit(X_scaled)
    labels = kmeans.labels_
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    
    xkmeans = centroids[:, features.index(col1)]
    ykmeans = centroids[:, features.index(col2)]
    
    def one_silhouette_inertia():
        _score = silhouette_score(X_scaled, labels)
        _inertia = kmeans.inertia_
        return _score, _inertia
    
    score, inertia = one_silhouette_inertia()
    print(f'\nClustering Metrics - Silhouette Score: {score:.2f}, Inertia: {inertia:.2f}')
    
    return labels, (df[col1].values, df[col2].values), xkmeans, ykmeans, labels

def plot_clustered_data(labels, data, xkmeans, ykmeans, cenlabels):
    x, y = data
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(x, y, c=cenlabels, cmap='viridis', alpha=0.6)
    ax.scatter(xkmeans, ykmeans, c='red', s=200, marker='X', label='Cluster Centers')
    ax.set_xlabel('BALANCE')
    ax.set_ylabel('PURCHASES')
    ax.set_title('Customer Clusters')
    plt.legend()
    plt.savefig('clustering.png')
    plt.show()
    return

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
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X, y_true, alpha=0.6, label='Data')
    ax.plot(x, y, color='red', linewidth=2, label='Linear Fit')
    ax.set_xlabel('BALANCE')
    ax.set_ylabel('PAYMENTS')
    ax.set_title('Linear Regression Fit')
    plt.legend()
    plt.savefig('fitting.png')
    plt.show()
    return

def main():
    df = pd.read_csv('data.csv')
    df = preprocessing(df)
    col = 'BALANCE'
    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)
    moments = statistical_analysis(df, col)
    writing(moments, col)
    clustering_results = perform_clustering(df, 'BALANCE', 'PURCHASES')
    plot_clustered_data(*clustering_results)
    fitting_results = perform_fitting(df, 'BALANCE', 'PAYMENTS')
    plot_fitted_data(*fitting_results)
    return

if __name__ == '__main__':
    main()
