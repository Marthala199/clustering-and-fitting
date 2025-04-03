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

    sns.scatterplot(
        data=df, x='BALANCE', y='PURCHASES',
        alpha=0.7, edgecolor='black',
        palette='coolwarm', hue='PURCHASES', s=100
    )

    plt.title('Balance vs Purchases', fontsize=16,
             fontweight='bold', color='darkblue')  # Fixed indentation
    plt.xlabel('Balance ($)', fontsize=14, fontweight='bold', color='black')
    plt.ylabel('Purchases ($)', fontsize=14, fontweight='bold', color='black')

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('relational_plot.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_categorical_plot(df):
    sns.set_style("darkgrid")
    plt.figure(figsize=(12, 7))

    sns.histplot(df['BALANCE'], bins=50, kde=True,
                 color='royalblue', edgecolor='black', alpha=0.7)

    plt.title('Distribution of Balance', fontsize=16,
             fontweight='bold', color='darkblue')
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

    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap='coolwarm_r',
        linewidths=0.5, linecolor='black', square=True,
        cbar_kws={'shrink': 0.8}
    )

    plt.title('Correlation Heatmap', fontsize=16,
             fontweight='bold', color='darkblue')
    plt.savefig('statistical_plot.png', dpi=300, bbox_inches='tight')
    plt.show()


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
    skew_dir = ("right" if moments[2] > 0 
                else "left" if moments[2] < 0 
                else "no")
    kurt_type = ("leptokurtic" if moments[3] > 0 
                 else "platykurtic" if moments[3] < 0 
                 else "mesokurtic")
    print(f'The data was {skew_dir} skewed and {kurt_type}.')


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

    def plot_elbow_method(K_range, inertias):
        sns.set_style("whitegrid")
        plt.figure(figsize=(12, 7))

        plt.plot(K_range, inertias, marker='o', markersize=8, linestyle='-',
                 color='royalblue', linewidth=2, label="Inertia")

        elbow_idx = 2
        plt.scatter(K_range[elbow_idx], inertias[elbow_idx], color='red',
                    s=120, edgecolors='black', label="Elbow Point", zorder=3)

        plt.title('Elbow Method for Optimal Clusters', fontsize=16,
                 fontweight='bold', color='darkblue')
        plt.xlabel('Number of Clusters (K)', fontsize=14, fontweight='bold')
        plt.ylabel('Inertia (Within-Cluster SSE)', 
                   fontsize=14, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.savefig('elbow_plot.png', dpi=300, bbox_inches='tight')
        plt.show()

    plot_elbow_method(K_range, inertias)

    optimal_k = 3
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    kmeans.fit(X_scaled)
    labels = kmeans.labels_
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)

    xkmeans = centroids[:, features.index(col1)]
    ykmeans = centroids[:, features.index(col2)]

    def get_silhouette_inertia():
        _score = silhouette_score(X_scaled, labels)
        _inertia = kmeans.inertia_
        return _score, _inertia

    score, inertia = get_silhouette_inertia()
    print(f'\nClustering Metrics - Silhouette Score: {score:.2f}, '
          f'Inertia: {inertia:.2f}')

    return labels, (df[col1].values, df[col2].values), xkmeans, ykmeans


def plot_clustered_data(labels, data, xkmeans, ykmeans):
    x, y = data

    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 7))

    scatter = plt.scatter(x, y, c=labels, cmap='viridis',
                          alpha=0.7, edgecolors='black')

    plt.scatter(xkmeans, ykmeans, c='red', s=250, marker='X',
                edgecolors='black', label='Cluster Centers', linewidth=2)

    plt.xlabel('Balance ($)', fontsize=14, fontweight='bold')
    plt.ylabel('Purchases ($)', fontsize=14, fontweight='bold')
    plt.title('Customer Clusters', fontsize=16,
             fontweight='bold', color='darkblue')
    plt.grid(True, linestyle='--', alpha=0.6)

    cbar = plt.colorbar(scatter)
    cbar.set_label('Cluster Labels', fontsize=12, fontweight='bold')
    plt.legend()
    plt.savefig('clustering.png', dpi=300, bbox_inches='tight')
    plt.show()


def perform_fitting(df, col1, col2):
    X = df[[col1]]
    y = df[col2]
    model = LinearRegression()
    model.fit(X, y)
    x_min, x_max = X.min().values[0], X.max().values[0]
    x_range = np.linspace(x_min, x_max, 100).reshape(-1, 1)
    y_pred = model.predict(x_range)
    return (X.values.flatten(), y.values), x_range.flatten(), y_pred


def plot_fitted_data(data, x, y):
    X, y_true = data

    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 7))

    plt.scatter(X, y_true, alpha=0.7, color='royalblue',
                edgecolors='black', s=80, label='Data')
    plt.plot(x, y, color='red', linewidth=3, linestyle='-', label='Linear Fit')

    plt.xlabel('Balance ($)', fontsize=14, fontweight='bold')
    plt.ylabel('Payments ($)', fontsize=14, fontweight='bold')
    plt.title('Linear Regression Fit', fontsize=16,
             fontweight='bold', color='darkblue')
    plt.grid(True, linestyle='-', alpha=0.6)
    plt.legend(fontsize=12)
    plt.savefig('fitting.png', dpi=300, bbox_inches='tight')
    plt.show()


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


if __name__ == "__main__":
    main()

