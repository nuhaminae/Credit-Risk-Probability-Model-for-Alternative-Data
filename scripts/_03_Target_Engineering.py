import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

class RiskTargetBuilder:
    """
    Builds a proxy target variable for credit risk using RFM (Recency, Frequency, Monetary) clustering.
    Customers are segmented using K-Means, and the least engaged cluster is labeled as high-risk.
    """

    def __init__(self, df_path, snapshot_date=None, n_clusters=3, random_state=42):
        """
        Initialises the class.

        Args:
            df_path (str): The path to the dataset.
            snapshot_date (str or datetime, optional): Reference date for Recency. 
                                        If None, uses max transaction date + 1 day.
            n_clusters (int): Number of clusters for K-Means.
            random_state (int): Random seed for reproducibility.
        """
        self.df_path = df_path
        self.df = pd.read_csv(df_path)
        self.snapshot_date = snapshot_date
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.proxy_labels = None

    def compute_rfm(self):
        """
        Computes RFM metrics from transaction history.

        Recency: Days since last transaction from snapshot date
        Frequency: Number of transactions
        Monetary: Total spend

        Args:
            self.df (pd.DataFrame): Raw transaction data with TransactionStartTime, TransactionId, Amount.

        Returns:
            pd.DataFrame: RFM values per customer
        """
        print("Computing RFM metrics...")
        self.df["TransactionStartTime"] = pd.to_datetime(self.df["TransactionStartTime"], errors='coerce')

        if self.snapshot_date is None:
            self.snapshot_date = self.df["TransactionStartTime"].max() + pd.Timedelta(days=1)

        rfm = self.df.groupby("CustomerId").agg({
            "TransactionStartTime": lambda x: (self.snapshot_date - x.max()).days,
            "TransactionId": "count",
            "Amount": "sum"
        }).reset_index()

        rfm.columns = ["CustomerId", "Recency", "Frequency", "Monetary"]
        print("\nRFM table computed.")
        return rfm

    def scale_rfm(self, rfm_df):
        """
        Standardizes RFM features for clustering.

        Args:
            rfm_df (pd.DataFrame): RFM metrics.

        Returns:
            np.ndarray: Scaled RFM values
        """
        print("\nScaling RFM features...")
        scaler = StandardScaler()
        scaled = scaler.fit_transform(rfm_df[["Recency", "Frequency", "Monetary"]])
        print("\nRFM features scaled.")
        return scaled

    def cluster_customers(self, scaled_rfm):
        """
        Applies K-Means clustering to segment customers.

        Args:
            scaled_rfm (np.ndarray): Scaled RFM data.

        Returns:
            np.ndarray: Cluster labels
        """
        print(f"\nClustering customers into {self.n_clusters} segments...")
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        clusters = kmeans.fit_predict(scaled_rfm)
        print("\nClustering completed.")
        return clusters

    def assign_high_risk(self, rfm_df, clusters):
        """
        Assigns a binary high-risk label to customers based on cluster analysis.

        Args:
            rfm_df (pd.DataFrame): RFM metrics
            clusters (np.ndarray): Cluster labels

        Returns:
            pd.DataFrame: CustomerId + is_high_risk column
        """
        print("\nAssigning high-risk labels...")
        rfm_df["cluster"] = clusters

        risk_cluster = (
            rfm_df.groupby("cluster")[["Recency", "Frequency", "Monetary"]]
            .mean()
            .sort_values(["Frequency", "Monetary", "Recency"], ascending=[True, True, False])
            .index[0]
        )

        print(f"\nIdentified high-risk cluster: {risk_cluster}")
        rfm_df["is_high_risk"] = (rfm_df["cluster"] == risk_cluster).astype(int)
        print("\nRisk labels assigned.")
        return rfm_df[["CustomerId", "is_high_risk"]]

    def generate_target(self):
        """
        Full pipeline to generate proxy risk target from raw transaction data.

        Args:
            self.df (pd.DataFrame): Raw transaction data.

        Returns:
            pd.DataFrame: Proxy target with CustomerId + is_high_risk
        """
        rfm = self.compute_rfm()
        scaled = self.scale_rfm(rfm)
        clusters = self.cluster_customers(scaled)
        self.proxy_labels = self.assign_high_risk(rfm, clusters)
        print("\nProxy target variable ready.")
        return self.proxy_labels