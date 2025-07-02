import warnings
warnings.simplefilter("ignore", category=FutureWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)
    import scorecardpy as sc
    from monotonic_binning.monotonic_woe_binning import Binning
    import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, classification_report
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
import importlib.util
import glob
import os
import joblib

class FraudDetectionPipeline:
    """
    A pipeline class for fraud detection that includes data loading, splitting,
    WOE transformation, variable filtering, model training, and evaluation.
    """
    def __init__(self, df_path, target = 'FraudResult', 
                df_dir = None, plot_path = None, mdl_dir = None):
        """
        Initialises the FraudDetectionPipeline with data path and target variable.

        Args:
            df_path (str): The path to the dataset.
            target (str, optional): The name of the target variable. Defaults to 'FraudResult'.
            df_dir (str, optional): The directory to save processed DataFrames. Defaults to None.
            plot_path (str, optional): The directory to save plots. Defaults to None.
            mdl_dir (str, optional): The directory to save trained models. Defaults to None.
        """
        self.df_path = df_path
        self.target = target
        self.df_dir = df_dir
        self.plot_path = plot_path
        self.mdl_dir = mdl_dir
        self.categorical_vars = ['ProviderId', 'ProductId', 
                                'ProductCategory', 'ChannelId', 'PricingStrategy']

        self.numeric_vars = ['Amount', 'Value', 'txn_hour', 'txn_day', 'txn_month', 'txn_year',
                            'total_txn_amount', 'avg_txn_amount', 'txn_count', 'std_txn_amount']

        self.drop_cols = ['CurrencyCode', 'CountryCode', 'TransactionId', 'TransactionStartTime',
                        'BatchId', 'AccountId', 'SubscriptionId','CustomerId']
        self.breaks = {}
        self.bins_adj = None

    @staticmethod
    def custom_split_df(df, y_col, ratio=0.7, seed=999):
        """
        Splits a DataFrame into training and testing sets, stratified by y_col.

        Args:
            df (pd.DataFrame): The full DataFrame.
            y_col (str): Target column name.
            ratio (float): Fraction of data to go into training set.
            seed (int): Random seed for reproducibility.
        Returns:
            train (pd.DataFrame), test (pd.DataFrame)
        """
        # Shuffle and split stratified by target variable
        train = (
            df.groupby(y_col, group_keys=False)
            .apply(lambda x: x.sample(frac=ratio, random_state=seed))
        )

        test = df.loc[~df.index.isin(train.index)]

        return train.reset_index(drop=True), test.reset_index(drop=True)

    def load_and_split_data(self):
        """
        Loads data, splits it into training and testing sets, and fills missing categorical values.
        """
        self.df = pd.read_csv(self.df_path)

        # Convert transaction timestamp
        self.df["TransactionStartTime"] = pd.to_datetime(self.df["TransactionStartTime"], errors='coerce')

        # Extract time features
        self.df["txn_hour"] = self.df["TransactionStartTime"].dt.hour
        self.df["txn_day"] = self.df["TransactionStartTime"].dt.day
        self.df["txn_month"] = self.df["TransactionStartTime"].dt.month
        self.df["txn_year"] = self.df["TransactionStartTime"].dt.year

        # --- Create Aggregate Features ---
        agg_features = (
            self.df.groupby("CustomerId")
            .agg(total_txn_amount=("Amount", "sum"),
                avg_txn_amount=("Amount", "mean"),
                txn_count=("Amount", "count"),
                std_txn_amount=("Amount", "std"))
            .reset_index()
        )

        # Merge back into main DataFrame
        self.df = self.df.merge(agg_features, on="CustomerId", how="left")
        
        # Drop unneeded columns after feature extraction
        self.df = self.df.drop(columns=self.drop_cols)

        self.train, self.test = self.custom_split_df(self.df, self.target, ratio=0.7, seed=999)

        # Fill missing categorical values
        for col in self.categorical_vars:
            self.df[col] = self.df[col].fillna('missing')
            self.train[col] = self.train[col].fillna('missing')

        print("Data loaded, split, aggregated, and categorical NAs filled.")


    def compute_monotonic_breaks(self):
        """
        Computes monotonic WOE breaks for numeric variables.
        """
        vars_numeric = self.train.drop([self.target] + self.categorical_vars, axis=1).columns
        #y = self.train[self.target]
        #bin_object = Binning(y, n_threshold=50, y_threshold=10, p_threshold=0.35, sign=False)
        bin_object = Binning(self.target, n_threshold=50, y_threshold=10, p_threshold=0.35, sign=False)

        for var in vars_numeric:
            bin_object.column = var
            bin_object.fit(self.train[[self.target, var]])
            self.breaks[var] = bin_object.bins[1:-1].tolist()
        print("\nNumeric WOE breaks computed.")

    def compute_categorical_breaks(self):
        """
        Computes WOE breaks for categorical variables using scorecardpy.
        """
        sc.woebin(self.train, y=self.target, x=self.categorical_vars, save_breaks_list='cat_breaks')
        py_file = max(glob.glob("cat_breaks_*.py"), key=os.path.getctime)
        spec = importlib.util.spec_from_file_location("cat_breaks_module", py_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.breaks.update(module.breaks_list)
        os.remove(py_file)
        if os.path.exists(py_file + 'c'): os.remove(py_file + 'c')
        print("\nCategorical WOE breaks computed and merged.")

    def apply_woe_transformation(self):
        """
        Applies WOE transformation to the entire dataset and the split train/test sets.
        """
        self.bins_adj = sc.woebin(self.df, self.target, breaks_list=self.breaks, positive='bad|0')
        self.train_woe = sc.woebin_ply(self.train, self.bins_adj)
        self.test_woe = sc.woebin_ply(self.test, self.bins_adj)
        print("\nWOE transformation applied.")

    def merge_and_clean(self):
        """
        Merges original train/test sets with their WOE transformed counterparts and cleans up columns.
        """
        self.train_final = self.train.merge(self.train_woe, how='left', left_index=True, right_index=True)
        self.test_final = self.test.merge(self.test_woe, how='left', left_index=True, right_index=True)
        self.train_final = self.train_final.drop(columns=f'{self.target}_y').rename(columns={f'{self.target}_x': 'vd'})
        self.test_final = self.test_final.drop(columns=f'{self.target}_y').rename(columns={f'{self.target}_x': 'vd'})
        self.train_final = self.train_final.drop(columns=self.categorical_vars)
        self.test_final = self.test_final.drop(columns=self.categorical_vars)
        print("\nFinal dataset merged and cleaned.")   

    def run_iv_analysis(self):
        """
        Performs Information Value (IV) analysis on the final training dataset.

        Returns:
            pd.DataFrame: DataFrame containing the IV analysis results.
        """
        iv_table = sc.iv(self.train_final, y='vd')
        print("IV Analysis:\n")
        print(iv_table)
        return iv_table

    def filter_variables(self):
        """
        Filters variables based on Information Value and missing rate using scorecardpy.
        """
        self.train_final = sc.var_filter(self.train_final, y='vd')
        self.test_final = self.test_final[self.train_final.columns]
        print("\nVariables filtered based on IV and missing rate.")

    def save_plot(self, plot_name):
        """
        Saves the current plot to the designated plot folder.
        
        Args:
            plot_name (str): The name of the plot file (including extension, e.g., '.png').
        """
        #create the directory if it doesn't exist
        if not os.path.exists(self.plot_path):
            os.makedirs(self.plot_path)
            
        plot_path = os.path.join(self.plot_path, plot_name)
            
        #calculate the relative path
        relative_plot_path = os.path.relpath(plot_path, os.getcwd())
            
        try:
            plt.savefig(plot_path)
            print(f'\nPlot saved to {relative_plot_path}')
        except Exception as e:
            print(f'\nError saving plot: {e}')

    def train_model(self):
        """
        Trains a Logistic Regression model on the filtered training data.
        """
        self.y_train = self.train_final['vd']
        self.X_train = self.train_final.drop(columns='vd')
        self.y_test = self.test_final['vd']
        self.X_test = self.test_final.drop(columns='vd')

        # Impute missing values in training and testing data
        self.X_train = self.X_train.fillna(self.X_train.median(numeric_only=True))
        self.X_test = self.X_test.fillna(self.X_train.median(numeric_only=True))
        
        self.lr = LogisticRegression(penalty='l1', C=0.9, solver='liblinear')
        self.lr.fit(self.X_train, self.y_train)
        print("Model trained.\n")
        print("Coefficients:")
        print(self.lr.coef_)

    def evaluate_model(self):
        """
        Evaluates the trained Logistic Regression model on the training and testing data.
        Prints performance metrics and displays a confusion matrix.
        """
        train_pred = self.lr.predict_proba(self.X_train)[:, 1]
        test_pred = self.lr.predict_proba(self.X_test)[:, 1]

        print("\nTrain Performance:")
        sc.perf_eva(self.y_train, train_pred, title='Train')
        print("Test Performance:")
        sc.perf_eva(self.y_test, test_pred, title='Test')

        predictions = self.lr.predict(self.X_test)
        print("Accuracy:", accuracy_score(self.y_test, predictions))
        print("AUC Score:", roc_auc_score(self.y_test, predictions))
        print("\nClassification Report:")
        print(classification_report(self.y_test, predictions))

        conf_matrix = confusion_matrix(self.y_test, predictions)
        sns.heatmap(data=conf_matrix, annot=True, fmt='.0f', cmap='magma', linewidth=0.7)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix - Logistic Regression')
        plt.tight_layout()   

        #select plot directory and plot name to save plot
        plot_name = f"Confusion Matrix - Logistic Regression.png"
        self.save_plot (plot_name)

        #show plot
        plt.show()
        #close plot to free up space
        plt.close()

    def predict(self, new_df):
        """
        Applies the trained model to a new DataFrame to generate predictions.

        Args:
            new_df (pd.DataFrame): The new DataFrame to make predictions on.

        Returns:
            numpy.ndarray: An array of predicted probabilities for the positive class.
        """
        new_df = new_df.drop(columns=self.drop_cols)

        # Fill missing categorical values (same cleanup as training)
        for col in self.categorical_vars:
            if col in new_df.columns:
                new_df[col] = new_df[col].fillna('missing')

        # Apply WOE transformation
        new_woe = sc.woebin_ply(new_df, self.bins_adj)

        # Match trained feature columns
        new_woe = new_woe[self.X_train.columns]

        return self.lr.predict_proba(new_woe)[:, 1]
    

    def save_processed_data(self, filename = 'feature_engineering_data.csv'):
        """
        Saves the processed DataFrame to the specified directory.

        Args:
            filename (str, optional): The name of the output file. 
                                    Defaults to 'feature_engineering_data.csv'.
        """

        if not hasattr(self, 'df') or self.df is None:
            print("No processed DataFrame available to save.")
            return

        # Create output folder if it doesn't exist
        if not os.path.exists(self.df_dir):
            os.makedirs(self.df_dir)
            
        df_name = os.path.join(self.df_dir, filename)
            
        # Calculate the relative path
        current_directory = os.getcwd()
        relative_path = os.path.relpath(df_name, current_directory)
            
        # Save processed data to CSV
        self.df.to_csv(df_name, index=False)
        print(f'\nProcessed DataFrame saved to: {relative_path}')

        print('\nDataFrame Head:')
        out_head=self.df.head()
        display (out_head)
        
        print('\nDataFrame Description:')
        display(self.df.describe())

    def save_model(self, filename="fraud_model.pkl"):
        """
        Saves the trained logistic regression model to a file.

        Args:
            filename (str): Model file name to save the model as. Defaults to 'fraud_model.pkl'.
        """
        if self.lr:
            # Create model directory if it doesn't exist
            if not os.path.exists(self.mdl_dir):
                os.makedirs(self.mdl_dir)
            filepath = os.path.join(self.mdl_dir, filename)
            joblib.dump(self.lr, filepath)
            
            # Calculate the relative path
            relative_path = os.path.relpath(filepath, os.getcwd())
            print(f"\nFraud detection model saved to: {relative_path}")
        else:
            print("\nNo model trained yet.")

    def load_model(self, filename = "fraud_model.pkl"):
        """
        Loads a pre-trained logistic regression model from the specified model directory.

        Args:
            filename (str): Model filename to load. Defaults to 'fraud_model.pkl'.
        """
        if not self.mdl_dir:
            print(f"\nModel directory {self.mdl_dir} is not set.")
            return

        filepath = os.path.join(self.mdl_dir, filename)

        if os.path.exists(filepath):
            self.lr = joblib.load(filepath)
            relative_path = os.path.relpath(filepath, os.getcwd())
            print(f"\nModel loaded from: {relative_path}")
        else:
            print(f"\nModel file not found at: {relative_path}")
