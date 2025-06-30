import pandas as pd
import numpy as np
import os
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns

class EDA:
    def __init__(self, df_path, plot_path = None, df_dir = None):
        """
        Initiate EDA class from DataFrame path.
        """

        self.df_path = df_path
        self.plot_path = plot_path
        self.df_dir = df_dir
        if self.df_path:
            self.load_df()
    
    def load_df (self):
        """
        Load DataFrame and understand the structure of the dataset
            - Number of rows, columns, and data types 
        """

        # Calculate DataFrame relative path 
        rel_df_path = os.path.relpath(self.df_path, os.getcwd())

        if self.df_path:
            try:
                self.df_raw = pd.read_csv(self.df_path)
                if 'TransactionStartTime' in self.df_raw.columns:
                    self.df_raw['TransactionStartTime'] = pd.to_datetime(self.df_raw['TransactionStartTime'], 
                                                                        errors='coerce')

                print(f"DataFrame loaded successfully from {rel_df_path}")
                print("\nDataFrame head:")
                display (self.df_raw.head())
                
                print("\nDataFrame shape:")
                display(self.df_raw.shape)
                
                print("\nDataFrame columns:")
                display(self.df_raw.columns)
                
                print("\nDataFrame summary:")
                self.df_raw.info()
            
            except Exception as e:
                print(f"Error loading DataFrame from {rel_df_path}: {e}")
                self.df_raw = None
        
        elif self.df_path:
            print(f"Error: File not found at {rel_df_path}")
            self.df_raw = None
        
        else:
            print("No DataFrame path provided during initialisation.")
            self.df_raw = None
        return self.df_raw
    
    def classify_columns(self):
        """
        Classifies columns into numerical and categorical after dataframe is loaded/cleaned.
        """
        if self.df is not None:
            # Identify categorical elements with dtype number
            numcols_are_catcols = ['CountryCode', 'PricingStrategy', 'FraudResult']
            # Numerical columns excluding 'CountryCode' and 'PricingStrategy' if present
            num_candidates = self.df.select_dtypes(include='number').columns
            self.num_cols = [col for col in num_candidates if col not in numcols_are_catcols]
            
            # Categorical columns excluding near-unique IDs
            threshold = 0.035 
            cat_candidates = self.df.select_dtypes(include='object').columns
            self.cat_cols = [
                col for col in cat_candidates
                if self.df[col].nunique() / self.df.shape[0] < threshold]
            self.cat_cols.extend([col for col in numcols_are_catcols if col in self.df.columns])
            self.cat_cols = list(set(self.cat_cols))  # Remove duplicates if any
            
        else:
            self.num_cols, self.cat_cols = [], []

    def missing_values (self):
        """
        Identify missing values and determine appropriate imputation strategies.
        """
        if hasattr(self, 'df_raw') and self.df_raw is not None:
            # Calculate % of missing values per column 
            mis_values = self.df_raw.isna().sum()
            mis_values_perc = (mis_values/len(self.df_raw))*100
            
            if mis_values.sum() == 0:
                print ('There are no columns with missing values')
                
            else:
                # Print total missing values per column
                print('Missing values per column:\n', mis_values)
                # List columns with <= 5% missing values
                col_null_low = mis_values_perc[mis_values_perc <= 5].index.tolist()
                # List columns with >5% missing values
                col_null_high = mis_values_perc[mis_values_perc > 5].index.tolist()

                if col_null_low:
                    print('\nColumns with less than 5% missing values:', col_null_low)
                    self.df_raw.dropna(subset=col_null_low, inplace=True)
            
                if col_null_high:
                    print('\nColumns with more than 5% missing values:', col_null_high)
                    self.df_raw[col_null_high] = self.df_raw[col_null_high].interpolate(method='linear', 
                                                                                        limit_direction='both')
                    print(f"Interpolated missing data in columns: {col_null_high}")
            
            self.df = self.df_raw.copy()
            self.classify_columns()    
            
        else:
            print("DataFrame not loaded. Please check initialisation.")
            return None

    def summary_statistics(self):
        """
        Understand the central tendency, dispersion, and shape of the dataset’s distribution.
        Exclude the 'CountryCode' column from numerical summaries.
        """
        if hasattr(self, 'df') and self.df is not None and self.num_cols:
                print("Summary statistics for numerical features (excluding 'CountryCode' and PricingStrategy):")
                display(self.df[self.num_cols].describe())
        else:
            print("No numerical features available for summary.")


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
    
    def remove_outliers_iqr_zscore(self, column):
        """
        Function to adress outliers
        """
        #IQR method
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound_iqr = Q1 - 1.5 * IQR
        upper_bound_iqr = Q3 + 1.5 * IQR
        
        #Z-score method
        data = self.df[column]
        z_scores = np.abs((data - data.mean()) / data.std())
        threshold = 2 #95% of data
        
        #Combined IQR and Z-score conditions
        self.df_out = self.df[
            (self.df[column] >= lower_bound_iqr) & 
            (self.df[column] <= upper_bound_iqr) & 
            (z_scores <= threshold)]

        return self.df_out
    
    def visualise_distribution (self):
        """
        Visualise the distribution of numerical features
            - identify patterns, skewness, and potential outliers.
        """
        if hasattr(self, 'df') and self.df is not None and self.num_cols:
            print ("Visualising distribution of numerical features ...\n")
            for col in self.num_cols:
                plt.figure(figsize = (8, 4))
                sns.histplot(data = self.df, x = col, kde= True, 
                                bins=20, color = 'Red', edgecolor='black')
                plt.title(f"Distribution of {col}")
                plt.tight_layout()    
                
                #select plot directory and plot name to save plot
                plot_name = f"Distribution of {col}.png"
                self.save_plot (plot_name)
                    
                #show plot
                plt.show()
                #close plot to free up space
                plt.close()
        else:
            print("No numerical features available for visualisation.")        
    
    def visualise_value_count (self):
        """
        Visualise the frquency of categorical features  
            - Insights into the frequency and variability of categories
        """        
        if hasattr(self, 'df') and self.df is not None:
            # Filter columns with just one unique value (eg., "CountryCode", "CurrencyCode")
            valid_cat_cols = [col for col in self.cat_cols if self.df[col].nunique() > 1]
            if valid_cat_cols:
                print ("Visualising value count for categorical features ...\n")
                for col in valid_cat_cols:
                    plt.figure(figsize=(9, 6))
                    ax = sns.countplot(data=self.df, x=col, 
                                    order=self.df[col].value_counts().index, hue = col)
                    plt.title(f"Frequency Distribution of '{col}'")
                    plt.xticks(rotation=90, ha='right')
                    plt.grid()
                    for container in ax.containers:
                        ax.bar_label(container, rotation=0, padding=3)
                    plt.tight_layout()
                        
                    #select plot directory and plot name to save plot
                    plot_name = f"Frequency Distribution of '{col}'.png"
                    self.save_plot (plot_name)
                        
                    #show plot
                    plt.show()
                    #close plot to free up space
                    plt.close()
            else:
                print("No categorical features available for visualisation.")        
        else:
            print("DataFrame not loaded. Please check initialisation.")            
    
    def correlation_analysis(self):
        """
        To understand the relationship between numerical features.
        Display a heatmap showing correlation between numerical features.
        """
        if hasattr(self, 'df') and self.df is not None and self.num_cols:
            print ("Visualising correlations ...\n")
            corr = self.df[self.num_cols].corr()
            plt.figure(figsize=(7, 3))
            sns.heatmap(corr, annot=True, fmt='.2f')
            plt.title("Correlation Matrix of Numerical Features")
            plt.tight_layout()
                
            #select plot directory and plot name to save plot
            plot_name = 'Correlation Matrix of Numerical Features.png'
            self.save_plot(plot_name)
                
            #show plot
            plt.show()
            #close plot to free up space
            plt.close()
        else:
            print("No numerical features available for correlation analysis.")

    def detect_outliers(self):
        """
        Visualise outliers in numerical features using boxplots.
        """
        if hasattr(self, 'df') and self.df is not None and self.num_cols:
            print ("Visualising outliers ...\n")
            for col in self.num_cols:
                plt.figure(figsize=(8, 4))
                sns.boxplot(data=self.df, x=col)
                plt.title(f"Outlier Detection for {col}")
                plt.tight_layout()
                
                #select plot directory and plot name to save plot
                plot_name = f"Outlier Detection for {col}.png"
                self.save_plot(plot_name)

                plt.show()
                plt.close()

            print("\nVisualising after handling outlier ...")
            for col in self.num_cols:
                self.df_out = self.remove_outliers_iqr_zscore(col)

                plt.figure(figsize=(8, 4))
                sns.histplot(data=self.df_out, x=col, kde=True, 
                            bins=20, color='green', edgecolor='black')
                plt.title(f"Distribution of {col} after Outlier Removal")
                plt.tight_layout()
                
                #select plot directory and plot name to save plot
                plot_name = f"Post-Outlier Distribution of {col}.png"
                self.save_plot(plot_name)
                
                #show plot
                plt.show()
                #close plot to free up space
                plt.close()

        else:
            print("No numerical features available for outlier detection.")
        
    def save_df(self, filename = 'processed_data.csv'):
            """
            Save processed DataFrame
            """

            if not hasattr(self, 'df_out') or self.df_out is None:
                print("No processed DataFrame found. Please run preprocessing steps before saving.")
                return

            ##create output folder if it doesn't exist
            if not os.path.exists(self.df_dir):
                os.makedirs(self.df_dir)
            
            df_name = os.path.join(self.df_dir, filename)
            
            ##calculate the relative path
            current_directory = os.getcwd()
            relative_path = os.path.relpath(df_name, current_directory)
            
            ##save processed data to CSV
            self.df_out.to_csv(df_name, index=False)
            print(f'\nProcessed DataFrame saved to: {relative_path}')

            print('\nDataFrame Head:')
            out_head=self.df_out.head()
            display (out_head)
            
            print('\nDataFrame Info:')
            self.df_out.info()
            
            print('\nDataFrame Shape:')
            display(self.df_out.shape)
            
            print("\nDataFrame summary:")
            self.df_out.info()
            
            print('\nDataFrame Description:')
            display(self.df_out[self.num_cols].describe())