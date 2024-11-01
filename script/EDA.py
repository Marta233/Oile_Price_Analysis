import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

class EDA:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        logging.info("EDA object created with DataFrame of shape: %s", df.shape)

    def missing_percentage(self):
        logging.info("Calculating missing percentage for each column.")
        # Calculate the percentage of missing values
        missing_percent = self.df.isnull().sum() / len(self.df) * 100

        # Create a DataFrame to display the results nicely
        missing_df = pd.DataFrame({
            'Column': self.df.columns,
            'Missing Percentage': missing_percent
        }).sort_values(by='Missing Percentage', ascending=False)

        logging.info("Missing percentage calculated.")
        return missing_df

    def data_types(self):
        logging.info("Retrieving data types for each column.")
        data_typs = self.df.dtypes
        # Create a DataFrame for data types
        types_df = pd.DataFrame({
            'Column': self.df.columns,
            'Data Type': data_typs
        }).sort_values(by='Data Type', ascending=False)

        logging.info("Data types retrieved.")
        return types_df

    def histogram_boxplot(self, feature, figsize=(12, 7), kde=False, bins=None):
        """
        Boxplot and histogram combined
        kde: whether to show the density curve (default False)
        bins: number of bins for histogram (default None)
        """
        logging.info("Creating boxplot and histogram for feature: %s", feature)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        f2, (ax_box2, ax_hist2) = plt.subplots(
            nrows=2,
            sharex=True,
            gridspec_kw={"height_ratios": (0.25, 0.75)},
            figsize=figsize,
        )

        # Boxplot
        sns.boxplot(
            data=self.df, x=feature, ax=ax_box2, showmeans=True, color="violet"
        )
        ax_box2.set_title(f"Boxplot of {feature}")

        # Histogram
        if bins:
            sns.histplot(data=self.df, x=feature, kde=kde, ax=ax_hist2, bins=bins, palette="winter")
        else:
            sns.histplot(data=self.df, x=feature, kde=kde, ax=ax_hist2)

        ax_hist2.axvline(self.df[feature].mean(), color="green", linestyle="--", label=f"Mean: {self.df[feature].mean():.2f}")
        ax_hist2.axvline(self.df[feature].median(), color="black", linestyle="-", label=f"Median: {self.df[feature].median():.2f}")
        ax_hist2.legend()
        ax_hist2.set_title(f"Histogram of {feature}")

        f2.suptitle(f"Boxplot and Histogram for {feature}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
        logging.info("Boxplot and histogram displayed.")

    def scatter_plot(self, x_feature, y_feature, hue=None, figsize=(12, 7)):
        """
        Scatter plot for two continuous features
        """
        logging.info("Creating scatter plot for features: %s vs %s", x_feature, y_feature)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        plt.figure(figsize=figsize)
        sns.scatterplot(data=self.df, x=x_feature, y=y_feature, hue=hue)
        plt.title(f"Scatter Plot: {x_feature} vs {y_feature}")
        plt.show()
        logging.info("Scatter plot displayed.")