import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import os

class Visualizer:
    def __init__(self, results_dir: str = 'results', output_dir: str = 'visualizations'):
        self.results_dir = results_dir
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def plot_sentiment_by_president(self, model_name: str):
        """Generates a bar chart of mean sentiment scores for each president."""
        file_path = os.path.join(self.results_dir, f'sentiment_statistics_{model_name}.csv')
        df = pd.read_csv(file_path)
        
        #Focus on English articles for a consistent comparison
        df_en = df[df['Language'] == 'en']
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Mean_Sentiment_Score', y='Article_Title', data=df_en, palette='viridis')
        plt.title(f'Mean Sentiment Score by U.S. President (English Articles - {model_name.upper()})')
        plt.xlabel('Mean Sentiment Score')
        plt.ylabel('President')
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, f'sentiment_by_president_{model_name}.png')
        plt.savefig(output_path)
        print(f"Saved sentiment by president plot to {output_path}")

    def plot_correlation_map(self, model_name: str):
        """Generates a choropleth map of sentiment correlation by country."""
        file_path = os.path.join(self.results_dir, f'favorable_correlation_{model_name}.csv')
        df = pd.read_csv(file_path)
        
        import zipfile
        with zipfile.ZipFile("ne_110m_admin_0_countries.zip", 'r') as zip_ref:
            zip_ref.extractall("ne_110m_admin_0_countries")
        world = gpd.read_file("ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp")
        # Map language codes to country names that geopandas can recognize
        # This is a simplified mapping and may need to be adjusted
        country_mapping = {
            'pl': 'Poland',
            'it': 'Italy',
            'hu': 'Hungary',
            'de': 'Germany',
            'nl': 'Netherlands',
            'el': 'Greece',
            'sv': 'Sweden',
            'ko': 'South Korea',
            'th': 'Thailand',
            'tl': 'Philippines', # Filipino (Tagalog)
            'ja': 'Japan',
            'si': 'Sri Lanka',
            'bn': 'Bangladesh',
            'hi': 'India',
            'ms': 'Malaysia',
            'he': 'Israel',
            'tr': 'Turkey'
        }
        df['country'] = df['Language_Code'].map(country_mapping)
        
        merged = world.merge(df, how='left', left_on='ADMIN', right_on='country')
        
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        merged.plot(column='Mean_Sentiment', cmap='coolwarm', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True,
                    missing_kwds={"color": "lightgrey", "label": "Missing data"})
        ax.set_title(f'Correlation Between Mean Sentiment and U.S. Favorability ({model_name.upper()})')
        ax.set_axis_off()
        output_path = os.path.join(self.output_dir, f'correlation_map_{model_name}.png')
        plt.savefig(output_path)
        print(f"Saved correlation map to {output_path}")

    def plot_correlation_scatter(self, model_name: str):
        """Generates a scatter plot of sentiment vs. favorability."""
        file_path = os.path.join(self.results_dir, f'favorable_correlation_{model_name}.csv')
        df = pd.read_csv(file_path)
        
        plt.figure(figsize=(10, 6))
        sns.regplot(x='Favorable_View', y='Mean_Sentiment', data=df)
        plt.title(f'Mean Sentiment vs. U.S. Favorability ({model_name.upper()})')
        plt.xlabel('Favorable View of U.S. (%)')
        plt.ylabel('Mean Sentiment Score')
        plt.grid(True)
        output_path = os.path.join(self.output_dir, f'correlation_scatter_{model_name}.png')
        plt.savefig(output_path)
        print(f"Saved correlation scatter plot to {output_path}")

if __name__ == "__main__":
    vis = Visualizer()
    # Generate visualizations for the BERT model results
    vis.plot_sentiment_by_president('bert')
    vis.plot_correlation_map('bert')
    vis.plot_correlation_scatter('bert')
