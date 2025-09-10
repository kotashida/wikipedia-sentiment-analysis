import wikipediaapi
from transformers import pipeline, AutoModel, AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, List
import logging
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
import argparse
import os

from config import FAVORABLE_VIEWS, PRESIDENTS, LANGUAGES, MODELS

class WikipediaAnalyzer:
    def __init__(self, model_name: str, user_agent: str = 'WikiSentimentAnalysis/1.0 (username@email.com)'):
        self.wiki = wikipediaapi.Wikipedia(user_agent, 'en')
        self.model_name = model_name
        self.device = self._get_device()
        self.analyzer = self._get_analyzer()
        self.tokenizer = self._get_tokenizer()
        self.model = self._get_model()
        self.languages = LANGUAGES

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Using device: {self.device}")

    def _get_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def _get_analyzer(self):
        if self.model_name in ["bert", "xlm_roberta"]:
            return pipeline("sentiment-analysis", model=MODELS[self.model_name], tokenizer=MODELS[self.model_name], device=0 if self.device.type == "cuda" else -1)
        return None

    def _get_tokenizer(self):
        if self.model_name == "labse":
            return AutoTokenizer.from_pretrained(MODELS[self.model_name])
        return None

    def _get_model(self):
        if self.model_name == "labse":
            return AutoModel.from_pretrained(MODELS[self.model_name]).to(self.device)
        return None

    def fetch_article_texts(self, article_name: str) -> Dict[str, Optional[str]]:
        texts = {}
        page = self.wiki.page(article_name)
        if not page.exists():
            self.logger.error(f"en article '{article_name}' not found")
            texts['en'] = None
        else:
            texts['en'] = page.text
            self.logger.info(f"Found en article: {article_name}")

        for lang in self.languages[1:]:
            try:
                lang_page = page.langlinks.get(lang)
                if lang_page:
                    texts[lang] = lang_page.text
                    self.logger.info(f"Found {lang} article: {lang_page.title}")
                else:
                    self.logger.warning(f"No {lang} translation available")
                    texts[lang] = None
            except Exception as e:
                self.logger.error(f"Error fetching {lang} article: {str(e)}")
                texts[lang] = None
        return texts

    def analyze_chunk_sentiment(self, chunk: str) -> Optional[float]:
        try:
            if not chunk.strip():
                return None
            return self.analyzer(chunk)[0]['score']
        except Exception as e:
            self.logger.error(f"Error analyzing chunk: {str(e)}")
            return None

    def get_embedding(self, text: str, chunk_size: int = 512) -> Optional[np.ndarray]:
        try:
            if not text.strip():
                return None
            inputs = self.tokenizer(text[:chunk_size], return_tensors="pt", padding=True, truncation=True, max_length=chunk_size)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            return embeddings[0]
        except Exception as e:
            self.logger.error(f"Error getting embedding: {str(e)}")
            return None

    def analyze_sentiment(self, text: str, chunk_size: int = 512) -> Optional[Dict[str, float]]:
        if not text:
            return None
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        
        if self.model_name in ["bert", "xlm_roberta"]:
            with ThreadPoolExecutor() as executor:
                scores = list(filter(None, executor.map(self.analyze_chunk_sentiment, chunks)))
            if not scores:
                return None
            return {
                'average_score': float(np.mean(scores)),
                'median_score': float(np.median(scores)),
                'standard_deviation': float(np.std(scores))
            }
        elif self.model_name == "labse":
            embeddings = [self.get_embedding(chunk, chunk_size) for chunk in chunks if self.get_embedding(chunk, chunk_size) is not None]
            if not embeddings:
                return None
            similarities = cosine_similarity(embeddings)
            upper_tri_indices = np.triu_indices_from(similarities, k=1)
            if len(similarities[upper_tri_indices]) > 0:
                return {
                    'average_score': float(np.mean(similarities[upper_tri_indices])),
                    'median_score': float(np.median(similarities[upper_tri_indices])),
                    'standard_deviation': float(np.std(similarities[upper_tri_indices]))
                }
            else:
                return {
                    'average_score': 1.0,
                    'median_score': 1.0,
                    'standard_deviation': 0.0
                }
        return None

    def analyze_article_sentiments(self, article_name: str, chunk_size: int = 512) -> pd.DataFrame:
        self.logger.info(f"Starting sentiment analysis of article: {article_name}")
        texts = self.fetch_article_texts(article_name)
        results = {}
        for lang, text in texts.items():
            self.logger.info(f"Analyzing sentiment for language: {lang}")
            results[lang] = self.analyze_sentiment(text, chunk_size) if text else None
        
        report_data = []
        for language, data in results.items():
            if data:
                report_data.append({
                    'Article_Title': article_name,
                    'Language': language,
                    'Mean_Sentiment_Score': f"{data['average_score']:.3f}",
                    'Median_Sentiment_Score': f"{data['median_score']:.3f}",
                    'Standard_Deviation': f"{data['standard_deviation']:.3f}"
                })
            else:
                report_data.append({
                    'Article_Title': article_name,
                    'Language': language,
                    'Mean_Sentiment_Score': 'NA',
                    'Median_Sentiment_Score': 'NA',
                    'Standard_Deviation': 'NA'
                })
        return pd.DataFrame(report_data)

    def calculate_article_correlation(self, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        df = sentiment_df.copy()
        df['Favorable_View'] = df['Language'].map(lambda x: FAVORABLE_VIEWS.get(x, (np.nan, ''))[0])

        def calculate_correlations(group):
            clean_group = group.dropna(subset=['Mean_Sentiment_Score', 'Median_Sentiment_Score', 'Favorable_View'])
            clean_group = clean_group[
                np.isfinite(clean_group['Mean_Sentiment_Score'].astype(float)) & 
                np.isfinite(clean_group['Median_Sentiment_Score'].astype(float)) & 
                np.isfinite(clean_group['Favorable_View'])
            ]
            
            if len(clean_group) > 2:
                try:
                    correlations = {
                        'Mean_Sentiment_Correlation': round(stats.pearsonr(clean_group['Favorable_View'], clean_group['Mean_Sentiment_Score'].astype(float))[0], 3),
                        'Median_Sentiment_Correlation': round(stats.pearsonr(clean_group['Favorable_View'], clean_group['Median_Sentiment_Score'].astype(float))[0], 3),
                        'Total_Languages': len(clean_group)
                    }
                    return pd.Series(correlations)
                except Exception as e:
                    self.logger.error(f"Error calculating correlation: {e}")
                    return pd.Series({'Mean_Sentiment_Correlation': np.nan, 'Median_Sentiment_Correlation': np.nan, 'Total_Languages': len(clean_group)})
            else:
                return pd.Series({'Mean_Sentiment_Correlation': np.nan, 'Median_Sentiment_Correlation': np.nan, 'Total_Languages': len(clean_group)})
        
        correlation_results = df.groupby('Article_Title').apply(calculate_correlations).reset_index()
        unique_articles = df['Article_Title'].unique()
        correlation_results = correlation_results.set_index('Article_Title').loc[unique_articles].reset_index()
        return correlation_results

    def calculate_language_correlation(self, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        df = sentiment_df.copy()
        language_averages = df.groupby('Language').agg({
            'Mean_Sentiment_Score': lambda x: pd.to_numeric(x, errors='coerce').mean(),
            'Median_Sentiment_Score': lambda x: pd.to_numeric(x, errors='coerce').mean(),
            'Standard_Deviation': lambda x: pd.to_numeric(x, errors='coerce').mean()
        }).round(3)

        combined_data = []
        for lang_code in FAVORABLE_VIEWS.keys():
            if lang_code in language_averages.index:
                row = {
                    'Language_Code': lang_code,
                    'Language_Name': FAVORABLE_VIEWS[lang_code][1],
                    'Favorable_View': FAVORABLE_VIEWS[lang_code][0],
                    'Mean_Sentiment': language_averages.loc[lang_code, 'Mean_Sentiment_Score'],
                    'Median_Sentiment': language_averages.loc[lang_code, 'Median_Sentiment_Score'],
                    'Standard_Deviation': language_averages.loc[lang_code, 'Standard_Deviation']
                }
                combined_data.append(row)
        
        result_df = pd.DataFrame(combined_data)
        correlations = {
            'Mean_Sentiment': stats.pearsonr(result_df['Favorable_View'], result_df['Mean_Sentiment'])[0],
            'Median_Sentiment': stats.pearsonr(result_df['Favorable_View'], result_df['Median_Sentiment'])[0],
        }
        self.logger.info("\nCorrelations with US Favorable Views:")
        for metric, corr in correlations.items():
            self.logger.info(f"{metric}: {corr:.3f}")
        self.logger.info(f"\nMean Standard Deviation Across Languages: {result_df['Standard_Deviation'].mean():.3f}")
        return result_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Wikipedia Sentiment Analysis')
    parser.add_argument('--model', type=str, required=True, choices=['bert', 'xlm_roberta', 'labse'], help='Model to use for analysis')
    parser.add_argument('--analysis', type=str, required=True, choices=['sentiment', 'article_correlation', 'language_correlation'], help='Type of analysis to perform')
    parser.add_argument('--input_file', type=str, help='Input CSV file for correlation analysis')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save the results')
    args = parser.parse_args()

    analyzer = WikipediaAnalyzer(model_name=args.model)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.analysis == 'sentiment':
        all_sentiments = pd.DataFrame()
        for president in PRESIDENTS:
            sentiment_df = analyzer.analyze_article_sentiments(president)
            all_sentiments = pd.concat([all_sentiments, sentiment_df], ignore_index=True)
        output_file = os.path.join(args.output_dir, f'sentiment_statistics_{args.model}.csv')
        all_sentiments.to_csv(output_file, index=False)
        print(f"Sentiment analysis complete. Results saved to {output_file}")

    elif args.analysis == 'article_correlation':
        if not args.input_file:
            raise ValueError("Input file is required for article correlation analysis")
        sentiment_df = pd.read_csv(args.input_file)
        correlation_df = analyzer.calculate_article_correlation(sentiment_df)
        output_file = os.path.join(args.output_dir, f'article_favorable_correlation_{args.model}.csv')
        correlation_df.to_csv(output_file, index=False)
        print(f"Article correlation analysis complete. Results saved to {output_file}")

    elif args.analysis == 'language_correlation':
        if not args.input_file:
            raise ValueError("Input file is required for language correlation analysis")
        sentiment_df = pd.read_csv(args.input_file)
        correlation_df = analyzer.calculate_language_correlation(sentiment_df)
        output_file = os.path.join(args.output_dir, f'favorable_correlation_{args.model}.csv')
        correlation_df.to_csv(output_file, index=False)
        print(f"Language correlation analysis complete. Results saved to {output_file}")
