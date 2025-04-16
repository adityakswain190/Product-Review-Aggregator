# app.py
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import pandas as pd
import re
import nltk
import os
import requests
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import plotly.express as px
import plotly.graph_objects as go
import json
from dotenv import load_dotenv
import time
from urllib.parse import urlparse, parse_qs

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "amazon-review-analyzer-secret")

# Download necessary NLTK resources
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')

# Oxylabs API credentials
OXYLABS_USERNAME = os.getenv("OXYLABS_USERNAME")
OXYLABS_PASSWORD = os.getenv("OXYLABS_PASSWORD")

# Check if credentials are available, otherwise use fallbacks
if not OXYLABS_USERNAME or not OXYLABS_PASSWORD:
    print("WARNING: Using fallback credentials. Set OXYLABS_USERNAME and OXYLABS_PASSWORD in your .env file.")
    OXYLABS_USERNAME = "Tyrant_1uLvI" 
    OXYLABS_PASSWORD = "0000_Tyrantis"

class ReviewAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text):
        """Clean and preprocess text"""
        text = re.sub(r'[^\w\s]', '', str(text).lower())
        tokens = word_tokenize(text)
        return " ".join([w for w in tokens if w not in self.stop_words])
        
    def analyze_sentiment(self, reviews):
        """Analyze sentiment of reviews"""
        sentiments = []
        for review in reviews:
            score = self.sia.polarity_scores(review)
            sentiments.append(score)
        
        # Convert to DataFrame for easier analysis
        sentiment_df = pd.DataFrame(sentiments)
        return sentiment_df
    
    def extract_topics(self, reviews, num_topics=5, num_words=10):
        """Extract main topics from reviews using LDA"""
        # Vectorize reviews
        vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
        dtm = vectorizer.fit_transform(reviews)
        
        # Create and fit LDA model
        lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda.fit(dtm)
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Collect topics
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[:-num_words-1:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics.append({
                "id": topic_idx,
                "words": top_words
            })
        
        return topics
    
    def extract_product_features(self, reviews, top_n=10):
        """Extract product features mentioned in reviews using TF-IDF"""
        # Use TF-IDF to find important terms
        tfidf_vectorizer = TfidfVectorizer(max_df=0.9, min_df=3, stop_words='english', ngram_range=(1, 2))
        tfidf_matrix = tfidf_vectorizer.fit_transform(reviews)
        
        # Get feature names
        feature_names = tfidf_vectorizer.get_feature_names_out()
        
        # Calculate importance score (sum of TF-IDF values across all documents)
        importance_scores = tfidf_matrix.sum(axis=0).A1
        
        # Create a dictionary of feature name to importance score
        feature_scores = {feature_names[i]: importance_scores[i] for i in range(len(feature_names))}
        
        # Get top features by importance score
        top_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        return top_features
    
    def analyze_reviews(self, reviews, product_info=None):
        """Main analysis function"""
        if not reviews:
            return {
                "recommendation": "Insufficient Data",
                "sentiment": {
                    "average": 0,
                    "positive_pct": 0,
                    "negative_pct": 0,
                    "neutral_pct": 0
                },
                "topics": [],
                "sample_positives": [],
                "sample_negatives": [],
                "product_info": product_info or {}
            }
            
        # Clean reviews
        cleaned_reviews = [self.clean_text(review) for review in reviews]
        
        # Analyze sentiment
        sentiment_df = self.analyze_sentiment(reviews)
        
        # Extract topics
        topics = self.extract_topics(cleaned_reviews)
        
        # Extract important product features
        product_features = self.extract_product_features(cleaned_reviews)
        
        # Calculate metrics
        avg_sentiment = sentiment_df['compound'].mean()
        positive_pct = (sentiment_df['compound'] > 0.05).mean() * 100
        negative_pct = (sentiment_df['compound'] < -0.05).mean() * 100
        neutral_pct = ((sentiment_df['compound'] >= -0.05) & (sentiment_df['compound'] <= 0.05)).mean() * 100
        
        # Generate recommendation
        recommendation = "Buy" if avg_sentiment > 0.2 and positive_pct > 60 else "Consider" if avg_sentiment > 0 else "Avoid"
        
        # Key pros and cons
        positive_reviews = [r for i, r in enumerate(reviews) if sentiment_df['compound'].iloc[i] > 0.5]
        negative_reviews = [r for i, r in enumerate(reviews) if sentiment_df['compound'].iloc[i] < -0.5]
        
        # Calculate helpful metrics
        verified_reviews_count = len(reviews)  # In real app, count verified purchases
        recent_sentiment = avg_sentiment  # In real app, calculate sentiment trend
        
        return {
            "recommendation": recommendation,
            "sentiment": {
                "average": avg_sentiment,
                "positive_pct": positive_pct,
                "negative_pct": negative_pct,
                "neutral_pct": neutral_pct
            },
            "topics": topics,
            "product_features": product_features,
            "sample_positives": positive_reviews[:5] if positive_reviews else [],
            "sample_negatives": negative_reviews[:5] if negative_reviews else [],
            "metrics": {
                "verified_reviews": verified_reviews_count,
                "recent_sentiment": recent_sentiment,
                "review_count": len(reviews)
            },
            "product_info": product_info or {}
        }

def extract_asin_from_url(url):
    """Extract ASIN from Amazon URL"""
    # Try to parse URL
    parsed_url = urlparse(url)
    
    # Check if it's an Amazon URL
    if 'amazon' not in parsed_url.netloc:
        return None
    
    # Extract product ID from path
    path_parts = parsed_url.path.split('/')
    
    # Look for dp or gp/product in the path
    if 'dp' in path_parts:
        idx = path_parts.index('dp')
        if idx + 1 < len(path_parts):
            return path_parts[idx + 1]
    elif 'gp' in path_parts and 'product' in path_parts:
        idx = path_parts.index('product')
        if idx + 1 < len(path_parts):
            return path_parts[idx + 1]
    
    # Check if ASIN is in query parameters
    query_params = parse_qs(parsed_url.query)
    if 'ASIN' in query_params:
        return query_params['ASIN'][0]
    
    return None

def fetch_amazon_reviews(product_id, domain="in", location="110001", pages=3):
    """Fetch Amazon reviews using Oxylabs API"""
    print(f"Starting to fetch reviews for product ID: {product_id} from Amazon {domain}")
    
    # For testing - try a direct Amazon URL for debugging
    try:
        # First try a test HTTP request to check connectivity
        test_response = requests.get("https://httpbin.org/ip", timeout=5)
        print(f"Network connectivity test: {test_response.status_code}")
    except Exception as e:
        print(f"Network connectivity test failed: {e}")
    
    payload = {
        'source': 'amazon_reviews',
        'query': product_id,
        'domain': domain,
        'geo_location': location,
        'start_page': '1',
        'pages': str(pages),
        'parse': True,
        'context': [
            {'key': 'product_info', 'value': True}
        ]
    }
    
    print(f"Using credentials: {OXYLABS_USERNAME[:3]}...{OXYLABS_USERNAME[-3:]} / {len(OXYLABS_PASSWORD) * '*'}")
    print(f"Targeting Amazon domain: amazon.{domain}")
    
    try:
        # Send the request with increased timeout
        response = requests.request(
            'POST',
            'https://realtime.oxylabs.io/v1/queries',
            auth=(OXYLABS_USERNAME, OXYLABS_PASSWORD),
            json=payload,
            timeout=60  # Increased timeout to 60 seconds
        )
        
        print(f"Response status code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            
            if not results:
                print("No results returned from Oxylabs API")
                return [], {}
                
            # Extract reviews and product info
            reviews = []
            product_info = {}
            
            for result in results:
                content = result.get('content', {})
                
                # Get product info from the first result
                if not product_info and 'product_info' in content:
                    product_info = {
                        'title': content['product_info'].get('title', ''),
                        'rating': content['product_info'].get('rating', ''),
                        'image_url': content['product_info'].get('image_url', ''),
                        'price': content['product_info'].get('price', '')
                    }
                
                # Extract reviews
                if 'reviews' in content:
                    for review in content['reviews']:
                        if 'content' in review and review['content']:
                            reviews.append(review['content'])
            
            print(f"Successfully fetched {len(reviews)} reviews for product {product_id}")
            return reviews, product_info
        else:
            print(f"Error response from Oxylabs API: {response.status_code}")
            print(response.text)
            return [], {}
            
    except Exception as e:
        print(f"Exception when fetching reviews: {e}")
        return [], {}

# Initialize the ReviewAnalyzer
analyzer = ReviewAnalyzer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        product_url = request.form.get('product_url')
        
        # Extract ASIN from URL
        product_id = extract_asin_from_url(product_url)
        
        if not product_id:
            flash("Invalid Amazon URL. Please provide a valid Amazon product URL.", "danger")
            return redirect(url_for('index'))
        
        # Extract domain from URL
        parsed_url = urlparse(product_url)
        domain = parsed_url.netloc.split('.')[-1] if 'amazon' in parsed_url.netloc else 'in'
        
        # Fetch reviews using Oxylabs API
        reviews, product_info = fetch_amazon_reviews(product_id, domain=domain)
        
        if not reviews:
            # Use sample data if no reviews fetched
            reviews = load_sample_data()
            flash("Could not fetch reviews from Amazon. Using sample data instead.", "warning")
        
        # Analyze reviews
        analysis_results = analyzer.analyze_reviews(reviews, product_info)
        
        # Create visualizations
        sentiment_data = {
            "categories": ["Positive", "Neutral", "Negative"],
            "values": [
                analysis_results["sentiment"]["positive_pct"],
                analysis_results["sentiment"]["neutral_pct"],
                analysis_results["sentiment"]["negative_pct"]
            ]
        }
        
        return render_template(
            'results.html', 
            product_url=product_url,
            product_id=product_id,
            analysis=analysis_results,
            sentiment_data=json.dumps(sentiment_data),
            review_count=len(reviews)
        )

def load_sample_data():
    # Dummy data for testing
    return [
        "This product is amazing! It works exactly as described and the quality is excellent.",
        "Not worth the money. Broke after two weeks of use.",
        "Average product, nothing special but gets the job done.",
        "The customer service was terrible when I had issues with this product.",
        "Best purchase I've made this year! Highly recommend to everyone.",
        "Battery life is short, disappointing overall.",
        "The design is beautiful but functionality is lacking.",
        "Great value for the price, exceeded my expectations.",
        "The product arrived damaged, very frustrating experience.",
        "It's okay but there are better alternatives in the market."
    ]

if __name__ == '__main__':
    app.run(debug=True)
