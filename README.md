# 🔍 Amazon Review Analyzer

![Amazon Review Analyzer Banner](https://raw.githubusercontent.com/username/amazon-review-analyzer/main/static/banner.png)

[![Deployed on Render](https://img.shields.io/badge/Deployed%20on-Render-purple?style=for-the-badge&logo=render)](https://amazon-review-analyzer.onrender.com)
[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0-green?style=for-the-badge&logo=flask)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](https://opensource.org/licenses/MIT)

An AI-powered web application that analyzes Amazon product reviews to help you make informed purchasing decisions. Should you buy it, consider it, or avoid it? Let the data decide! 💡

## 🌟 Features

- *One-Click Analysis*: Simply paste an Amazon product URL and get instant insights
- *Smart Review Processing*: Uses advanced NLP techniques to analyze sentiment and extract key topics
- *Clear Recommendations*: Get straightforward Buy/Consider/Avoid recommendations based on review data
- *Visual Insights*: Interactive charts and visualizations of review sentiment and product features
- *Key Topics Extraction*: Identifies the most discussed topics and features in reviews
- *Sample Reviews*: See examples of the most positive and negative reviews for quick assessment

## 📊 How It Works

1. *Data Collection*: The app extracts the product ID from the Amazon URL and fetches reviews using the Oxylabs API
2. *Sentiment Analysis*: NLTK's SentimentIntensityAnalyzer evaluates the emotional tone of each review
3. *Topic Modeling*: Latent Dirichlet Allocation (LDA) identifies key topics mentioned across reviews
4. *Feature Extraction*: TF-IDF vectorization highlights important product features
5. *Results Visualization*: Interactive charts display sentiment distribution and key insights
6. *Recommendation Engine*: Based on aggregated metrics, the app provides a clear recommendation

## 🔧 Technology Stack

- *Backend*: Python, Flask
- *NLP Processing*: NLTK, Scikit-learn
- *Data Visualization*: Chart.js, Plotly
- *Frontend*: HTML5, CSS3, JavaScript, Bootstrap 5
- *APIs*: Oxylabs (for Amazon data scraping)
- *Deployment*: Render

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Oxylabs API credentials (or you can use the fallback test mode)

### Installation

1. Clone the repository:
   
   git clone https://github.com/yourusername/amazon-review-analyzer.git
   cd amazon-review-analyzer
   

2. Install the required dependencies:
   
   pip install -r requirements.txt
   

3. Create a .env file in the root directory with your Oxylabs credentials:
   
   OXYLABS_USERNAME=your_username
   OXYLABS_PASSWORD=your_password
   SECRET_KEY=your_secret_key
   

4. Run the application:
   
   python app.py
   

5. Open your browser and navigate to http://localhost:5000

## 🖼 Screenshots

  🔹 Home Page

A clean and minimal homepage where users can paste the Amazon product URL and initiate analysis.
<p align="center">
  <img src="https://raw.githubusercontent.com/adityakswain190/Product-Review-Aggregator/main/static/screenshot-home.png" alt="Home Page" width="45%">
</p>
  🔹 Results Page

Displays a vertical, scrollable section with visual insights, sentiment distribution, key topics, and a Buy/Consider/Avoid recommendation.
<p align="center">
  <img src="https://raw.githubusercontent.com/adityakswain190/Product-Review-Aggregator/main/static/screenshot-results.png" alt="Results Page" width="45%">
</p>

## 📝 Usage

1. On the homepage, paste the full Amazon product URL in the input field
2. Click "Analyze Reviews" and wait for the processing to complete
3. Browse through the analysis results, including:
   - Overall recommendation (Buy/Consider/Avoid)
   - Sentiment distribution chart
   - Key topics mentioned in reviews
   - Important product features
   - Sample positive and negative reviews

## 🧠 Under the Hood

### ReviewAnalyzer Class

The heart of the application is the ReviewAnalyzer class, which:

- Cleans and preprocesses text data
- Performs sentiment analysis on reviews
- Extracts key topics using LDA
- Identifies important product features using TF-IDF
- Generates recommendations based on aggregated metrics

### Data Processing

The application uses several sophisticated NLP techniques:

- *Tokenization*: Breaking text into individual words
- *Stopword Removal*: Filtering out common words that don't add meaning
- *Sentiment Analysis*: Evaluating the emotional tone of reviews
- *Topic Modeling*: Identifying themes and topics across reviews
- *TF-IDF Vectorization*: Highlighting important terms

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to help improve this project.

1. Fork the repository
2. Create your feature branch: git checkout -b feature/amazing-feature
3. Commit your changes: git commit -m 'Add some amazing feature'
4. Push to the branch: git push origin feature/amazing-feature
5. Open a pull request


## 🙏 Acknowledgments

- [NLTK](https://www.nltk.org/) for natural language processing
- [Scikit-learn](https://scikit-learn.org/) for machine learning algorithms
- [Flask](https://flask.palletsprojects.com/) for the web framework
- [Bootstrap](https://getbootstrap.com/) for the responsive UI
- [Chart.js](https://www.chartjs.org/) for data visualization
- [Oxylabs](https://oxylabs.io/) for the web scraping API

---

<p align="center">
  Made with ❤ by <a href="https://github.com/yourusername">Your Name</a>
</p>
