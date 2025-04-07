# fetch_tweets.py
import data_processing_framework as dpf
import os
import time
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from dotenv import load_dotenv
import tweepy
from tweepy import Client, TooManyRequests, Paginator
import re
from tweepy.errors import TweepyException, TooManyRequests, TwitterServerError, BadRequest
from requests.exceptions import ConnectionError
from typing import Optional

from textblob import TextBlob
from wordcloud import WordCloud
from collections import Counter
import nltk
from datetime import datetime as dt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
import joblib

class TwitterServerError(Exception):
    pass

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Load credentials
load_dotenv()
bearer_token = os.getenv("X_BEARER_TOKEN")
if not bearer_token:
    raise ValueError("Missing X_BEARER_TOKEN in environment variables")

# Create Tweepy Client
client = tweepy.Client(bearer_token=BEARER_TOKEN)

# -----------------------------
# Load Data
# ---------------------------

def load_tweets(interactive=True, query=None, csv_path=None, limit=450):
    """
    Load tweets either from Twitter API or from a local CSV.

    Args:
        interactive (bool): Ask user what to do (default True)
        query (str): Twitter query (used if interactive=False and API is chosen)
        csv_path (str): Path to CSV file (used if interactive=False and local file is chosen)
        limit (int): Max tweets to fetch if using API

    Returns:
        pd.DataFrame: Loaded tweet data
    """
    if interactive:
        print("\n🔍 How would you like to load tweets?")
        print("1. Fetch from Twitter API")
        print("2. Load from Kaggle")
        choice = input("Enter 1 or 2: ").strip()

        if choice == "1":
            query = input("Enter Twitter search query (e.g. 'UK immigration lang:en -is:retweet'): ").strip()
            limit_input = input("Enter number of tweets to fetch (max 450): ").strip()
            limit = min(int(limit_input or 450), 450)
            return fetch_tweets(query=query, cooldown=True)

        elif choice == "2":
            # Get dataset and handle None case
            df = dpf.fetch_kaggle_dataset(search_query="sentiment140")
            if df is None:
                raise ValueError("❌ No datasets found for search query 'tweets_labeled'")
                   
            print(f"✅ Loaded {len(df)} tweets from Kaggle dataset")
            return df
            
        else:
            raise ValueError("Invalid choice. Please enter 1 or 2.")

    else:
        # Non-interactive mode remains unchanged
        if query:
            return fetch_tweets(query=query, cooldown=True)
        elif csv_path:
            if not os.path.isfile(csv_path):
                raise FileNotFoundError(f"❌ File not found: {csv_path}")
            return safe_read_csv(csv_path)
        else:
            raise ValueError("Provide either `query` or `csv_path`.")

def safe_read_csv(path):
    """Tries multiple encodings to safely read a CSV."""
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return pd.read_csv(path, encoding="ISO-8859-1")  # Latin-1 fallback
        except Exception as e:
            raise RuntimeError(f"❌ Failed to read CSV: {e}")


def fetch_tweet(
    search_query: str,
    max_results: int = 10,
    tweet_fields: str = "author_id,created_at,text,public_metrics,lang",
    expansions: str = None,
    next_token: str = None,
    verbose: bool = False,
    max_retries: int = 3,
    base_wait: int = 60
) -> pd.DataFrame:
    """
    Fetch tweets with rate limit handling and automatic retries.
    """
    endpoint = "https://api.twitter.com/2/tweets/search/recent"
    headers = {
        "Authorization": f"Bearer {bearer_token}",
        "User-Agent": "v2RecentSearchPython"
    }

    params = {
        'query': search_query,
        'max_results': max(min(max_results, 100), 10),
        'tweet.fields': tweet_fields,
    }

    if expansions:
        params['expansions'] = expansions
    if next_token:
        params['next_token'] = next_token

    retries = 0
    last_status = 200

    while retries <= max_retries:
        try:
            response = requests.get(endpoint, headers=headers, params=params)
            
            if response.status_code == 429:
                # Handle rate limits with exponential backoff
                retry_after = int(response.headers.get('Retry-After', base_wait * (2 ** retries)))
                print(f"Rate limited. Waiting {retry_after} seconds (retry {retries + 1}/{max_retries})")
                time.sleep(retry_after)
                retries += 1
                continue
                
            if response.status_code != 200:
                raise Exception(f"API Error {response.status_code}: {response.text}")

            json_response = response.json()
            
            if verbose:
                print(json.dumps(json_response, indent=4, sort_keys=True))

            # Process data
            data = json_response.get('data', [])
            meta = json_response.get('meta', {})
            
            df = pd.DataFrame(data)
            
            if not df.empty:
                df['created_at'] = pd.to_datetime(df['created_at'])
                if 'public_metrics' in df.columns:
                    metrics_df = pd.json_normalize(df['public_metrics'])
                    df = df.drop('public_metrics', axis=1).join(metrics_df)
            
            df.attrs['next_token'] = meta.get('next_token')
            return df

        except requests.exceptions.RequestException as e:
            if retries < max_retries:
                wait_time = base_wait * (2 ** retries)
                print(f"Connection error: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                retries += 1
            else:
                raise

    raise Exception(f"Failed after {max_retries} retries. Last status: {last_status}")



def save_to_csv(df, filename="tweets.csv"):
    """
    Save tweet DataFrame to a CSV file.
    """
    df.to_csv(filename, index=False)
    print(f"✅ Tweets saved to {filename}")

# -----------------------------
# Data Cleaning Functions
# ---------------------------

def drop_and_encode_sentiment(df):
    """
    Drops duplicates & irrelevant columns, parses date, and encodes sentiment labels.
    Also extracts time features: month, day of week, and season.
    """
    # Drop duplicates
    df = df.drop_duplicates(subset="text")

    # Drop irrelevant columns
    df = df.drop(columns=["id", "query", "user"], errors="ignore")

    # Clean timezone strings from date (e.g. 'PDT', 'GMT')
    df["created_at"] = df["created_at"].str.replace(r"\s[A-Z]{2,4}\s", " ", regex=True)

    # Convert to datetime
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

    # Map sentiment labels
    sentiment_map = {0: "negative", 4: "positive"}
    df["sentiment"] = df["sentiment"].map(sentiment_map)

    # Drop rows missing key values
    df = df.dropna(subset=["sentiment", "created_at", "text"])

    # Extract time features
    df["month"] = df["created_at"].dt.month_name()
    df["day_of_week"] = df["created_at"].dt.day_name()

    # Map month to season (Northern Hemisphere)
    def month_to_season(month):
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        elif month in [9, 10, 11]:
            return "Autumn"
        else:
            return "Unknown"

    df["season"] = df["created_at"].dt.month.apply(month_to_season)

    return df.reset_index(drop=True)


def clean_text(text, *,
               remove_urls=True,
               remove_mentions=True,
               remove_hashtags=True,
               remove_non_alpha=True,
               lowercase=True,
               stopwords_set=None,
               min_token_length=1):
    """
    Cleans tweet text using regex and customizable options. No NLTK required.

    Args:
        text (str): The raw tweet text
        remove_urls (bool): Remove URLs (http, https, www)
        remove_mentions (bool): Remove Twitter @mentions
        remove_hashtags (bool): Remove hashtags (but keep words if False)
        remove_non_alpha (bool): Remove non-alphabetic characters
        lowercase (bool): Convert text to lowercase
        stopwords_set (set): Custom stopwords to remove (case-insensitive)
        min_token_length (int): Drop tokens shorter than this length

    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return ""

    # Apply regex-based cleaning
    if remove_urls:
        text = re.sub(r"http\S+|www\S+", "", text)
    if remove_mentions:
        text = re.sub(r"@\w+", "", text)
    if remove_hashtags:
        text = re.sub(r"#\w+", "", text)
    if remove_non_alpha:
        text = re.sub(r"[^a-zA-Z\s]", "", text)

    if lowercase:
        text = text.lower()

    # Split and optionally filter
    tokens = text.split()
    if stopwords_set:
        tokens = [t for t in tokens if t.lower() not in stopwords_set]
    if min_token_length > 1:
        tokens = [t for t in tokens if len(t) >= min_token_length]

    return " ".join(tokens)

# -----------------------------
# Stopword Removal Function
# -----------------------------

def remove_stopwords(df, text_col, new_col=None, extra_stopwords=None):
    """
    Removes stopwords from a DataFrame text column (Twitter-aware).

    Args:
        df (pd.DataFrame): DataFrame containing the text data
        text_col (str): Name of the input column containing cleaned text
        new_col (str or None): Name of the output column. If None, overwrites input column
        extra_stopwords (set or None): Custom stopwords to include

    Returns:
        pd.DataFrame: DataFrame with stopwords removed from text
    """
    if text_col not in df.columns:
        raise KeyError(f"'{text_col}' column not found in DataFrame.")

    if new_col is None:
        new_col = text_col

    # Base + social media stopwords
    stop_words = set(ENGLISH_STOP_WORDS)
    twitter_noise = {
        "im", "amp", "u", "dont", "got", "gonna", "yeah", "tbh", "idk", "rt","thats",
        "ya", "wanna", "lol", "omg", "like", "just", "really", "day", "oh",
        "know", "one", "make", "even", "get", "thing", "think", "facebook"
    }
    stop_words.update(twitter_noise)

    # Add custom stopwords
    if extra_stopwords:
        stop_words.update(extra_stopwords)

    # Apply filtering
    df[new_col] = df[text_col].apply(
        lambda x: " ".join([
            word for word in str(x).split()
            if word.lower() not in stop_words
        ])
    )

    return df

# -----------------------------
# Lemmatization Function
# -----------------------------
def lemmatize_text(text):
    return " ".join([lemmatizer.lemmatize(w) for w in text.split()])


# ----------------------------------------------------------------
# Visualization Functions: Top Words Plots, Time Trends,
# N-gram Frequency Plot, Word Cloud, and Sentiment Distribution     
# ---------------------------------------------------------------

def plot_time_trend(df, time_col="created_at", freq="W", title="Tweet Trend Over Time"):
    """
    Plots tweet volume over time using resampling by frequency.

    Args:
        df (pd.DataFrame): DataFrame with datetime column
        time_col (str): Name of datetime column
        freq (str): Pandas resample freq ('D', 'W', 'M', etc.)
        title (str): Plot title
    """
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    trend = df.set_index(time_col).resample(freq).size().reset_index(name="tweet_count")

    plt.figure(figsize=(12, 4))
    sns.lineplot(data=trend, x=time_col, y="tweet_count", marker="o")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Tweet Count")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_seasonal_trend(df, season_col="season", title="Tweet Distribution by Season"):
    """
    Plots tweet percentage distribution by season with percentage labels.

    Args:
        df (pd.DataFrame): DataFrame with a season column
        season_col (str): Column name for seasons
        title (str): Plot title
    """
    # Count and calculate percentages
    season_counts = df[season_col].value_counts(normalize=True).reset_index()
    season_counts.columns = ["season", "percent"]
    season_counts["percent"] = season_counts["percent"] * 100

    # Define color map
    color_map = {
        "Spring": "green",
        "Summer": "orange",
        "Autumn": "brown",
        "Winter": "blue"
    }

    palette = [color_map.get(season, "gray") for season in season_counts["season"]]

    # Plot
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(
        data=season_counts,
        x="season",
        y="percent",
        hue="season",
        palette=palette,
        legend=False
    )

    # Add percentage labels
    for i, row in season_counts.iterrows():
        label = f"{row['percent']:.1f}%"
        ax.text(i, row["percent"] + 1, label, ha="center", va="bottom")

    plt.title(title)
    plt.xlabel("Season")
    plt.ylabel("Percentage of Tweets")
    plt.ylim(0, max(season_counts["percent"]) + 10)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

def plot_sentiment_distribution(df, sentiment_col=None, title="Sentiment Distribution"):
    """
    Plots sentiment distribution as percentages using seaborn.
    Robust to column naming variations and empty data.

    Args:
        df (pd.DataFrame): DataFrame with sentiment labels
        sentiment_col (str): Column name. If None, auto-detects
        title (str): Title for the plot
    """
    # Auto-detect sentiment column
    if sentiment_col is None:
        candidates = ["sentiment", "Sentiment", "label", "polarity"]
        sentiment_col = next((col for col in candidates if col in df.columns), None)

    if sentiment_col is None or sentiment_col not in df.columns:
        raise KeyError(f"Sentiment column not found. Expected one of: {candidates}")

    # Calculate percentages safely
    try:
        sentiment_percent = (
            df[sentiment_col]
            .value_counts(normalize=True)
            .reset_index(name='percent')
            .rename(columns={'index': 'sentiment'})
        )
        sentiment_percent["percent"] *= 100
    except KeyError:
        raise
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        return

    # Handle empty data case
    if sentiment_percent.empty:
        print("No sentiment data to plot")
        return

    # Color map with fallback
    sentiment_colors = {
        "positive": "#4CAF50",  # Green
        "neutral": "#9E9E9E",   # Grey
        "negative": "#F44336"   # Red
    }
    
    # Create palette ensuring lowercase labels
    labels = sentiment_percent["sentiment"].str.lower()
    palette = [sentiment_colors.get(label, "#2196F3") for label in labels]  # Blue fallback

    # Plot
    plt.figure(figsize=(8, 5))
    sns.set(style="whitegrid", font_scale=1.1)

    ax = sns.barplot(
        data=sentiment_percent,
        x="sentiment",
        y="percent",
        hue="sentiment",
        palette=palette,
        dodge=False,
        legend=False
    )

    # Add percentage labels
    for i, (_, row) in enumerate(sentiment_percent.iterrows()):
        ax.text(
            i,
            row["percent"] + 1,
            f"{row['percent']:.1f}%",
            ha='center',
            va='bottom',
            fontsize=12,
            color='#212121'
        )

    plt.title(title, pad=20, fontsize=14, fontweight='semibold')
    plt.xlabel("Sentiment Category", labelpad=10)
    plt.ylabel("Percentage (%)", labelpad=10)
    plt.ylim(0, min(100, sentiment_percent["percent"].max() * 1.2))
    plt.xticks(rotation=15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_top_words(df, text_col="clean_text", top_n=20):# Top Words Analysis/Word Frequency (Unigrams)
    """
    Plots top N most frequent words from a text column.

    Args:
        df (pd.DataFrame): DataFrame containing text
        text_col (str): Column name with space-separated cleaned text
        top_n (int): Number of top words to plot
    """
    all_words = " ".join(df[text_col].dropna()).split()
    common_words = Counter(all_words).most_common(top_n)

    data = pd.DataFrame(common_words, columns=["word", "count"])

    plt.figure(figsize=(10, 4))
    sns.set(style="whitegrid")
    ax = sns.barplot(data=data, x="word", y="count", hue="word", palette="viridis", legend=False)

    plt.title(f"Top {top_n} Words")
    plt.xticks(rotation=45)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def plot_ngrams(df, text_col="clean_text", ngram_range=(2, 2), top_n=20):
    """
    Plots top N n-grams (bigram/trigram) from the text column using Seaborn.

    Args:
        df (pd.DataFrame): DataFrame with cleaned text
        text_col (str): Column name for text
        ngram_range (tuple): (n, n) e.g., (2,2)=bigrams, (3,3)=trigrams
        top_n (int): Number of n-grams to display
    """
    vec = CountVectorizer(ngram_range=ngram_range, stop_words="english")
    ngram_matrix = vec.fit_transform(df[text_col])
    ngram_counts = ngram_matrix.sum(axis=0).A1
    vocab = vec.get_feature_names_out()

    top_ngrams = sorted(zip(vocab, ngram_counts), key=lambda x: x[1], reverse=True)[:top_n]
    ngram_df = pd.DataFrame(top_ngrams, columns=["ngram", "count"])

    plt.figure(figsize=(10, 4))
    sns.set(style="whitegrid")
    ax = sns.barplot(
        data=ngram_df,
        x="ngram",
        y="count",
        hue="ngram",         
        palette="magma",
        legend=False         
    )
    plt.title(f"Top {top_n} {'-'.join(map(str, ngram_range))}-grams")
    plt.xticks(rotation=45)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

def plot_wordcloud(df, text_col="clean_text", sentiment=None):
    text = " ".join(df[df["sentiment"] == sentiment][text_col]) if sentiment else " ".join(df[text_col])
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Word Cloud - {sentiment if sentiment else 'All'} Tweets")
    plt.tight_layout()
    plt.show()

# -----------------------------
# Model Training
# -----------------------------
def split_data(df, text_col="filtered_text", label_col="sentiment"):
    X = df[text_col]
    y = df[label_col]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


def vectorize_text(X_train, X_test, max_features=3000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return X_train_vec, X_test_vec, vectorizer

def train_and_evaluate_models(X_train_vec, y_train, X_test_vec, y_test, metric="accuracy"):
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "NaiveBayes": MultinomialNB(),
       # "RandomForest": RandomForestClassifier(n_estimators=100),
        "LinearSVC": LinearSVC(max_iter=1000)
    }
    results = []
    best_model = None
    best_score = 0
    best_name = None
    best_cm = None
    best_cr = None

    for name, model in models.items():
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        report_str = classification_report(y_test, y_pred, output_dict=False)

        print(f"\n{name} Accuracy: {acc:.4f}")
        print(report_str)

        # Normalized Confusion Matrix
        cm = confusion_matrix(y_test, y_pred, normalize='true')
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", cbar=False,
                    xticklabels=model.classes_, yticklabels=model.classes_)
        plt.title(f"Normalized Confusion Matrix - {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.show()

        results.append({
            "model": name,
            "accuracy": acc,
            "f1_score": f1,
            "precision_pos": report_dict.get("positive", {}).get("precision", None),
            "recall_pos": report_dict.get("positive", {}).get("recall", None),
            "precision_neg": report_dict.get("negative", {}).get("precision", None),
            "recall_neg": report_dict.get("negative", {}).get("recall", None)
        })

        score = acc if metric == "accuracy" else f1
        if score > best_score:
            best_score = score
            best_model = model
            best_name = name
            best_cm = cm
            best_cr = report_str

    results_df = pd.DataFrame(results)
    best_model.name = best_name
    best_model.score = best_score
    best_model.cm = best_cm
    best_model.report = best_cr
    return best_model, results_df

# -----------------------------
# Plot Model Comparison
# -----------------------------
def plot_model_performance(results_df):
    plt.figure(figsize=(8, 4))
    sns.barplot(data=results_df.melt(id_vars="model"), x="model", y="value", hue="variable", palette="Set2")
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()


def save_best_model(model, vectorizer, model_path="models/sentiment_model.pkl", vec_path="models/tfidf_vectorizer.pkl"):
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vec_path)
    timestamp = dt.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n✅ [{timestamp}] Best model '{model.name}' saved with accuracy: {model.score:.4f}")


# -----------------------------
# Unified Pipeline
# -----------------------------
def run_training_pipeline(df, text_col="filtered_text", label_col="sentiment", max_features=3000):
    print("\n...Splitting data...")
    X_train, X_test, y_train, y_test = split_data(df, text_col, label_col)

    print("\n...Vectorizing text with TF-IDF...")
    X_train_vec, X_test_vec, vectorizer = vectorize_text(X_train, X_test, max_features)

    print("\n...Training and evaluating models...")
    best_model, results = train_and_evaluate_models(X_train_vec, y_train, X_test_vec, y_test)

    return best_model, vectorizer, results


# -----------------------------
# Predict New Sentiment
# -----------------------------
def predict_sentiment(text, model, vectorizer, clean_text_fn, stopword_fn):
    clean = clean_text_fn(text)
    filtered = stopword_fn(
        pd.DataFrame({"text": [clean]}), text_col="text", new_col="text"
    )["text"].iloc[0]
    vec = vectorizer.transform([filtered])
    return model.predict(vec)[0]

# -----------------------------
# Load Model
# -----------------------------
def load_model(model_path="models/sentiment_model.pkl", vec_path="models/tfidf_vectorizer.pkl"):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    return model, vectorizer


# -----------------------------
# get_sentiment
# -----------------------------

def get_sentiment(text):
    """Classifies sentiment using TextBlob."""
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return "positive"
    elif polarity < -0.1:
        return "negative"
    else:
        return "neutral"


def add_sentiment_column(df, clean_text_col="clean_text"):
    """Adds a sentiment column to a DataFrame of cleaned tweets."""
    df["sentiment"] = df[clean_text_col].apply(get_sentiment)
    return df
