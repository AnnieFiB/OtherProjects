from transformers import pipeline, MarianMTModel, MarianTokenizer
from langdetect import detect
import torch
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple  

# Extended language support
LANG_CODE_MAP = {
    "es": "es-en",  # Spanish to English
    "fr": "fr-en",  # French to English
    "de": "de-en",  # German to English
    "it": "it-en",  # Italian to English
    "pt": "pt-en",  # Portuguese to English
    "zh": "zh-en",  # Chinese to English
    "ja": "ja-en"   # Japanese to English
}

# Reverse map for translating replies back
REVERSE_LANG_MAP = {
    "es": "en-es",
    "fr": "en-fr",
    "de": "en-de",
    "it": "en-it",
    "pt": "en-pt",
    "zh": "en-zh",
    "ja": "en-ja"
}

def detect_language(text):
    """Detect the language of input text"""
    try:
        return detect(text)
    except:
        return "unknown"

def load_translation_model(code: str) -> Tuple[MarianMTModel, MarianTokenizer]:
    """Load translation model for given language code"""
    try:
        model_name = f"Helsinki-NLP/opus-mt-{code}"
        logger.info(f"Loading translation model: {model_name}")
        
        # Check for SentencePiece availability
        try:
            import sentencepiece  # No need to actually use it, just check existence
        except ImportError as e:
            logger.error("SentencePiece library not found. Please install with: pip install sentencepiece")
            raise RuntimeError("Missing SentencePiece dependency") from e
        
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load translation model: {str(e)}")
        raise

def translate_to_english(text, lang):
    """Translate text to English if not already English"""
    if lang == "en" or lang not in LANG_CODE_MAP:
        return text
    model, tokenizer = load_translation_model(LANG_CODE_MAP[lang])
    tokens = tokenizer([text], return_tensors="pt", padding=True)
    translated = model.generate(**tokens)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

def translate_from_english(text, lang):
    """Translate text from English to target language"""
    if lang == "en" or lang not in REVERSE_LANG_MAP:
        return text
    model, tokenizer = load_translation_model(REVERSE_LANG_MAP[lang])
    tokens = tokenizer([text], return_tensors="pt", padding=True)
    translated = model.generate(**tokens)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# =========================
# Banking AutoReply Logic
# =========================
CATEGORY_KEYWORDS = {
    "Account Access": ["log in", "login", "account locked", "forgot password", "can't access"],
    "Card Issues": ["lost card", "stolen card", "unauthorized", "declined", "blocked card"],
    "Loan Inquiry": ["loan", "apply", "interest rate", "loan status", "mortgage"],
    "Transaction Dispute": ["refund", "charge", "dispute", "not received", "fraud"],
    "General Banking Help": ["open account", "branch", "change contact", "help", "support"]
}

REPLY_TEMPLATES = {
    "Account Access": "Thank you for reaching out. For security reasons, we can't make account changes via email. Please use the 'Forgot Password' option on our login page or call us directly at [Support Number].",
    "Card Issues": "We're sorry to hear you're having trouble with your card. Please contact our card services team immediately at [Phone Number] to block your card and investigate the issue.",
    "Loan Inquiry": "Thank you for your interest in our loan services. You can check eligibility, rates, and apply directly on our loan portal at [Loan Page URL].",
    "Transaction Dispute": "We understand your concern regarding the transaction. Please complete our dispute form available on your online banking portal or contact our support team.",
    "General Banking Help": "Thank you for contacting us. A member of our support team will review your message and get back to you shortly."
}

def classify_email(text):
    """Classify support email into predefined categories"""
    text = text.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return category
    return "General Banking Help"

def generate_reply(category):
    """Generate appropriate reply based on category"""
    return REPLY_TEMPLATES.get(category, REPLY_TEMPLATES["General Banking Help"])

# =========================
# Sentiment Analysis Logic
# =========================
def load_sentiment_model():
    """Load pre-trained sentiment analysis model"""
    return pipeline(
        "sentiment-analysis", 
        model="distilbert-base-uncased-finetuned-sst-2-english",
        torch_dtype=torch.float16  # Add this line
    )

def analyze_sentiment(text, sentiment_model):
    """Analyze sentiment of text and return label with confidence score"""
    result = sentiment_model(text)[0]
    return result["label"], result["score"]

# =========================
# File Handling & Visualization
# =========================
def load_text_from_csv(file_path, text_column="text"):
    """Load text data from CSV file"""
    try:
        df = pd.read_csv(file_path)
        return df[text_column].tolist()
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return []

def save_results_to_csv(results, output_path="results.csv"):
    """Save analysis results to CSV file"""
    pd.DataFrame(results).to_csv(output_path, index=False)
    return output_path

def plot_sentiment_distribution(results):
    """Visualize sentiment distribution"""
    df = pd.DataFrame(results)
    if 'sentiment' in df.columns:
        df['sentiment'].value_counts().plot(kind='bar')
        plt.title("Sentiment Distribution")
        plt.show()

def plot_support_categories(results):
    """Visualize support category distribution"""
    df = pd.DataFrame(results)
    if 'category' in df.columns:
        df['category'].value_counts().plot(kind='pie', autopct='%1.1f%%')
        plt.title("Support Categories")
        plt.show()

# =========================
# Main Processing Function
# =========================
def process_input(text, mode="auto", sentiment_model=None):
    """Main function to process input text based on selected mode"""
    lang = detect_language(text)
    translated = translate_to_english(text, lang)
    
    if mode == "review":
        label, score = analyze_sentiment(translated, sentiment_model)
        label_out = f"{translate_from_english(label, lang)} ({label})" if lang != "en" else label
        return {
            "original_text": text,
            "task": "sentiment",
            "language": lang,
            "translated": translated,
            "sentiment": label_out,
            "confidence": round(score, 3)
        }
    
    elif mode == "support":
        category = classify_email(translated)
        reply_en = generate_reply(category)
        reply_local = translate_from_english(reply_en, lang) if lang != "en" else reply_en
        return {
            "original_text": text,
            "task": "support_reply",
            "language": lang,
            "translated": translated,
            "category": category,
            "auto_reply": f"{reply_local} ({reply_en})" if lang != "en" else reply_en
        }

    return {"error": "Invalid mode. Use 'review' or 'support'"}

# Add these new functions to your existing smart_nlp_assistant.py

def initialize_models():
    """Initialize all required models and return them"""
    print("🔁 Loading models...")
    sentiment_model = load_sentiment_model()
    print("✅ Models loaded successfully")
    return sentiment_model

def get_user_mode():
    """Get user's choice of mode (review/support)"""
    print("Smart NLP Assistant - Select Mode:")
    print("1. Review Sentiment Analysis")
    print("2. Banking Support Auto-Reply")
    choice = input("Enter choice (1 or 2): ")
    return "review" if choice == "1" else "support"

def get_input_method():
    """Get user's choice of input method (text/csv)"""
    print("\nSelect Input Method:")
    print("1. Enter text directly")
    print("2. Process CSV file")
    choice = input("Enter choice (1 or 2): ")
    return "text" if choice == "1" else "csv"

def get_text_input():
    """Get text input from user"""
    return input("\nEnter your text: ")

def get_csv_input():
    """Get CSV file details from user"""
    file_path = input("\nEnter CSV file path: ")
    text_column = input("Enter column name containing text (default 'text'): ") or "text"
    return file_path, text_column

def process_single_text(text, mode, sentiment_model):
    """Process a single text input"""
    result = process_input(text, mode=mode, sentiment_model=sentiment_model)
    print("\nResults:")
    for key, value in result.items():
        print(f"{key.title().replace('_', ' ')}: {value}")
    return [result]  # Return as list for consistency

def process_csv_file(file_path, text_column, mode, sentiment_model):
    """Process a CSV file of texts"""
    texts = load_text_from_csv(file_path, text_column)
    
    if not texts:
        print("No valid text data found in the CSV file")
        return []
    
    print(f"\nProcessing {len(texts)} entries...")
    results = []
    for text in texts:
        results.append(process_input(text, mode=mode, sentiment_model=sentiment_model))
    
    # Save and show results
    output_path = save_results_to_csv(results)
    print(f"\n✅ Results saved to {output_path}")
    print("\nSample Results:")
    print(pd.DataFrame(results).head())
    
    return results

def show_visualizations(results, mode):
    """Display appropriate visualizations based on mode"""
    if not results:
        print("No results to visualize")
        return
    
    if mode == "review":
        print("\nSentiment Distribution:")
        plot_sentiment_distribution(results)
    else:
        print("\nSupport Categories Distribution:")
        plot_support_categories(results)