import os
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

def fetch_tweet(
    search_query: str,
    max_results: int = 10,
    tweet_fields: str = "author_id,created_at,text,public_metrics,lang",
    expansions: str = None,
    next_token: str = None,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Fetch tweets from Twitter's recent search API v2.
    
    Args:
        search_query (str): Twitter search query (e.g., 'from:twitterdev OR #twitterdev -is:retweet')
        max_results (int): Number of tweets to return (10-100)
        tweet_fields (str): Comma-separated fields to return (default: basic fields)
        expansions (str): Comma-separated expansions (optional)
        next_token (str): Pagination token for next results page
        verbose (bool): Print raw API response
    
    Returns:
        pd.DataFrame: DataFrame containing tweets and metadata
    """
    # Authentication setup
    bearer_token = os.getenv("X_BEARER_TOKEN")
    if not bearer_token:
        raise ValueError("Missing X_BEARER_TOKEN in environment variables")
    
    endpoint = "https://api.twitter.com/2/tweets/search/recent"
    
    # Configure request parameters
    params = {
        'query': search_query,
        'max_results': max(min(max_results, 100), 10),  # Enforce 10-100 range
        'tweet.fields': tweet_fields,
    }
    
    if expansions:
        params['expansions'] = expansions
    if next_token:
        params['next_token'] = next_token

    # Helper function for authentication
    def bearer_oauth(req):
        req.headers["Authorization"] = f"Bearer {bearer_token}"
        req.headers["User-Agent"] = "v2RecentSearchPython"
        return req

    # Make API request
    response = requests.get(endpoint, auth=bearer_oauth, params=params)
    
    # Handle response
    if response.status_code != 200:
        raise Exception(
            f"API Request failed (Status {response.status_code}): {response.text}"
        )
    
    json_response = response.json()
    
    if verbose:
        print(json.dumps(json_response, indent=4, sort_keys=True))
    
    # Process and return data
    data = json_response.get('data', [])
    meta = json_response.get('meta', {})
    
    df = pd.DataFrame(data)
    
    if not df.empty:
        df['created_at'] = pd.to_datetime(df['created_at'])
        if 'public_metrics' in df.columns:
            metrics_df = pd.json_normalize(df['public_metrics'])
            df = df.drop('public_metrics', axis=1).join(metrics_df)
    
    # Add pagination token to results
    df.attrs['next_token'] = meta.get('next_token')
    
    return df

