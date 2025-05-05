import requests
import json
import time
import random
import os
from typing import Dict, Optional, List
from dotenv import load_dotenv

load_dotenv()   

bearer_token = os.getenv("X_BEARER_TOKEN")

if not bearer_token:
    raise EnvironmentError("Missing BEARER_TOKEN in .env file")

# Base endpoint URL
endpoint_url = "https://api.twitter.com/2/tweets/search/recent"

# Default query parameters with expanded fields
default_query_parameters = {
    "query": '("netanyahu" OR "gaza" OR "isreal") lang:en -is:retweet',
    "tweet.fields": "id,text,author_id,created_at,public_metrics,entities,context_annotations,referenced_tweets",
    "user.fields": "id,name,username,description,public_metrics",
    "expansions": "author_id,referenced_tweets.id,referenced_tweets.id.author_id",
    "max_results": 10,
}

def request_headers(bearer_token: str) -> dict:
    """
    Sets up the request headers. 
    Returns a dictionary summarising the bearer token authentication details.
    """
    return {"Authorization": "Bearer {}".format(bearer_token)}

headers = request_headers(bearer_token)

def handle_rate_limits(response: requests.Response) -> None:
    """
    Handles rate limiting by checking response headers and implementing backoff.
    """
    if 'x-rate-limit-remaining' in response.headers:
        remaining = int(response.headers['x-rate-limit-remaining'])
        if remaining == 0:
            reset_time = int(response.headers['x-rate-limit-reset'])
            current_time = int(time.time())
            sleep_time = reset_time - current_time + 5  # Add 5 seconds buffer
            print(f"Rate limit reached. Sleeping for {sleep_time} seconds...")
            time.sleep(sleep_time)

def connect_to_endpoint(
    endpoint_url: str, 
    headers: dict, 
    parameters: dict,
    next_token: Optional[str] = None
) -> Dict:
    """
    Connects to the endpoint and requests data with pagination support.
    Returns a json with Twitter data if a 200 status code is yielded.
    Implements rate limiting and error handling.
    """
    # Add pagination token if provided
    if next_token:
        parameters['next_token'] = next_token

    response = requests.request(
        "GET", url=endpoint_url, headers=headers, params=parameters
    )
    
    # Handle rate limits
    handle_rate_limits(response)
    
    response_status_code = response.status_code
    if response_status_code != 200:
        if response_status_code == 429:  # Rate limit exceeded
            error_data = json.loads(response.text)
            if "UsageCapExceeded" in error_data.get("title", ""):
                raise Exception(
                    "Monthly API usage cap exceeded. Please check your Twitter API subscription status.\n"
                    f"Details: {error_data.get('detail', 'No additional details available')}"
                )
            else:
                raise Exception(
                    "Rate limit exceeded. Please try again later or upgrade your API subscription.\n"
                    f"HTTP {response_status_code}: {response.text}"
                )
        elif response_status_code >= 400 and response_status_code < 500:
            raise Exception(
                "Cannot get data, the program will stop!\nHTTP {}: {}".format(
                    response_status_code, response.text
                )
            )
        
        # Implement exponential backoff
        sleep_seconds = min(2 ** (response_status_code // 100) + random.random(), 60)
        print(
            "Cannot get data, your program will sleep for {} seconds...\nHTTP {}: {}".format(
                sleep_seconds, response_status_code, response.text
            )
        )
        time.sleep(sleep_seconds)
        return connect_to_endpoint(endpoint_url, headers, parameters, next_token)
    
    return response.json()

def get_all_tweets(
    endpoint_url: str,
    headers: dict,
    parameters: dict,
    max_pages: int = 5
) -> List[Dict]:
    """
    Fetches all tweets using pagination up to max_pages.
    Returns a list of tweet data.
    """
    all_tweets = []
    next_token = None
    page_count = 0
    
    while page_count < max_pages:
        response = connect_to_endpoint(endpoint_url, headers, parameters, next_token)
        
        # Add tweets to the collection
        if 'data' in response:
            all_tweets.extend(response['data'])
        
        # Check for next token
        if 'meta' in response and 'next_token' in response['meta']:
            next_token = response['meta']['next_token']
            page_count += 1
        else:
            break
    
    return all_tweets

# Example usage
if __name__ == "__main__":
    try:
        # Get all tweets with pagination
        all_tweets = get_all_tweets(endpoint_url, headers, default_query_parameters)
        
        # Print the number of tweets collected
        print(f"Collected {len(all_tweets)} tweets")
        
        # Save to file
        with open('tweets_data.json', 'w') as f:
            json.dump(all_tweets, f, indent=2)
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")