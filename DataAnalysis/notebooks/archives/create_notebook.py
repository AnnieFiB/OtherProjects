import nbformat as nbf

# Create a new notebook
nb = nbf.v4.new_notebook()

# Add markdown cell for title
nb.cells.append(nbf.v4.new_markdown_cell("# Twitter Data Analysis\n\nThis notebook demonstrates how to use the Twitter API v2 script and analyze the collected data."))

# Add code cell for imports
imports = '''# Import necessary libraries
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scripts.twitterapiv2 import get_all_tweets, default_query_parameters, request_headers
import os'''
nb.cells.append(nbf.v4.new_code_cell(imports))

# Add markdown cell for Step 1
nb.cells.append(nbf.v4.new_markdown_cell("## Step 1: Fetch Tweets\n\nFirst, let's fetch tweets using our Twitter API script."))

# Add code cell for fetching tweets
fetch_tweets = '''# Set up headers with your bearer token
bearer_token = os.environ.get("X_BEARER_TOKEN")
headers = request_headers(bearer_token)

# Fetch tweets
tweets = get_all_tweets(
    "https://api.twitter.com/2/tweets/search/recent",
    headers,
    default_query_parameters,
    max_pages=2  # Adjust this number based on your needs
)

print(f"Fetched {len(tweets)} tweets")'''
nb.cells.append(nbf.v4.new_code_cell(fetch_tweets))

# Add markdown cell for Step 2
nb.cells.append(nbf.v4.new_markdown_cell("## Step 2: Convert to DataFrame\n\nLet's convert the tweets data into a pandas DataFrame for easier analysis."))

# Add code cell for DataFrame conversion
df_conversion = '''# Convert to DataFrame
df = pd.DataFrame(tweets)

# Display basic information
print("DataFrame Info:")
df.info()

print("\\nFirst few tweets:")
df.head()'''
nb.cells.append(nbf.v4.new_code_cell(df_conversion))

# Add markdown cell for Step 3
nb.cells.append(nbf.v4.new_markdown_cell("## Step 3: Basic Analysis\n\nLet's perform some basic analysis on the tweets."))

# Add code cell for basic analysis
analysis = '''# Convert created_at to datetime
df['created_at'] = pd.to_datetime(df['created_at'])

# Extract engagement metrics
df['retweet_count'] = df['public_metrics'].apply(lambda x: x['retweet_count'])
df['reply_count'] = df['public_metrics'].apply(lambda x: x['reply_count'])
df['like_count'] = df['public_metrics'].apply(lambda x: x['like_count'])
df['quote_count'] = df['public_metrics'].apply(lambda x: x['quote_count'])

# Display basic statistics
print("Engagement Statistics:")
df[['retweet_count', 'reply_count', 'like_count', 'quote_count']].describe()'''
nb.cells.append(nbf.v4.new_code_cell(analysis))

# Add markdown cell for Step 4
nb.cells.append(nbf.v4.new_markdown_cell("## Step 4: Visualization\n\nLet's create some visualizations to better understand the data."))

# Add code cell for visualizations
visualizations = '''# Set style
plt.style.use('seaborn')

# Create a figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Tweet timeline
df.set_index('created_at').resample('D').size().plot(ax=axes[0,0], title='Tweets per Day')

# Plot 2: Engagement metrics
engagement_metrics = ['retweet_count', 'reply_count', 'like_count', 'quote_count']
df[engagement_metrics].boxplot(ax=axes[0,1])
axes[0,1].set_title('Engagement Metrics Distribution')
axes[0,1].tick_params(axis='x', rotation=45)

# Plot 3: Correlation heatmap
sns.heatmap(df[engagement_metrics].corr(), annot=True, ax=axes[1,0])
axes[1,0].set_title('Engagement Metrics Correlation')

# Plot 4: Top 10 most liked tweets
top_liked = df.nlargest(10, 'like_count')[['text', 'like_count']]
top_liked.plot(kind='barh', x='text', y='like_count', ax=axes[1,1])
axes[1,1].set_title('Top 10 Most Liked Tweets')

plt.tight_layout()
plt.show()'''
nb.cells.append(nbf.v4.new_code_cell(visualizations))

# Add markdown cell for Step 5
nb.cells.append(nbf.v4.new_markdown_cell("## Step 5: Save Processed Data\n\nFinally, let's save our processed data for future use."))

# Add code cell for saving data
save_data = '''# Save to CSV
df.to_csv('processed_tweets.csv', index=False)
print("Data saved to processed_tweets.csv")'''
nb.cells.append(nbf.v4.new_code_cell(save_data))

# Write the notebook to a file
with open('DataAnalysis/notebooks/twitter_data_analysis.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook created successfully!") 