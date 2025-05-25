import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def analyze_sentiment(text):
    # Initialize the VADER sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()
    
    # Get the sentiment scores
    scores = analyzer.polarity_scores(text)
    
    # Return the compound score which ranges from -1 to 1
    return scores['compound']

# Example usage
""" text = "I love this product! It's amazing and works perfectly."
sentiment_score = analyze_sentiment(text)
print(f"Sentiment score (-1 to 1): {sentiment_score}") """
#%%
def input_text(path):
    data=pd.read_excel(path,sheet_name='Sheet1')
    data=data.iloc[:,1:3].dropna()
    return data
#%%
currency='eur'
year=2019
path='Users/renee/Downloads/usd'+currency+"/"+str(year)+currency+'_r'+'.xlsx'
data=input_text(path)
sentiment_scores=[]
for i in data['news']:
    sentiment_score=analyze_sentiment(i)
    sentiment_scores.append(sentiment_score)
data['sentiment_score']=sentiment_scores
data.to_excel('Users/renee/Downloads/usd'+currency+"/"+str(year)+currency+'_r'+'_sentiment'+'.xlsx',index=False)
#%%
# Group by date and calculate mean sentiment score
daily_sentiment = data.groupby('date')['sentiment_score'].mean().reset_index()
print("Daily averaged sentiment scores:")
print(daily_sentiment)
# Convert date to datetime if it's not already
daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])

# Get the week information
daily_sentiment['week'] = daily_sentiment['date'].dt.isocalendar().week
daily_sentiment['year'] = daily_sentiment['date'].dt.isocalendar().year

# Handle first week if less than 3 days
first_week = daily_sentiment[daily_sentiment['week'] == daily_sentiment['week'].min()]
if len(first_week) < 3:
    # Merge first week with second week
    daily_sentiment.loc[daily_sentiment['week'] == daily_sentiment['week'].min(), 'week'] = \
        daily_sentiment['week'].min() + 1

# Calculate weekly bounds
weekly_sentiment = []
for (yr, wk), group in daily_sentiment.groupby(['year', 'week']):
    pos_scores = group[group['sentiment_score'] > 0]['sentiment_score']
    neg_scores = group[group['sentiment_score'] < 0]['sentiment_score']
    
    upper = pos_scores.mean() if len(pos_scores) > 0 else 0
    lower = neg_scores.mean() if len(neg_scores) > 0 else 0
    
    # Get the last date of the week (Sunday)
    week_end = group['date'].max()
    
    weekly_sentiment.append({
        'week_number': wk,
        'week_end_date': week_end,
        'upperbound': upper,
        'lowerbound': lower
    })

weekly_df = pd.DataFrame(weekly_sentiment)
print("\nWeekly sentiment bounds:")
print(weekly_df)
weekly_df.to_excel('Users/renee/Downloads/usd'+currency+"/"+str(year)+currency+'_r'+'_sentiment'+'.xlsx',index=False)
#%%


