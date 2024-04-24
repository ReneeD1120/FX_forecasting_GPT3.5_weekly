## Step 1: Topic classification based on ChatGPT3.5
Data source: googlenews
Prompt:

![Screenshot of a comment on a GitHub issue showing an image, added in the Markdown, of an Octocat smiling and raising a tentacle.](https://github.com/ReneeD1120/FX_forecasting_GPT3.5_weekly/blob/main/Prompt.png)

Sample of results of Topic classification based on ChatGPT:
```
import pandas as pd

# Create a dictionary with the data
data = {
    "Topic": ["Euro-debt issuance rising in prominence",
              "Pound to US Dollar exchange rate reaches 4-month highs",
              "Coinbase granted e-money license in Ireland",
              "Singapore Dollar Seen Sliding as Central Bank Faces Downturn",
              "Majority Disagree with Sale of Maltese Citizenship",
              "Pound-Euro Exchange Rate Crashes to Month Low",
              "Weak German Imports Reinforce Recession Fears",
              "UK Stocks that Benefit from a Weak Pound",
              "Global Banks and Synthetic Funding",
              "Top 5 Most-Traded Forex Pairs in Summer 2019 and Beyond"],
    "Sentiment Score": [60, 70, 50, -70, -30, -80, -60, 40, -50, 20],
    "Importance Score": [0.920, 0.890, 0.870, 0.780, 0.710, 0.660, 0.590, 0.550, 0.480, 0.410]
}

# Create DataFrame
df = pd.DataFrame(data)

# Print DataFrame
print(df)
```
## Step 2: Constructing the upper and lower bounds of TIS index

```
def weekly_score(df):
  transfer=MinMaxScaler(feature_range=(0,1))
  df_1=np.array(df.iloc[:,2]).reshape(-1, 1)
  df_1=transfer.fit_transform(df_1)
  d=np.array(df.iloc[:,1])*df_1.reshape(1, -1)
  upper=int(sum(np.maximum(d,0).T))
  lower=int(sum(np.minimum(d,0).T))
  return upper,lower


upper,lower=weekly_score(df)
bound.append([upper,lower])
raw_data.append(df)
print(bound)
```
## Step 3: Interval Forecasting
1. Interval Multilayer Perceptron (IMLP)
2. Interval Extreme Learning Machine (IELM)
3. Autoregressive conditional interval (ACI) model
4. Threshold autoregressive interval (TARI) model

## Step 4: High-frequency event analysis
Model: Bertopic 
Weekly Google News - Weekly TEN pertinent topics -Quarterly/Annual pertinent topics
```
year=2023
model = BERTopic(verbose=True,top_n_words =5,min_topic_size = 2)
topics=topics.iloc[:,1:11]
#convert to list
#docs = list(topics)
topic_list=[]
for i in range(len(topics)):
  l=list(topics.iloc[i,:].T)
  for j in range(len(l)):
    topic_list.append(l[j])
ts, probabilities = model.fit_transform(topic_list)
model.get_topic_freq().head(10)
```
