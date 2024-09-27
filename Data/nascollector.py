import pandas as pd
import os

def list_wikipedia_nasdaq100() -> pd.DataFrame:
    url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
    return pd.read_html(url, attrs={'id': "constituents"}, index_col='Ticker')[0]

# Retrieve the Nasdaq-100 data
df = list_wikipedia_nasdaq100()

# Define directory and file path
directory = os.path.expanduser("~/.quantlib/data/nasdaq100")
if not os.path.exists(directory):
    os.makedirs(directory)

file_path = os.path.join(directory, "nasdaq100.csv")

# Save the DataFrame to CSV with only Company and Ticker
df[['Company']].to_csv(file_path)
print(f"CSV saved to {file_path}")
