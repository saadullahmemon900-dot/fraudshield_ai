import pandas as pd

df = pd.read_csv("credit_card_frauds.csv")

df.to_parquet("credit_card_frauds.parquet")

print("Parquet file created successfully!")