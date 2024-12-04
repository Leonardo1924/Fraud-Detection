import pandas as pd

# Load the data
transactions = pd.read_csv("Data/transactions.csv")
merchants = pd.read_csv("Data/merchants.csv")
customers = pd.read_csv("Data/customers.csv")
cities = pd.read_csv("Data/cities.csv")

# Join the data
data = (transactions
        .merge(merchants, on="merchant", how="left")
        .merge(customers, on="cc_num", how="left")
        .merge(cities, on="city", how="left"))

# Save the data
data.to_csv("Data/data.csv", index=False)