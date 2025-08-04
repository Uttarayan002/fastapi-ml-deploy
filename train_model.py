# train_model.py

import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Sample data (square footage and number of bedrooms)
data = pd.read_csv('house_pricing_dataset.csv')
df = pd.DataFrame(data)

X = df.values[:, :-1]
y = df.values[:, -1]

model = LinearRegression()
model.fit(X, y)

# Save the model
with open('house_price_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved successfully.")
