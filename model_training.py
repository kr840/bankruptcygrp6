#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("Data.csv")  # Ensure the file is in the same folder

# Convert categorical target to numeric
df["class"] = df["class"].map({"non-bankruptcy": 0, "bankruptcy": 1})

# Prepare data
X = df.drop(columns=["class"])
y = df["class"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save model and scaler
joblib.dump(model, "bankruptcy_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model training complete. Files saved.")


# In[ ]:





# In[ ]:




