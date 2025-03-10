
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_excel("Bankruptcy.xlsx")  # Make sure this file is in the same folder

# Convert categorical target to numeric
df["class"] = df["class"].map({"non-bankruptcy": 0, "bankruptcy": 1})

# EDA: Class distribution
plt.figure(figsize=(6, 4))
sns.barplot(x=df["class"].value_counts().index, y=df["class"].value_counts(normalize=True) * 100, palette="viridis")
plt.title("Class Distribution (Bankruptcy vs. Non-Bankruptcy)")
plt.xlabel("Class (0 = Not Bankrupt, 1 = Bankrupt)")
plt.ylabel("Percentage (%)")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# Prepare data for training
X = df.drop(columns=["class"])
y = df["class"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save model and scaler
joblib.dump(model, "bankruptcy_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model training complete. Files saved as `bankruptcy_model.pkl` and `scaler.pkl`.")
