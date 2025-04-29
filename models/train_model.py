import numpy as np
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print("Loading data...")
# Load data from the Django project directory
csv_path = os.path.join(project_dir, 'Crop_recommendation.csv')
crop_data = pd.read_csv(csv_path)

crop_dict = {
    'rice': 1, 'maize': 2, 'jute': 3, 'cotton': 4, 'coconut': 5, 'papaya': 6, 'orange': 7,
    'apple': 8, 'muskmelon': 9, 'watermelon': 10, 'grapes': 11, 'mango': 12, 'banana': 13,
    'pomegranate': 14, 'lentil': 15, 'blackgram': 16, 'mungbean': 17, 'mothbeans': 18,
    'pigeonpeas': 19, 'kidneybeans': 20, 'chickpea': 21, 'coffee': 22
}

# Reverse mapping for display purposes
crop_names = {v: k for k, v in crop_dict.items()}

# Map the labels 
crop_data['label'] = crop_data['label'].map(crop_dict)

# Split into features and target
X = crop_data.drop('label', axis=1)
y = crop_data['label']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
mx = MinMaxScaler()
X_train_minmax = mx.fit_transform(X_train)
X_test_minmax = mx.transform(X_test)

sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train_minmax)
X_test_scaled = sc.transform(X_test_minmax)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Verify accuracy
accuracy = model.score(X_test_scaled, y_test)
print(f"Model accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Additional model evaluation
y_pred = model.predict(X_test_scaled)
print("\nClassification Report:")
target_names = [crop_names[i] for i in sorted(crop_names.keys())]
report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)

# Convert to DataFrame for better display
df_report = pd.DataFrame(report).transpose()
# Display only precision, recall, and f1-score (drop support)
print(df_report[['precision', 'recall', 'f1-score']].round(2))

# Evaluate feature importance
feature_importances = model.feature_importances_
feature_names = X.columns
print("\nFeature Importance:")
for feature, importance in sorted(zip(feature_names, feature_importances), key=lambda x: x[1], reverse=True):
    print(f"{feature}: {importance:.4f}")

# Save the model and scalers in the models directory
with open(os.path.join(current_dir, 'model.pkl'), 'wb') as f:
    pickle.dump(model, f)
    
with open(os.path.join(current_dir, 'minmax_scaler.pkl'), 'wb') as f:
    pickle.dump(mx, f)
    
with open(os.path.join(current_dir, 'standard_scaler.pkl'), 'wb') as f:
    pickle.dump(sc, f)

print("\nModel and scalers saved successfully to the Django models directory!") 