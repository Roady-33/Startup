# Model is gemaakt met behulp van ChatGPT
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib  # For saving the model

# Load the extracted features from CSV
csv_path = r"F:\3. Minor HU\Startup_met_code_werkend_model\Startup_met_code_werkend_model\Startup\Startup\features_with_dimensions.csv"
data = pd.read_csv(csv_path)

# Convert the 'scale' column from string format (e.g., "1:200") to a numeric scale factor
if 'scale' in data.columns:
    data['scale'] = data['scale'].apply(lambda x: float(x.split(':')[1]) if isinstance(x, str) and ':' in x else x)

# Separate features (X) and labels (y)
X = data.drop(columns=['label', 'image_path'])  # Drop non-feature columns
y = data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model
model_path = r"F:\3. Minor HU\Startup_met_code_werkend_model\Startup_met_code_werkend_model\Startup\Startup\trained_model.pkl"
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")
