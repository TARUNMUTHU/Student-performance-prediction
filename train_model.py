import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Load dataset
df = pd.read_csv('data/student-mat.csv', sep=';')

# ✅ Only these features are used (no 'school')
selected_features = [
    'sex', 'age', 'address', 'famsize', 'Pstatus',
    'Medu', 'Fedu', 'studytime', 'failures', 'absences', 'G1', 'G2'
]

# Create binary label (pass = 1 if G3 ≥ 10)
df['G3_binary'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)

# Keep only the selected features + target
df = df[selected_features + ['G3_binary']]

# Label encode categorical columns
encoders = {}
for col in ['sex', 'address', 'famsize', 'Pstatus']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Prepare training data
X = df[selected_features]
y = df['G3_binary']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save model and encoders
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/model.pkl')
joblib.dump(encoders, 'model/label_encoders.pkl')
print("✅ Model and encoders saved.")
