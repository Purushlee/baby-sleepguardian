import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from features.utils import load_dataset
import numpy as np

# Load multi-class dataset
X, y = load_dataset('../dataset/multi_class')
X = np.array(X)
y = LabelEncoder().fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train multi-class model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Save model & scaler
joblib.dump(clf, '../models/multi_class_model.pkl')
joblib.dump(scaler, '../models/scaler_multi_class.pkl')

print("Multi-class model trained and saved.")
