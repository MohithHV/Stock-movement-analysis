import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
import joblib

# Load preprocessed data
X_train = pd.read_csv('X_train.csv', header=0)
X_test = pd.read_csv('X_test.csv', header=0)

# Load the target variables from CSV files and specify no header
y_train_df = pd.read_csv('y_train.csv', header=None)
y_test_df = pd.read_csv('y_test.csv', header=None)

# Assign column names manually if they are not available
y_train_df.columns = ['movement']  # This should be the correct column name
y_test_df.columns = ['movement']  # This should also be the correct column name

# Select the correct column name dynamically
y_train = y_train_df['movement']
y_test = y_test_df['movement']

# Check for class imbalance
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Initialize and train the logistic regression model
model = LogisticRegression(random_state=42, solver='liblinear')  # Using 'liblinear' to support L1 penalty
model.fit(X_train, y_train)

# Predict stock movements on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(report)

# Save the trained model
joblib.dump(model, 'stock_movement_model.pkl')
print("Trained model saved as stock_movement_model.pkl!")
