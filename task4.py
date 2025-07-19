# iris_model.py

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle

# 1. Load dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# 2. Split dataset
X = df.iloc[:, :-1]   # features
y = df.iloc[:, -1]    # target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# 4. Evaluate model
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# 5. Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# 6. Predict new data
sample = [[5.1, 3.5, 1.4, 0.2]]  # Example input
loaded_model = pickle.load(open('model.pkl', 'rb'))
prediction = loaded_model.predict(sample)
print("Prediction for sample:", iris.target_names[prediction[0]])
