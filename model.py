import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('Crop_recommendation.csv')

# Split the dataset into input features (X) and target variable (y)
X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = data['label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train all models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
with open('random_forest_model.pkl', 'wb') as file:
    pickle.dump(rf_model, file)

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
with open('decision_tree_model.pkl', 'wb') as file:
    pickle.dump(dt_model, file)

# Using linear kernel for SVM to speed up training
svm_model = SVC(kernel='linear', probability=True, random_state=42)
svm_model.fit(X_train, y_train)
with open('svm_model.pkl', 'wb') as file:
    pickle.dump(svm_model, file)

knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
with open('knn_model.pkl', 'wb') as file:
    pickle.dump(knn_model, file)

logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)
with open('logistic_model.pkl', 'wb') as file:
    pickle.dump(logistic_model, file)

# Save model accuracies
model_accuracies = {
    'random_forest': rf_model.score(X_test, y_test),
    'decision_tree': dt_model.score(X_test, y_test),
    'svm': svm_model.score(X_test, y_test),
    'knn': knn_model.score(X_test, y_test),
    'logistic': logistic_model.score(X_test, y_test)
}
with open('model_accuracies.pkl', 'wb') as file:
    pickle.dump(model_accuracies, file)

print("Models trained and saved successfully.")
