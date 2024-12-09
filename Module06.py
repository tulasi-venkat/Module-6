import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

file_path = 'Final_Report_of_the_Asian_American_Quality_of_Life__AAQoL_.csv'
data = pd.read_csv(file_path)

satisfaction_mapping = {
    "Strongly disagree": 1, "Disagree": 2, "Slightly disagree": 3,
    "Neither agree or disagree": 4, "Slightly agree": 5, "Agree": 6, "Strongly agree": 7
}
mental_health_mapping = {
    "Poor": 1, "Fair": 2, "Good": 3, "Very Good": 4, "Excellent": 5
}
ethnic_identity_mapping = {
    "Not at all": 1, "Not very close": 2, "Somewhat close": 3, "Very close": 4
}

data['Life_Satisfaction'] = data['Satisfied With Life 1'].map(satisfaction_mapping)
data['Mental_Health'] = data['Present Mental Health'].map(mental_health_mapping)
data['Ethnic_Identity'] = data['Identify Ethnically'].map(ethnic_identity_mapping)
data['Discrimination'] = data['Discrimination ']

def categorize_mental_health(value):
    if value in [1, 2]:
        return 'Low'
    elif value == 3:
        return 'Moderate'
    else:
        return 'High'

data['Mental_Health_Label'] = data['Mental_Health'].apply(categorize_mental_health)

features = ['Life_Satisfaction', 'Ethnic_Identity', 'Discrimination', 'Age', 'Marital Status']
data = data.dropna(subset=features + ['Mental_Health_Label'])
X = pd.get_dummies(data[features], drop_first=True)
y = data['Mental_Health_Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

conf_matrix = confusion_matrix(y_test, y_pred, labels=['Low', 'Moderate', 'High'])
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'Moderate', 'High'], yticklabels=['Low', 'Moderate', 'High'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

print("Classification Report:\n", classification_report(y_test, y_pred))

feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title("Feature Importances")
plt.show()

wrong_predictions = X_test[y_test != y_pred]
wrong_predictions['True_Label'] = y_test[y_test != y_pred]
wrong_predictions['Predicted_Label'] = y_pred[y_test != y_pred]
print("Incorrectly Classified Samples:")
print(wrong_predictions.head(5))
