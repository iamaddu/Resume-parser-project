import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

def train_cognitive_model(csv_path="synthetic_cognitive_resumes.csv"):
    df = pd.read_csv(csv_path)
    X = df.drop("success_label", axis=1)
    y = df["success_label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [3, 5, None]
    }
    clf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    return clf.best_estimator_

if __name__ == "__main__":
    model = train_cognitive_model()