from sklearn.linear_model import LogisticRegression
from Data_processing import aggregate_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd

if __name__ == "__main__":


    df = aggregate_data.df_agg_merge.copy()

    # Encode categorical variables
    categorical_cols = ['gender', 'region', 'highest_education', 'imd_band', 'age_band', 'disability']
    df_encoded = df.copy()

    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])

    # Exclude non-predictive columns
    exclude_cols = ['code_module', 'code_presentation', 'id_student', 'final_result']
    feature_cols = [col for col in df_encoded.columns if col not in exclude_cols]
    X = df_encoded[feature_cols] # Features
    y = df_encoded["final_result"] # Target value

    # Checking data before splitting data and modelling
    print(f"Dataset shape: {df_encoded.shape}")
    print(f"Features: {len(feature_cols)}")
    print(f"Target classes: {y.value_counts()}")


    # Split X and y into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16, stratify=y)

    # Instantiate the model
    logreg = LogisticRegression(random_state=16)

    # Fit the model with data
    logreg.fit(X_train, y_train)

    y_pred = logreg.predict(X_test)

    # Predictions
    print(f"Prediction result: {y_pred}")
    
    # Evaluation
    print("=== LOGISTIC REGRESSION RESULTS ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    # Feature importance (coefficients)
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': logreg.coef_[0] if len(logreg.coef_) == 1 else logreg.coef_.mean(axis=0)
    }).sort_values('coefficient', key=abs, ascending=False)
    print(f"\nTop 10 Most Important Features:\n{feature_importance.head(10)}")





