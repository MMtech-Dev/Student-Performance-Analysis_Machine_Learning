from sklearn.ensemble import RandomForestClassifier
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
    # Random Forest model
    rf_model = RandomForestClassifier(random_state=16, n_estimators=100)
    rf_model.fit(X_train, y_train)

    # Predictions
    y_pred = rf_model.predict(X_test)
    print(f"\nPrediction result: {y_pred}")

    # Evaluation
    print("\n=== RANDOM FOREST RESULTS ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 Most Important Features:\n{feature_importance.head(10)}")
    print(f"\nFeature Importance > 0.05:\n{feature_importance[feature_importance['importance'] > 0.05]}")
