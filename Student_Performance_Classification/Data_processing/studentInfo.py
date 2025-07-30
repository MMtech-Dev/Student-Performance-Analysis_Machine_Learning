import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# load and clean data
def load_clean_studentInfo():
   # load data
    df = pd.read_csv("data.csv")

    # Data quality assessment
    print("=== STUDENTINFO DATA QUALITY ASSESSMENT ===")
    print(f"Shape: {df.shape}")
    print(f"Missing Values:\n{df.isnull().sum()}")
    print(f"Duplicates: {df.duplicated().sum()}")
    
    # Categorical summary  
    print(f"\nCategorical Summary:")
    for col in df.select_dtypes(include=['object']).columns:
        print(f"{col}: {df[col].nunique()} unique values")

    # Data cleaning
    df_clean = df.dropna(how="all")
    df_clean['imd_band'] = df_clean['imd_band'].fillna('Unknown')
    df_clean['imd_band'] = df_clean['imd_band'].replace('10-20', '10-20%')  # Fix parsing error
    
    print(f"\nCleaning Summary:")
    print(f"Final shape: {df_clean.shape}")
    print(f"IMD 'Unknown' entries: {(df_clean['imd_band'] == 'Unknown').sum()}")

    return df_clean


# Visualization dashboard
def analyse_studentInfo():
    df = load_clean_studentInfo()

    # Visualisation
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    df['final_result'].value_counts().plot(kind='bar', ax=axes[0,0])
    axes[0,0].set_title('Final Result Distribution')
    
    df['age_band'].value_counts().plot(kind='bar', ax=axes[0,1])
    axes[0,1].set_title('Age Band Distribution')
    
    pd.crosstab(df['gender'], df['final_result']).plot(kind='bar', ax=axes[1,0])
    axes[1,0].set_title('Gender vs Final Result')
    
    df['imd_band'].value_counts().plot(kind='bar', ax=axes[1,1])
    axes[1,1].set_title('IMD Band Distribution')
    
    plt.tight_layout()

    return plt.show()

# Execute analysis
df_studentInfo = load_clean_studentInfo()
studentInfo_visualisation = analyse_studentInfo()

print(df_studentInfo)
print(studentInfo_visualisation)