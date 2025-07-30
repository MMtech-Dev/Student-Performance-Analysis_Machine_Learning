import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# load and clean data
def load_clean_student_Assessment():

    df = pd.read_csv("data.csv")

    # Data quality assessment
    print(f"Shape: {df.shape}")
    print(f"Data types: {df.dtypes}")
    print(f"Duplicates: {df.duplicated()}")
    print(f"Missing data: {df.isnull().sum()}")

    # Data cleaning
    df_clean = df.dropna(how="all")
    df_clean = df_clean.dropna(subset=["score"]) 
    print(df_clean['id_student'].nunique())

    return df_clean  

# Visualization dashboard
def analyse_studentAssessment():    
    df = load_clean_student_Assessment()

    # Visualisation
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Score distribution
    df['score'].hist(bins=30, ax=axes[0,0], edgecolor='black')
    axes[0,0].set_title('Score Distribution')
    axes[0,0].set_xlabel('Score')

    # Pass/Fail analysis
    pass_fail = (df['score'] >= 40).value_counts()
    pass_fail.plot(kind='bar', ax=axes[0,1])
    axes[0,1].set_title('Pass/Fail Distribution')

    # Banking status
    df['is_banked'].value_counts().plot(kind='bar', ax=axes[0,2])
    axes[0,2].set_title('Banked vs New Assessments')

    # Submission timing
    df['date_submitted'].hist(bins=30, ax=axes[1,0], edgecolor='black')
    axes[1,0].set_title('Submission Timeline')
    axes[1,0].set_xlabel('Days from Course Start')

    # Score vs submission timing
    axes[1,1].scatter(df['date_submitted'], df['score'], alpha=0.3)
    axes[1,1].set_title('Score vs Submission Timing')
    axes[1,1].set_xlabel('Days from Start')
    axes[1,1].set_ylabel('Score')

    # Score by banking status
    sns.boxplot(data=df, x='is_banked', y='score', ax=axes[1,2])
    axes[1,2].set_title('Score Distribution by Banking Status')

    plt.tight_layout()

    return plt.show()

    

# Execute analysis
df_student_assessment = load_clean_student_Assessment()
studentAssessment = analyse_studentAssessment()

print(df_student_assessment)
print(studentAssessment)


