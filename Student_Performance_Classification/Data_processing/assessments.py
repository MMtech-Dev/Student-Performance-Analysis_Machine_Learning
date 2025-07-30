import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load and clean data
def load_clean_assessments():

    df = pd.read_csv("data.csv")

    # Data cleaning
    df_clean = df.dropna(how="all").ffill().drop_duplicates()

    # Summary statistics
    assessment_summary = {
        'Total assessments': len(df_clean),
        'Modules covered': df_clean['code_module'].nunique(),
        'Assessment types': df_clean['assessment_type'].value_counts().to_dict(),
        'Date range': f"{df_clean['date'].min()} to {df_clean['date'].max()} days",
        'Weight statistics': df_clean.groupby('assessment_type')['weight'].agg(['mean', 'std']).round(2)
    }
    print(assessment_summary)
    return df_clean


# Visualization dashboard
def analyse_assessments():
    df = load_clean_assessments()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Assessment type distribution
    df['assessment_type'].value_counts().plot(kind='bar', ax=axes[0])
    axes[0].set_title('Assessment Type Distribution')
    axes[0].set_xlabel('Assessment Type')

    # Weight distribution by type
    sns.boxplot(data=df, x='assessment_type', y='weight', ax=axes[1])
    axes[1].set_title('Weight Distribution by Type')

    # Assessment timeline
    sns.scatterplot(data=df, x='date', y='code_module', 
                    hue='assessment_type', size='weight', sizes=(50, 200), ax=axes[2])
    axes[2].set_title('Assessment Timeline')
    axes[2].set_xlabel('Days from Course Start')

    print("Assessment Dataset Summary:")
    for key, value in df.assessment_summary.items():
        if key != 'Weight statistics':
            print(f"{key}: {value}")
    print("\nWeight Statistics by Assessment Type:")
    print(df.assessment_summary['Weight statistics'])


    plt.tight_layout()
    
    return plt.show()

    

# Execute analysis
df_assessment = load_clean_assessments()
assessments_visualisation = analyse_assessments()
print(df_assessment)
print(assessments_visualisation)
