import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load, clean data
def load_clean_studentVle():

    df = pd.read_csv("data.csv")
    # Data quality check
    print(f"Shape: {df.shape}")
    print(f"Data type: {df.dtypes}")
    print(f"Duplicated: {df.duplicated()}")
    print(f"Missing data: {df.isnull().sum()}")

    # Data cleaning
    df_clean = df.drop_duplicates()
    print(f"Shape after cleaning: {df_clean.shape}")

    return df_clean

# Data visualization 
def analyse_studentVle():  
    df = load_clean_studentVle()

    # Visualisation
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Click distribution (log scale for better view)
    df['sum_click'].hist(bins=50, ax=axes[0,0], edgecolor='black')
    axes[0,0].set_title('Daily Click Distribution')
    axes[0,0].set_xlabel('Clicks per Day')
    axes[0,0].set_yscale('log')  # Many students have low clicks
    
    # 2. Engagement over time (timeline)
    df['date'].hist(bins=50, ax=axes[0,1], edgecolor='black')
    axes[0,1].set_title('Student Activity Timeline')
    axes[0,1].set_xlabel('Days from Course Start')
    
    # 3. Most active students (top 20)
    top_students = df.groupby('id_student')['sum_click'].sum().nlargest(20)
    top_students.plot(kind='bar', ax=axes[0,2])
    axes[0,2].set_title('Top 20 Most Active Students')
    axes[0,2].set_ylabel('Total Clicks')
    
    # 4. Daily engagement pattern
    daily_engagement = df.groupby('date')['sum_click'].sum()
    axes[1,0].plot(daily_engagement.index, daily_engagement.values)
    axes[1,0].set_title('Daily Course Engagement')
    axes[1,0].set_xlabel('Days from Start')
    axes[1,0].set_ylabel('Total Clicks')
    
    # 5. Most popular materials
    popular_sites = df.groupby('id_site')['sum_click'].sum().nlargest(15)
    popular_sites.plot(kind='bar', ax=axes[1,1])
    axes[1,1].set_title('Top 15 Most Accessed Materials')
    axes[1,1].set_ylabel('Total Clicks')
    
    # 6. Student engagement levels
    student_engagement = df.groupby('id_student')['sum_click'].sum()
    student_engagement.hist(bins=50, ax=axes[1,2], edgecolor='black')
    axes[1,2].set_title('Student Engagement Distribution')
    axes[1,2].set_xlabel('Total Clicks per Student')
    axes[1,2].set_yscale('log')
    
    # Summary stats
    print(f"\n=== VLE ENGAGEMENT SUMMARY ===")
    print(f"Total interactions: {len(df):,}")
    print(f"Unique students: {df['id_student'].nunique():,}")
    print(f"Unique materials: {df['id_site'].nunique():,}")
    print(f"Average clicks per interaction: {df['sum_click'].mean():.1f}")
    print(f"Course duration: {df['date'].min()} to {df['date'].max()} days")
    
    plt.tight_layout()

    return plt.show()

# Execute analysis
df_studentVle = load_clean_studentVle()
studentVle_visualisation = analyse_studentVle

print(df_studentVle)
print(studentVle_visualisation)