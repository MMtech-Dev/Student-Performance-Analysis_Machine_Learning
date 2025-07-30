import pandas as pd
from studentInfo import df_studentInfo
from studentAssessment import df_student_assessment
from student_virtualData import df_studentVle

def aggrerate_merge_data():

    # 1/ Base foundation from studentInfo data
    base_df = df_studentInfo.copy()

    # 2/ Assessment feature from studentAssessment data
    assessment_features = df_student_assessment.groupby('id_student').agg({
        'score': ['mean', 'std', 'count', 'min', 'max'],
        'date_submitted': ['mean', 'std'],  
        'is_banked': 'sum' 
    }).round(2)

    # flatten column names
    assessment_features.columns = ['avg_score', 'score_std', 'num_assessments',
                                'min_score', 'max_score', 'avg_submission_day',
                                'submission_consistency', 'banked_count']

    #=========================================================================

    # 3/ Engagement features from studentVle data
    engagement_features = df_studentVle.groupby('id_student').agg({
        'sum_click': ['sum', 'mean', 'std', 'count'],
        'date': ['min', 'max', 'nunique'], 
        'id_site': 'nunique' }).round(2)

    # Flatten columns
    engagement_features.columns = ['total_clicks', 'avg_clicks', 'click_consistency',
                                'active_days', 'first_access', 'last_access',
                                'days_active', 'materials_accessed']

    # Create derived features
    engagement_features["engagement_duration"] = engagement_features['last_access'] - engagement_features['first_access']
    engagement_features['clicks_per_material'] = engagement_features['total_clicks'] / engagement_features['materials_accessed'].replace(0, 1)

    #=========================================================================

    # 4/ Merge features 

    final_df = base_df.merge(assessment_features, left_on='id_student', right_index=True, how='left')
    final_df = final_df.merge(engagement_features, left_on='id_student', right_index=True, how='left')

    # print(f"Final dataset shape: {final_df.shape}")
    # print(f"Feature created: {final_df.columns.to_list()}")
    # print(f"Missing values: \n{final_df.isnull().sum()}")
    # print(f"\nTarget variable distribution: \n{final_df['final_result'].value_counts()}")

    # Clean final dataframe
    final_df['avg_score'] = final_df['avg_score'].fillna(0)
    final_df['score_std'] = final_df['score_std'].fillna(0)
    final_df['num_assessments'] = final_df['num_assessments'].fillna(0)
    final_df['min_score'] = final_df['min_score'].fillna(0)
    final_df['max_score'] = final_df['max_score'].fillna(0)
    final_df['avg_submission_day'] = final_df['avg_submission_day'].fillna(0)
    final_df['submission_consistency'] = final_df['submission_consistency'].fillna(0)
    final_df['banked_count'] = final_df['banked_count'].fillna(0)
    final_df['total_clicks'] = final_df['total_clicks'].fillna(0)
    final_df['avg_clicks'] = final_df['avg_clicks'].fillna(0)
    final_df['click_consistency'] = final_df['click_consistency'].fillna(0)
    final_df['active_days'] = final_df['active_days'].fillna(0)
    final_df['first_access'] = final_df['first_access'].fillna(0)
    final_df['last_access'] = final_df['last_access'].fillna(0)
    final_df['days_active'] = final_df['days_active'].fillna(0)
    final_df['materials_accessed'] = final_df['materials_accessed'].fillna(0)
    final_df['engagement_duration'] = final_df['engagement_duration'].fillna(0)
    final_df['clicks_per_material'] = final_df['clicks_per_material'].fillna(0)


    # Create binary flags 
    final_df['submitted_assessments'] = (final_df['num_assessments'] > 0).astype(int)
    final_df['used_vle'] = (final_df['total_clicks'] > 0).astype(int)

    print("Final dataset ready for modeling!")
    print(f"\nDataset Summary:")
    print(f"Shape: {final_df.shape}")
    print(f"Missing values: {final_df.isnull().sum().sum()}")
    print(f"Students with assessments: {final_df['submitted_assessments'].sum()}")
    print(f"Students with VLE activity: {final_df['used_vle'].sum()}")
    print(f"Target distribution:\n{final_df['final_result'].value_counts()}")

    return final_df

# Execute
df_agg_merge = aggrerate_merge_data()
print(df_agg_merge)