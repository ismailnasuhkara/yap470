import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, QuantileTransformer

pd.set_option('display.max_columns', 23)
df = pd.read_csv('diabetes_binary_5050split_health_indicators_BRFSS2023.csv')
df.loc[df['KidneyDisease'] == 2, 'KidneyDisease'] = 0
df.loc[df['Asthma'] == 2, 'Asthma'] = 0
df.loc[df['COPD'] == 2, 'COPD'] = 0

df = df[df['KidneyDisease'].isin([0.0,1.0])].reset_index(drop=True)
df = df[df['Asthma'].isin([0.0,1.0])].reset_index(drop=True)
df = df[df['COPD'].isin([0.0,1.0])].reset_index(drop=True)

target_col = "Diabetes_binary"

y = df[target_col]
X = df.drop(columns=[target_col])

nominal_cols = ['KidneyDisease','HighBP','HighChol','CholCheck','Asthma','COPD','Smoker','Stroke','HeartDiseaseorAttack','HvyAlcoholConsump','AnyHealthcare','NoDocbcCost','DiffWalk','Sex',]
ordinal_cols = ['GenHlth','MentHlth','PhysHlth','AgeGroup','Education','Income']
bmi_col = ['BMI']

ColTrans = ColumnTransformer(
        transformers= [
            ("ordinal_scaler",RobustScaler(), ordinal_cols),
            ("bmi_scaler", QuantileTransformer(output_distribution='uniform'), bmi_col),
            ("pass_nominal", "passthrough", nominal_cols)
        ],
        remainder="passthrough"
    )

df_transformed = ColTrans.fit_transform(X)

feature_names = ColTrans.get_feature_names_out()

df_transformed = pd.DataFrame(
    df_transformed,
    columns=feature_names
)

df_transformed[target_col] = y.values


df_transformed.to_csv("dataset_preprocessed.csv", index=False)

