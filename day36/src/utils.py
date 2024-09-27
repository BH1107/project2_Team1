import pandas as pd
from sklearn.preprocessing import LabelEncoder


def convert_category_into_integer(df: pd.DataFrame, columns: list):
    label_encoders = {}
    
    for column in columns:
        label_encoder = LabelEncoder()
        
        df.loc[:, column] = label_encoder.fit_transform(df[column])
        
        label_encoders.update({column: label_encoder})
    
    return df, label_encoders
