#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler
from sklearn.decomposition import PCA
#%%
# used_column: a list of strings specifying which columns one wants to use
# replaceUni: decide whether to fill the missing values of the enrolled_university column
# shouldScale: decide whether to scale numeric values or not
def load_and_process(train_path: str, test_path: str ,used_columns: list, replaceUni:bool=True, shouldScale:bool=False, shouldDropNA:bool=True, PCAtarget:int = 0) -> tuple:
    # read data from csv
    df = pd.read_csv(train_path)
    tdf = pd.read_csv(test_path)
    if'test_input.csv' in train_path:
        df['target'] = np.load("csv/test_input_target.npy")
    elif 'test_input.csv' in test_path:
        tdf['target'] = np.load("csv/test_input_target.npy")

    # list if columns that, if called for should be encoded
    TO_ONEHOT = {'gender', 'major_discipline', 'company_type', 'city'}
    TO_ORDINAL = {'relevent_experience', 'enrolled_university', 'education_level', 'experience', 'company_size', 'last_new_job'}

    # initialize encoders and scaler
    ordinal_encoder = OrdinalEncoder()
    one_hot_encoder = OneHotEncoder()
    scaler = MinMaxScaler()
    pca = PCA(n_components=PCAtarget)

    if 'enrolled_university' in used_columns and replaceUni:
        df['enrolled_university'] = df['enrolled_university'].fillna('no_enrollment')
        tdf['enrolled_university'] = tdf['enrolled_university'].fillna('no_enrollment')

    # remove rows that are not called for, or contains missing values
    df = df[used_columns]
    tdf = tdf[used_columns]
    if shouldDropNA:
        df = df.dropna(axis=0)
        tdf = tdf.dropna(axis=0)

        # encode columns according to the above defined lists
        for to_encode in used_columns:
            if to_encode in TO_ONEHOT:
                res = one_hot_encoder.fit_transform(df[[to_encode]])
                df = add_encoded_info(df, res, to_encode, one_hot_encoder)
                res = one_hot_encoder.transform(tdf[[to_encode]])
                tdf = add_encoded_info(tdf, res, to_encode, one_hot_encoder)
                
            elif to_encode in TO_ORDINAL:
                df[to_encode] = ordinal_encoder.fit_transform(df[[to_encode]])
                tdf[to_encode] = ordinal_encoder.transform(tdf[[to_encode]])

        # the same min max scaler is used everywhere temporarily, may change during further development
        if shouldScale:
            for index, max_value in df.max().iteritems():
                if max_value > 1 and index != 'enrollee_id':
                    df[index] = scaler.fit_transform(df[[index]])
                    tdf[index] = scaler.transform(tdf[[index]])
        elif PCAtarget > 0:
            y_train = df['target']
            y_test = tdf['target']
            df = pca.fit_transform(df.drop('target', axis=1))
            tdf = pca.fit_transform(tdf.drop('target', axis=1))
            return (df, y_train, tdf, y_test)

    return (df, tdf)
#%%
# merges the two test data files into one data table
def merge_test_input():
    test_target = np.load("csv/test_input_target.npy")
    test_rest = pd.read_csv("csv/test_input.csv")
    return test_rest.join(pd.DataFrame(test_target, columns=["target"]))
# %%
# splits the input training data into training and development parts
def split_train_data(data:pd.DataFrame):
    return (data.head(2000),data.tail(-2000))
# %%

def add_encoded_info(df,res,to_encode,one_hot_encoder):
    df = df.drop(to_encode, axis=1)
    df = df.reset_index(drop=True)
    # there are multiple onehotencoded values that start os at "Other", the original column name is concatenated so resolve the conflict
    df = df.join(pd.DataFrame(res.toarray(), columns = [c_value + '_' + to_encode if c_value == 'Other' else c_value for c_value in one_hot_encoder.categories_[0]]))
    return df
