#%%
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler
#%%
# used_column: a list of strings specifying which columns one wants to use
# replaceUni: decide whether to fill the missing values of the enrolled_university column
# shouldScale: decide whether to scale numeric values or not
def load_and_process(used_columns: list, replaceUni:bool=True, shouldScale:bool=False) -> pd.DataFrame:
    # read data from csv
    df = pd.read_csv("csv/train_input.csv")

    # list if columns that, if called for should be encoded
    TO_ONEHOT = {'gender', 'major_discipline', 'company_type', 'city'}
    TO_ORDINAL = {'relevent_experience', 'enrolled_university', 'education_level', 'experience', 'company_size', 'last_new_job'}

    # initialize encoders and scaler
    ordinal_encoder = OrdinalEncoder()
    one_hot_encoder = OneHotEncoder()
    scaler = MinMaxScaler()


    if 'enrolled_university' in used_columns and replaceUni:
        df['enrolled_university'] = df['enrolled_university'].fillna('no_enrollment')

    # remove rows that are not called for, or contains missing values
    df = df[used_columns]
    df = df.dropna(axis=0)

    # encode columns according to the above defined lists
    for to_encode in used_columns:
        if to_encode in TO_ONEHOT:
            res = one_hot_encoder.fit_transform(df[[to_encode]])
            df = df.drop(to_encode, axis=1)
            df = df.reset_index(drop=True)
            # there are multiple onehotencoded values that start os at "Other", the original column name is concatenated so resolve the conflict
            df = df.join(pd.DataFrame(res.toarray(), columns = [c_value + '_' + to_encode if c_value == 'Other' else c_value for c_value in one_hot_encoder.categories_[0]]))
        elif to_encode in TO_ORDINAL:
            df[to_encode] = ordinal_encoder.fit_transform(df[[to_encode]])

    # the same min max scaler is used everywhere temporarily, may change during further development
    if shouldScale:
        for index, max_value in df.max().iteritems():
            if max_value > 1 and index != 'enrollee_id':
                df[index] = scaler.fit_transform(df[[index]])

    return df
