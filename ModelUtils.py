import os
import time
import getpass
import numpy as np
import pandas as pd
import textdistance as td

align_width = 18


def is_category(dtype):
    return pd.api.types.is_categorical_dtype(dtype)


def get_obj_cols(df):
    """
    Returns names of 'object' columns in the DataFrame.
    """
    obj_cols = []
    for idx, dt in enumerate(df.dtypes):
        if dt == 'object' or is_category(dt):
            obj_cols.append(df.columns.values[idx])

    return obj_cols


def pandas_datetime_cols(data):
    data = data.infer_objects()
    obj_features = get_obj_cols(data)
    for feature in obj_features:
        dates = data[feature]
        if pd.core.dtypes.common.is_datetime64_dtype(dates):
            continue

        data[feature] = pd.to_datetime(dates, infer_datetime_format=True, errors='ignore')

    return data


def get_username():
    return getpass.getuser()


def get_current_dir():
    return os.path.dirname(os.path.abspath(__file__))


def classCount(classes, labels):
    num_c = len(classes)
    num_l = len(labels)
    c = np.full((num_l, num_c), False)
    for i in range(num_c):
        c[:, i] = (labels == classes[i])

    return c


def print_path(full_path):
    return full_path.replace(get_current_dir(), '$MODELDIR')


def log_writer(my_string):
    my_string = '[%s]: %s' % (time.strftime('%H:%M:%S'), my_string)
    print(my_string)


def clean_string(string):
    string = string.replace('\n', ' ')
    string = string.rstrip()
    replacements = ['\'', '"']
    for char in replacements:
        string = string.replace(char, '')

    while '  ' in string:
        string = string.replace('  ', ' ')

    return string


def ismember(a_vec, b_vec, full_match=True):
    """ MATLAB equivalent ismember function """
    if isinstance(a_vec, list):
        a_vec = np.array(a_vec)

    if not full_match:
        func = lambda x: ','.join(np.intersect1d(x.replace('', ' ').split(','), a_vec))
        b_vec = pd.Series(b_vec).apply(func).values

    bool_idx = np.isin(a_vec, b_vec)
    common = a_vec[bool_idx]
    common_unique, common_inv = np.unique(common, return_inverse=True)
    b_unique, b_idx = np.unique(b_vec, return_index=True)
    common_idx = b_idx[np.isin(b_unique, common_unique, assume_unique=True)]

    return bool_idx, common_idx[common_inv]


def shift_array(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result = arr
    return result


def get_most_similar(string, df):
    idx = df.apply(lambda x: td.levenshtein.normalized_similarity(string, x)).idxmax()

    return df[idx]


def fill_missing_with_similar(df, dct):
    categories = pd.Series(dct.keys())
    idx = ~df.isin(categories)
    new_df = df[idx].apply(lambda x: get_most_similar(x, categories))
    new_df.index = df[idx]
    map_missing = new_df.to_dict()

    return map_missing


class OrdinalEncoder:
    def __init__(self, mapping=None, feature_names=None):
        self.mapping = mapping
        self.feature_names = feature_names

    def fit(self, X_in):
        X = X_in.copy(deep=True)

        if self.feature_names is None:
            self.feature_names = get_obj_cols(X)

        idx, idxF = ismember(self.feature_names, X.columns.values)
        if idx.sum() == 0:
            log_writer('! None of the column(s): %s, exist(s) in the dataframe' % self.feature_names)
            return self

        if idx.sum() < len(idx):
            log_writer('! Some columns do not exist in the dataframe')

        self.feature_names = X.columns[idxF].tolist()
        log_writer('Here are the categorical features to be fitted: %s' % ', '.join(self.feature_names))
        X = self.ordinal_encoding(X)

        return self

    def transform(self, X_in):
        X = X_in.copy(deep=True)

        if (self.mapping is None) or (self.feature_names is None):
            log_writer('! Must train encoder before it can be used to transform data')
            return X

        idx, idxF = ismember(self.feature_names, X.columns.values)
        if idx.sum() == 0:
            log_writer('! None of the column(s): %s, exist(s) in the dataframe' % self.feature_names)
            return X

        if idx.sum() < len(idx):
            log_writer('! Some columns do not exist in the dataframe')

        self.feature_names = X.columns[idxF].tolist()

        X = self.ordinal_encoding(X)

        return X

    def inverse_transform(self, X_in):
        X = X_in.copy(deep=True)

        if (self.mapping is None) or (self.feature_names is None):
            log_writer('! Must train encoder before it can be used to transform data')
            return X

        idx, idxF = ismember(self.feature_names, X.columns.values)
        if idx.sum() == 0:
            log_writer('! None of the column(s): %s, exist(s) in the dataframe' % self.feature_names)
            return X

        if idx.sum() < len(idx):
            log_writer('! Some columns do not exist in the dataframe')

        self.feature_names = X.columns[idxF].tolist()

        for switch in self.mapping:
            column_mapping = switch.get('mapping')
            inverse = pd.Series(data=column_mapping.index, index=column_mapping.get_values())
            X[switch.get('col')] = X[switch.get('col')].map(inverse).astype(switch.get('data_type'))

        return X

    def ordinal_encoding(self, X_in):
        X = X_in.copy(deep=True)

        if (self.mapping is not None) or ():
            for switch in self.mapping:
                column = switch.get('col')
                df_unique = pd.Series(X[column].unique())
                map_missing = fill_missing_with_similar(df_unique, switch['mapping'])
                X[column] = X[column].replace(map_missing)
                X[column] = X[column].map(switch['mapping'])

                try:
                    X[column] = X[column].astype(int)
                except ValueError as e:
                    X[column] = X[column].astype(float)
        else:
            mapping_out = []
            for feature in self.feature_names:
                if is_category(X[feature].dtype):
                    categories = X[feature].cat.categories
                else:
                    categories = X[feature].unique()

                index = pd.Series(categories).unique()
                data = pd.Series(index=index, data=range(1, len(index) + 1))
                mapping_out.append({'col': feature, 'mapping': data, 'data_type': X[feature].dtype}, )

            self.mapping = None if len(mapping_out) == 0 else mapping_out

        return X
