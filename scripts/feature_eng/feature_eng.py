import logging
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

def bin_age(C_AGE):
    """
    Categorize an individual's age into different age groups.

    Args:
        C_AGE (int): Age of the individual.

    Returns:
        str: Age group of the individual. Possible values are 'Young Adult', 'Middle-aged', 'Senior Citizen', or 'Minor'.

    Raises:
        None

    Example:
        age_group = bin_age(35)  # Returns 'Middle-aged'
    """
    if 18 <= C_AGE < 30:
        return 'Young Adult'
    elif 30 <= C_AGE < 60: 
        return 'Middle-aged'
    elif C_AGE >= 60:
        return 'Senior Citizen'
    else:
        return 'Minor'
    
def savings_behaviour(MTHCASA, MTHTD):
    """
    Determine the savings behavior of an individual based on their monthly cash and time deposits.

    Args:
        MTHCASA (float): Monthly cash amount.
        MTHTD (float): Monthly time deposit amount.

    Returns:
        int: Binary value indicating savings behavior. Returns 1 if both monthly cash and time deposits are greater than 0, otherwise returns 0.

    Raises:
        None

    Example:
        savings = savings_behaviour(1000, 500)  # Returns 1
    """
    if MTHCASA > 0 and MTHTD > 0:
        return 1
    else:
        return 0
    

def process_data(df):
    """
    Process the input DataFrame by performing various data transformations.

    Args:
        df (DataFrame): Input DataFrame containing the data to be processed.

    Returns:
        DataFrame: Processed DataFrame with transformed features and added columns.

    Raises:
        None

    Example:
        processed_df = process_data(input_df)
    """
    data = df.copy()
    logging.info('Lower the column names')
    data.columns = [i.lower().strip().replace(" ", "_") for i in data.columns]
    logging.info('Bin the age')
    data['age_bin'] = data['c_age'].apply(lambda x: bin_age(x))
    logging.info('Adding wealth accumulation feature')
    data['wealth_accumulation']  = data['asset_value'] / data['c_age']
    logging.info('Adding monthly transaction frequency feature')
    data['monthly_txn_frequency'] = data['ann_n_trx'] / 12
    logging.info('Adding credit utilization feature')
    data['credit_utilization'] = data['cc_ave'] / data['cc_lmt']
    logging.info('Adding saving behaviour flag')
    data['savings_behaviour'] = data.apply(lambda row: savings_behaviour(row['mthcasa'], row['mthtd']), axis=1)
    logging.info('Adding debt to asset ratio feature')
    data['debt_to_asset_ratio'] = data['ann_trn_amt'] / data['asset_value']
    logging.info('Adding transaction frequency per product feature')
    data['txn_freq_per_prd'] = data['ann_n_trx'] / data['num_prd']
    logging.info('Adding investment to debt ratio feature')
    data['investment_to_debt_ratio'] = data['ut_ave'] / data['ann_trn_amt']
    logging.info('Encode C_Seg feature as : AFFLUENT=1 and NORMAL=0')
    data['target'] = data['c_seg'].apply(lambda x: 1 if x=='AFFLUENT' else 0)
    print(data['c_seg'].value_counts(dropna=False))
    logging.info('Clean up home loan and auto loan tag')
    tag_columns = ['hl_tag', 'al_tag']
    data[tag_columns] = data[tag_columns].fillna(value=0)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    return data


def impute_cat_columns(df, cat_columns):
    """
    Impute missing values in categorical columns of a DataFrame.

    Args:
        df (DataFrame): Input DataFrame containing the data with missing values.
        cat_columns (list): List of column names for categorical columns to be imputed.

    Returns:
        DataFrame: DataFrame with missing values imputed in the specified categorical columns.

    Raises:
        None

    Example:
        filled_df = impute_cat_columns(input_df, ['incm_typ', 'other_cat_column'])
    """
    data = df.copy()
    for column in cat_columns:
        logging.info(f'Impute cat column: {column}')
        if column == 'incm_typ':
            data[column] = data[column].fillna(value=0.0)
        else:
            data[column] = data[column].fillna(value='UNKNOWN')
    return data


def encode_cols(df, cols_to_encode):
    """
    Encode categorical columns in a DataFrame using LabelEncoder and save the encoders.

    Args:
        df (DataFrame): Input DataFrame containing the data to be encoded.
        cols_to_encode (list): List of column names for categorical columns to be encoded.

    Returns:
        DataFrame: DataFrame with categorical columns encoded.

    Raises:
        None

    Example:
        encoded_df = encode_cols(input_df, ['col1', 'col2'])
    """
    data = df.copy()
    encoders = {}
    for column in cols_to_encode:
        logging.info(f'Encoding column: {column}')
        encoder = LabelEncoder()
        encoded_data = encoder.fit_transform(data[column])
        encoders[column] = encoder
        data[column] = encoded_data

    for column, encoder in encoders.items():
        logging.info(f'Saving encoder: {column}: filename: {column}_encoder.pkl')
        filename = f"./model_repo/{column}_encoder.pkl"
        joblib.dump(encoder, filename)
    return data

