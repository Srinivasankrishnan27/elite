import pandas as pd
import joblib
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from scripts.feature_eng.feature_eng import process_data
from scripts.feature_eng.feature_eng import impute_cat_columns
from scripts.feature_eng.feature_eng import encode_cols
from scripts.model.utils import feature_selection
from scripts.model.model_params import param_grid
from scripts.model.utils import train_model, get_report, plot_precision_recall_curve, plot_roc_curve
from scripts import random_state
import warnings
warnings.filterwarnings("ignore")


clf = RandomForestClassifier(max_features='log2', n_jobs=-1, random_state=random_state, class_weight='balanced')

def setup_logger():
    current_time = datetime.now()

    # Format the current date and time as a string for the filename
    current_time_str = current_time.strftime("%Y-%m-%d_%H%M%S")
    log_filename = f"segment_upgrade_{current_time_str}.log"

    # Configure root logger
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filemode='w'  # Use 'w' to overwrite the log file with each run
    )

    # Create a file handler to save logs to a file
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the file handler to the root logger
    logging.getLogger('').addHandler(file_handler)

    # Create a stream handler to print logs to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Add the console handler to the root logger
    logging.getLogger('').addHandler(console_handler)
    


def read_data():
    logging.info("Reading the data..")
    data = pd.read_excel('./data/Assessment.xlsx', sheet_name='Data')
    return data.copy()


def clean_data(df, cat_cols, num_cols, cols_to_encode):
    data = df.copy()
    data = impute_cat_columns(df=data, cat_columns=cat_cols)
    logging.info(f"Fill the numerical columns null value with zero")
    data[num_cols] = data[num_cols].fillna(value=0)
    logging.info(f"Type cast the numerical and categorical cols")
    data[cat_cols] = data[cat_cols].astype('category')
    data[num_cols] = data[num_cols].astype('float64')
    data = encode_cols(df=data, cols_to_encode=cols_to_encode)
    return data.copy()



if __name__ == "__main__":
    setup_logger()
    data = read_data()

    x_cols = ['c_edu', 'c_hse', 'incm_typ', 'gn_occ', 'num_prd', 'casatd_cnt', 'mthcasa', 'maxcasa', 'mincasa', 'drvcr', 'mthtd', 'maxtd',\
            'asset_value', 'hl_tag', 'al_tag', 'pur_price_avg','ut_ave', 'maxut', 'n_funds', 'cc_ave', 'max_mth_trn_amt', 'min_mth_trn_amt', \
                'avg_trn_amt', 'ann_trn_amt', 'ann_n_trx', 'cc_lmt','age_bin', 'wealth_accumulation', 'monthly_txn_frequency', 'credit_utilization',\
                    'savings_behaviour', 'debt_to_asset_ratio', 'txn_freq_per_prd', 'investment_to_debt_ratio']

    cat_cols = ['c_edu', 'c_hse', 'incm_typ', 'gn_occ', 'age_bin', 'hl_tag', 'al_tag', 'savings_behaviour']
    cols_to_encode = ['c_edu', 'c_hse', 'gn_occ', 'age_bin']
    num_cols = [i for i in x_cols if i not in cat_cols]

    data = process_data(data)


    count_df = pd.DataFrame(data.groupby('c_id')['c_age'].count().reset_index())
    count_df.rename(columns={'c_age': 'id_count'}, inplace=True)
    duplicate_c_id = count_df[count_df.id_count >1]['c_id'].values.tolist()
    non_duplicate_c_id = count_df[count_df.id_count ==1]['c_id'].values.tolist()

    logging.info(f"Number of unique customers: {count_df.shape[0]}")
    logging.info(f"% of customers with one record in the dataset {round(100*len(non_duplicate_c_id)/count_df.shape[0], 3)} %")
    logging.info(f"% of customers with multiple records for the same customer ID's {round(100*len(duplicate_c_id)/count_df.shape[0], 3)} %")

    data = clean_data(df=data, cat_cols=cat_cols, num_cols=num_cols, cols_to_encode=cols_to_encode)

    

    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, num_cols)])

    logging.info("Split data into training and testing set")
    X_train, X_test, Y_train, Y_test = train_test_split(data[x_cols], data['target'], test_size=0.2, random_state=random_state, stratify=data['target'])
    logging.info(f"Training records: {X_train.shape[0]}, {100*X_train.shape[0]/data.shape[0]:.2f} %")
    logging.info(f"Testing records: {X_test.shape[0]}, {100*X_test.shape[0]/data.shape[0]:.2f} %")

    X_train.reset_index(inplace=True, drop=True)
    X_test.reset_index(inplace=True, drop=True)
    Y_train = Y_train.reset_index(drop=True)
    Y_test = Y_test.reset_index(drop=True)

    selector, selected_cols, selected_num_cols, selected_cat_cols = feature_selection(X=X_train[x_cols], 
                                                                                    Y=Y_train, 
                                                                                    cat_cols=cat_cols, 
                                                                                    num_cols=num_cols, 
                                                                                    clf=clf, 
                                                                                    scoring='f1', 
                                                                                    method='load')


    model_dict = {}
    metrics_list = []
    for model_name, model_attr in param_grid.items():
        if model_name in ['LogisticRegression', 'MLPClassifier']:
            X_train_processed = preprocessor.fit_transform(X_train)
            X_train_processed = pd.DataFrame(X_train_processed, columns=num_cols)
            X_train_processed = pd.concat([X_train_processed, X_train[cat_cols]], axis=1)
            X_train_df = X_train_processed[selected_cols].copy()

            X_test_processed = preprocessor.fit_transform(X_test)
            X_test_processed = pd.DataFrame(X_test_processed, columns=num_cols)
            X_test_processed = pd.concat([X_test_processed, X_test[cat_cols]], axis=1)
            X_test_df = X_test_processed[selected_cols].copy()
        else:
            X_train_df = X_train[selected_cols].copy()
            X_test_df = X_test[selected_cols].copy()

        
        grid_search, best_estimator = train_model(X=X_train_df, 
                                                Y=Y_train, 
                                                model_name=model_name, 
                                                model=model_attr['model'], 
                                                model_param_grid=model_attr['param_grid'])
        predicted_probabilities = best_estimator.predict_proba(X_test_df)
        model_dict[f'{model_name}'] = {'model': best_estimator, 'search_meta': grid_search}

        print(''.join(200*['*']))
        plot_roc_curve(y_true=Y_test, predicted_prob=predicted_probabilities, model_name=model_name, show_fig=False)
        plot_precision_recall_curve(y_true=Y_test, predicted_prob=predicted_probabilities, model_name=model_name, show_fig=False)
        print(''.join(200*['*']))

        model_metrics_df = get_report(y_true=Y_test, predicted_prob=predicted_probabilities, model_name=model_name)
        metrics_list.append(model_metrics_df)

    metrics_df = pd.concat(metrics_list)
    joblib.dump(metrics_df, 'metrics_df.pkl')