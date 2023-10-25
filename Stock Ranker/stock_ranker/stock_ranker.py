import logging
import random

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from xgboost import XGBRanker

# ML Flow tracking
artifact_location = "gs://mlflow_warehouse/0"
mlflow_user = "mlflow_user"
mlflow_pass = "pass"
postgresql_database = "mlflow_db"
tracking_uri = f"postgresql://{mlflow_user}:{mlflow_pass}@127.0.0.1:5432/{postgresql_database}"
mlflow.set_tracking_uri(tracking_uri)
model_file = "model.pkl"
n_bins = 50

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def _create_bins(train, test):
    train['rating'] = pd.cut(train['return'], bins=n_bins, labels=np.arange(1, n_bins + 1, 1)[::-1])
    bin_info = train.groupby(['rating'])['return'].describe()[['min', 'max']]
    # mlflow.log_text(bin_info, model_file)
    test['rating'] = np.nan
    for r in bin_info.index:
        min_val = bin_info.loc[r]['min']
        max_val = bin_info.loc[r]['max']
        test['rating'] = np.where((test['return'] >= min_val) & (test['return'] <= max_val), r, test['rating'])
    test['rating'] = np.where((test['return'] <= bin_info.loc[n_bins]['min']), n_bins, test['rating'])
    test['rating'] = np.where((test['return'] >= bin_info.loc[1]['max']), 1, test['rating'])
    train = train.dropna(axis=0)
    test = test.dropna(axis=0)
    return train, test


def create_groups(df_):
    df_['analysis_date'] = pd.to_datetime(df_['analysis_date'])
    df_['u_id'] = pd.to_datetime(df_['analysis_date']).dt.strftime('%B %A %d %Y')
    q_list = df_['u_id'].value_counts()
    q_list = q_list.sort_index()
    # Do some data preprocessing
    return q_list


def _predict(data, model, features_to_use):
    preds = []
    for userId, df in data.groupby('u_id'):
        pred = model.predict(df[features_to_use])
        productId = np.array(df.reset_index()['symbol'])
        topK_index = np.argsort(pred)[::-1]
        df['stocks'] = productId[topK_index]
        df['id'] = [userId] * len(topK_index)
        df['ranks'] = list(range(1, len(topK_index) + 1))
        preds.append(df)
    results = pd.concat(preds)
    return results


def split_train_and_test(trades_df, split_type="random", train_split_ratio=80, test_split_ratio=20):
    trades_df = trades_df.sort_values('analysis_date')
    if split_type == "random":
        my_list = list(trades_df['analysis_date'].unique())
        train_date_list = random.sample(my_list, int(len(my_list) * train_split_ratio / 100))
        test_date_list = list(set(my_list) - set(train_date_list))
        train_df = trades_df[trades_df.analysis_date.isin(train_date_list)].copy(deep=True)
        test_df = trades_df[trades_df.analysis_date.isin(test_date_list)].copy(deep=True)
    else:
        my_list = list(trades_df['analysis_date'].unique())
        edge = int(len(my_list) * train_split_ratio / 100)
        train_df = my_list[:edge].copy(deep=True)
        test_df = my_list[edge:].copy(deep=True)
    return train_df, test_df


def create_estimator(t_id, trades_df, model_params):
    # Create experiment
    experiment_name = f"{t_id}_ranker"
    # check if the experiment already exists
    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(name=experiment_name, artifact_location=artifact_location)
    experiment = mlflow.get_experiment_by_name(experiment_name)

    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=f"run_{experiment_name}"):

        # ------- create train and test set -----------#
        logging.info("Splitting the data")
        train, test = split_train_and_test(trades_df, model_params.get('split_type'))
        mlflow.log_metric("number of unique analysis date in train dataset", train.shape[0])
        mlflow.log_metric("number of unique analysis date in test dataset", test.shape[0])

        # ------- Bucketize the returns ----------#
        logging.info("Bucketizing the returns")
        train, test = _create_bins(train, test)

        # ------------ create groups ---------------
        logging.info("Creating groups and group id")
        query_list_train = create_groups(train)
        query_list_test = create_groups(test)

        # ------------- select feature to be used for training --------------
        features_cols = train.drop(
            ['rating', 'analysis_date', 'symbol', 'return', 'u_id'], axis=1).columns
        mlflow.log_param("features used", features_cols)
        X_train = train[features_cols]
        y_train = train['rating']
        X_test = test[features_cols]
        y_test = test['rating']

        # ---------- model training ----------
        mlflow.log_params(model_params)
        logging.info("beginning model training")
        if model_params['model_engine'] == "XGBRanker":
            model_params.pop('model_engine')
            ranker = XGBRanker(**model_params)

        elif model_params['model_engine'] == "LGBRanker":
            model_params.pop('model_engine')
            ranker = XGBRanker(**model_params)

        else:
            ranker = XGBRanker(**model_params)

        ranker.fit(X_train, y_train, group=query_list_train,
                   eval_set=[(X_test, y_test)],
                   eval_group=[list(query_list_test)])

        logging.info("Model training completed..!")

        # ---- evaluate model ---------
        logging.info("Evaluating model based on ndcg score")
        predicted = _predict(test, ranker, features_cols)

        # ------------- rankwise retuns ---------------
        rankwise_return = predicted.groupby(['ranks'])['return'].describe()
        topn = rankwise_return[:10]['mean']
        plt.plot(topn)
        img = f"/Users/vanshika/Desktop/stock_ranker/{experiment.name + model_params.get('itr')}.png"
        plt.savefig(img)
        plt.close()
        # --------Save best model ----------#
        logging.info("saving best model")
