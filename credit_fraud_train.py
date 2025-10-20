#credit_fraud_train.py
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from credit_fraud_utils_data import (load_data, split_feature_target, remove_outliers,
                                     remove_duplicates, handle_missing_values, scale_features_train,
                                     scale_features_test, balance_dataset, load_test)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from credit_fraud_utils_utilities import (load_config,save_results,load_model,save_model)
import joblib
import os
import yaml
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from datetime import datetime
from credit_fraud_utils_eval import (save_metrics,save_best_model)
from sklearn.metrics import f1_score


model_map = {
    'LogisticRegression': LogisticRegression,
    'RandomForest': RandomForestClassifier,
    'XGBoost': XGBClassifier,
    'DecisionTree': DecisionTreeClassifier,
}


def train_model(model_name,X_train,y_train,X_val,y_val,config,time_dir):
    params = config['models'].get(model_name,{})
    model_class = model_map.get(model_name)

    if model_class is None:
        raise ValueError(f'Model {model_name} does not exist')

    model = model_class(**params)
    model.fit(X_train,y_train)
    save_model(model=model,dir='models',file_name=model_name,time_dir=time_dir)

    return model



def train_all_models(X_train,y_train,X_val,y_val,config,time_dir):
    trained_models = {}

    for model_name in config['models']:
        try:
            model = train_model(model_name, X_train, y_train, X_val, y_val, config,time_dir)
            trained_models[model_name] = model
            metrics_dir = os.path.join(time_dir,'metrics')
            plots_path = os.path.join(time_dir,'plots')

            #evaluate_on_train
            save_metrics(y_train,model.predict(X_train),model.predict_proba(X_train)[:,1],
                        model_name=f'{model_name}',metrics_path=metrics_dir,
                         plots_path=plots_path,dataset_type='train'
                         )

            #evaluate on validation
            save_metrics(y_val,model.predict(X_val),model.predict_proba(X_val)[:,1],
                        model_name=f'{model_name}',metrics_path=metrics_dir,
                         plots_path=plots_path,dataset_type='val'
                         )

        except Exception as e:
            print(f'Failed to train model {model_name}: {e}')

    return trained_models



if __name__ == '__main__':

    config = load_config('config/config.yaml')
    model_config = load_config('config/models.yaml')

    train, val = load_data(config)
    # train = remove_outliers(train, config['outlier_method']['method'])
    train = remove_duplicates(train)
    train = handle_missing_values(train, config['missing_method'])

    # ======== Split features and target before balancing ========
    X_train, y_train = split_feature_target(train, config['target_column'])




    # ======== Balance the dataset ========
    if 'balancing' in config:
        method = config['balancing'].get('method', 'smote')
        params = config['balancing'].get('smote_params', {})
        random_state = config['balancing'].get('random_state', 42)
        params['sampling_strategy'] = float(params['sampling_strategy'])
        X_train, y_train = balance_dataset(X_train, y_train, method, random_state, params)
    # =============================================================

    # Prepare validation data
    X_val, y_val = split_feature_target(val, config['target_column'])

    # Scale features
    X_train, scaler = scale_features_train(X_train, config['scaling']['method'])
    X_val = scale_features_test(X_val, scaler)

    scaler_dir = 'models/scalers'
    os.makedirs(scaler_dir, exist_ok=True)
    scaler_path = os.path.join(scaler_dir, 'scaler.pkl')

    joblib.dump(scaler, scaler_path)


# Create directory for this run
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M')
    time_dir = os.path.join('models', timestamp)
    os.makedirs(time_dir, exist_ok=True)

    # Train all models
    trained_models = train_all_models(X_train, y_train, X_val, y_val, model_config, time_dir)

    print(f'Trained models: {trained_models.keys()}')

    # Save the best model
    save_best_model(time_dir)












