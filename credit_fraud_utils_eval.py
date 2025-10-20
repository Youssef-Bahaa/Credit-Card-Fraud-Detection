#credit_fraud_utils_eval.py
import os
import shutil

import joblib
import numpy as np
import yaml
from numpy.core.defchararray import endswith
from sklearn.metrics import (
    precision_score,recall_score,f1_score,roc_auc_score,
    accuracy_score,precision_recall_curve,roc_curve,auc,
    confusion_matrix,ConfusionMatrixDisplay,classification_report as skl_classification_report)
import matplotlib.pyplot as plt
import json

from xgboost import XGBClassifier

from credit_fraud_utils_data import (load_data, split_feature_target, remove_outliers,
                                     remove_duplicates, handle_missing_values, scale_features_train,
                                     scale_features_test,balance_dataset,load_test)
from credit_fraud_utils_utilities import (load_config,save_results,load_model,save_model)
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report as skl_class_report


def find_best_threshold(y_true, y_probs, model_name='Model'):
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_idx]

    print(f'Best Threshold for {model_name} = {best_threshold:.3f}, Best F1 = {best_f1:.3f}')
    return float(best_threshold)


def apply_threshold(y_probs, threshold):
    return (y_probs >= threshold).astype(int)



def plot_confusion_matrix(y_true,y_pred,model_name='Model',save_path=''):
    cm = confusion_matrix(y_true,y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    fig, ax = plt.subplots(figsize=(8,6))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    ax.set_title(f'Confusion Matrix - {model_name}')
    ax.grid(False)
    plt.tight_layout()

    if save_path:
        plt.savefig(os.path.join(save_path, f'{model_name}_confusion_matrix.png'))
    plt.close()

def plot_precision_recall_curve(y_true,y_probs,model_name='Model',save_path=''):
    precision,recall,_ = precision_recall_curve(y_true,y_probs)
    PR_auc = auc(recall,precision)

    plt.plot(recall,precision,color='darkorange',lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision Recall Curve {model_name}')


    precision,recall,threshold = precision_recall_curve(y_true,y_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

    best_idx = np.argmax(f1_scores)
    best_threshold = threshold[best_idx] if best_idx < len(threshold) else 0.5
    best_f1 = f1_scores[best_idx]

    if save_path:
        plt.savefig(os.path.join(save_path, f'{model_name}_precision_recall_curve.png'))

    print(f'Best Threshold for {model_name} = {best_threshold:.3f}, Best F1 = {best_f1:.3f}')
    plt.close()
    return float(best_threshold),PR_auc


def plot_roc_curve(y_true,y_prob,model_name='Model',save_path=''):
    fpr,tpr,_ = roc_curve(y_true,y_prob)
    roc_auc = auc(fpr,tpr)

    plt.plot(fpr,tpr,color='blue',lw=2,label=f'AUC = {roc_auc:.2f}',linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve {model_name}')
    plt.tight_layout()
    plt.legend(loc='lower right')

    #save should be handeled
    if save_path:
        plt.savefig(os.path.join(save_path,f'{model_name}_roc_curve.png'))
    plt.close()
    return roc_auc


def classification_report(y_true,y_pred,model_name='Model'):
    print(f'Classification report for {model_name}')
    report_dic = skl_classification_report(y_true,y_pred,output_dict=True)
    print(report_dic)
    return report_dic


def save_metrics(y_true,y_pred,y_prob,model_name='',metrics_path='',plots_path='',dataset_type='train'):
    os.makedirs(metrics_path,exist_ok=True)
    os.makedirs(plots_path, exist_ok=True)

    model_file_name = f'{model_name}_{dataset_type}'

    plot_confusion_matrix(y_true,y_pred,model_file_name,plots_path)
    plot_roc_curve(y_true,y_prob,model_file_name,plots_path)
    best_threshold,PR_auc = plot_precision_recall_curve(y_true,y_prob,model_file_name,plots_path)


    metrics = {
        'model':model_file_name,
        'accuracy':accuracy_score(y_true,y_pred),
        'precision':precision_score(y_true,y_pred),
        'recall' : recall_score(y_true,y_pred),
        'f1score': f1_score(y_true,y_pred),
        'roc_auc':roc_auc_score(y_true,y_prob),
        'PR_AUC' : PR_auc,
        'classification report':classification_report(y_true,y_pred,model_file_name)
    }

    file_path = os.path.join(metrics_path, f'{model_file_name}_metrics.json')
    with open(file_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f'Classification report saved at path : {file_path}')
    return metrics


def save_best_model(time_dir, best_model_dir='models/best_model', metric='f1score'):
    metric_path = os.path.join(time_dir, 'metrics')
    metrics_files = [f for f in os.listdir(metric_path) if f.endswith('_val_metrics.json')]

    if not metrics_files:
        print(f"No metrics JSON files found in {time_dir}")
        return None

    best_metric = -float('inf')
    best_model_name = None

    for mfile in metrics_files:
        with open(os.path.join(metric_path, mfile), 'r') as f:
            metrics = json.load(f)
            value = metrics.get(metric, None)
            if value is not None and value > best_metric:
                best_metric = value
                best_model_name = metrics['model']

    # strip _train or _val
    if best_model_name.endswith('_train'):
        base_model_name = best_model_name.replace('_train', '')
    elif best_model_name.endswith('_val'):
        base_model_name = best_model_name.replace('_val', '')
    else:
        base_model_name = best_model_name


    os.makedirs(best_model_dir, exist_ok=True)


    best_model_file = os.path.join(best_model_dir, f'{base_model_name}_best.pkl')

    model_file_path = os.path.join(time_dir, f'{base_model_name}.pkl')
    if not os.path.exists(model_file_path):
        print(f"Model file not found, creating new pickle for best model: {best_model_file}")
    else:
        model = joblib.load(model_file_path)
        joblib.dump(model, best_model_file)



    print(f"Best model '{base_model_name}' (metric={metric}) saved at {best_model_file}")
    return {
        'model': best_model_file
    }



if __name__ == '__main__':
    config = load_config('config/config.yaml')
    model_config = load_config('config/models.yaml')
    X_test,y_test = load_test(config)

    model_dir = 'models/best_model'
    model_file = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]

    if not model_file:
        raise FileNotFoundError(f"No .pkl model found in {model_dir}")

    model_path = os.path.join(model_dir,model_file[0])
    model = joblib.load(model_path)


    scaler_path = 'models/scalers/scaler.pkl'
    scaler = joblib.load(scaler_path)

    train, val = load_data(config)

    X_train, y_train = split_feature_target(train, config['target_column'])
    X_val, y_val = split_feature_target(val, config['target_column'])

    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)


    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    f1_train = f1_score(y_train,y_pred_train)
    f1_val = f1_score(y_val,y_pred_val)
    f1_test = f1_score(y_test,y_pred_test)

    report = skl_class_report(y_test,y_pred_test)
    print('Classification Report for test data.')
    print(report)



