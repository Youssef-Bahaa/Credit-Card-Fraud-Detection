import yaml
import os
import joblib
import json
from datetime import datetime


def load_config(config_path = 'config/config.yaml'):
    try:
        with open(config_path,'r') as f:
            config = yaml.safe_load(f)
            return config
    except FileNotFoundError:
        raise FileNotFoundError(f'Config file not found at path: {config_path}')
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f'Error parsing yaml file: {e}')


def save_model(model,dir='models',file_name='model.pkl',time_dir=None):
    try:
        if not file_name.endswith('.pkl'):
            file_name += '.pkl'

        if time_dir is None:
            timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M')
            time_dir = os.path.join(dir,timestamp)
            os.makedirs(time_dir,exist_ok=True)

        file_path = os.path.join(time_dir,file_name)
        os.makedirs(dir,exist_ok=True)
        joblib.dump(model,file_path)
        print(f'Model saved to : {file_path}')
    except Exception as e:
        raise RuntimeError(f'Error saving model: {e}')



def save_best_model(model,dir='models/best_model',file_name='best_model.pkl'):
    try:
        if not file_name.endswith('.pkl'):
            file_name += '.pkl'

        os.makedirs(dir,exist_ok=True)
        file_path = os.path.join(dir,file_name)
        joblib.dump(model,file_path)
        print(f'Best model saved to : {file_path}')

    except Exception as e:
        raise RuntimeError(f'Error saving best model to : {e}')



def load_model(file_path='model.pkl'):
    try:
        model = joblib.load(file_path)
        print(f'Model loaded from : {file_path}')
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f'Model not found at path: {file_path}')
    except Exception as e:
        raise RuntimeError(f'Error loading model: {e}')


def save_results(results,file_path='results.json'):
    try:
        with open(file_path,'w') as f:
            json.dump(results,f,indent=3)
        print(f'Results saved to: {file_path}')
    except Exception as e:
        raise RuntimeError(f'Error saving results: {file_path}')
