import os
import pickle
import argparse
import yaml
import sys


def load(file):
    with open(file, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    return data


def save(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)


def get_config(config_file='gaia_config.yaml'):
    
    config_path = config_file # Assume absolute or relative to CWD initially
    
    if not os.path.isabs(config_file):
        script_dir = os.path.dirname(os.path.abspath(__file__)) 
        # Path 1: Inside a 'config' subdirectory relative to this script
        path_in_config_subdir = os.path.join(script_dir, 'config', config_file) 
        # Path 2: Directly in the same directory as this script
        path_in_script_dir = os.path.join(script_dir, config_file) 

        if os.path.exists(path_in_config_subdir):
            config_path = path_in_config_subdir
        elif os.path.exists(path_in_script_dir):
            config_path = path_in_script_dir
        # If neither found, it will try the original config_file path (relative to CWD)

    config_path = os.path.normpath(config_path)

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) 
        return config
    except Exception: # Catch FileNotFoundError and other loading errors silently
        return None

def min_max_normalized(feature):
    feature_copy = feature.copy().astype(float)
    for i in range(len(feature_copy)):
        min_f, max_f = min(feature_copy[i]), max(feature_copy[i])
        if min_f == max_f:
            feature_copy[i] = [0]*len(feature_copy[i])
        else:
            feature_copy[i] = (feature_copy[i] - min_f) / (max_f - min_f)
    return feature_copy


def deal_config(config, key):
    new_config = {}
    for k in config[key].keys():
        if 'path' in k or 'dir' in k:
            if config[key][k] or config[key][k] == '':
                path = os.path.join(config['base_path'], config['demo_path'],
                                    config['label'], config[key][k])
                if 'dir' in k:
                    if not os.path.exists(path):
                        os.makedirs(path)
                new_config[k] = path
            else:
                new_config[k] = config[key][k]
        else:
            new_config[k] = config[key][k]

    return new_config


if __name__ == '__main__':
    config = get_config()
    print(config['fasttext']['vector_dim'])
    cur_path = os.getcwd()
    print(cur_path[:cur_path.find('unirca')])

