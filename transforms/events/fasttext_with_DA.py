import random
import fasttext
import numpy as np
import pandas as pd
import public_function as pf
from collections import Counter
import hashlib
import time
import os


class FastTextLab:
    def __init__(self, config, cases, split=True):
        self.config = config
        self.cases = cases
        if self.config['supervised']:
            self.method = fasttext.train_supervised
        else:
            self.method = fasttext.train_unsupervised
        self.nodes = config['nodes'].split()
        unique_anomalies = cases['anomaly_type'].astype(str).unique() # Get unique types as strings
        unique_anomalies = [a for a in unique_anomalies if a != 'nan'] # Remove 'nan' if present
        self.anomaly_types = ['[normal]'] + sorted(unique_anomalies) # Create sorted list, always starting with [normal]
        self.anomaly_type_labels = {name: i for i, name in enumerate(self.anomaly_types)}
        # --- End replacement ---

        self.node_labels = dict(zip(self.nodes, range(len(self.nodes))))

        # Print the created label dictionaries for verification
        print("DEBUG __init__: Node Labels:", self.node_labels)
        print("DEBUG __init__: Anomaly Type Labels:", self.anomaly_type_labels)

        self.train_data, self.test_data = self.prepare_data()

        self.node_labels = dict(zip(self.nodes, range(len(self.nodes))))

        # Print the created label dictionaries for verification
        print("DEBUG __init__: Node Labels:", self.node_labels)
        print("DEBUG __init__: Anomaly Type Labels:", self.anomaly_type_labels)

        self.train_data, self.test_data = self.prepare_data()
    
    def prepare_data(self):
        metric_trace_text_path = self.config['text_path']
        temp_data = pf.load(metric_trace_text_path) # Keys are likely strings '0', '1', etc.

        train_indices = []
        test_indices = []
        
        # Check if 'data_type' column is usable
        if 'data_type' in self.cases.columns and any(self.cases['data_type'].isin(['train', 'test'])):
            print("[INFO] prepare_data: Splitting cases based on 'data_type' column.")
            # Get indices based on 'data_type'
            train_indices = self.cases[self.cases['data_type']=='train'].index.tolist()
            test_indices = self.cases[self.cases['data_type']=='test'].index.tolist()
        else:
            # Fallback: Split by index
            if 'data_type' not in self.cases.columns:
                 print("[WARNING] prepare_data: 'data_type' column not found in labels/cases DataFrame!")
            else:
                 print("[WARNING] prepare_data: 'data_type' column lacks 'train'/'test' values!")
                 
            print("[INFO] prepare_data: Using fallback: Splitting cases by index (80/20).")
            all_indices = self.cases.index.tolist()
            if not all_indices:
                 print("[ERROR] prepare_data: No cases found in the labels file index!")
            else:
                train_size = int(len(all_indices) * 0.8)
                train_indices = all_indices[:train_size]
                test_indices = all_indices[train_size:]

        # --- CRITICAL FIX: Convert indices to STRINGS to match pkl keys ---
        train_keys = [str(i) for i in train_indices]
        test_keys = [str(i) for i in test_indices]
        total_keys = [str(i) for i in self.cases.index.tolist()] # Assuming total might be needed later
        # --- END FIX ---

        print(f"DEBUG prepare_data: Found {len(train_keys)} training keys.")
        print(f"DEBUG prepare_data: Found {len(test_keys)} testing keys.")

        # Pass the STRING keys to save_to_txt
        self.save_to_txt(temp_data, train_keys, self.config['train_path'])
        self.save_to_txt(temp_data, test_keys, self.config['test_path'])

        # Read data back - Initialize to empty lists
        train_data = []
        test_data = []
        try:
            with open(self.config['train_path'], 'r', encoding='utf-8') as f: # Specify encoding
                train_data = f.read().splitlines()
        except FileNotFoundError:
            print(f"[WARNING] prepare_data: Could not find generated file: {self.config['train_path']}")
            
        try:
            with open(self.config['test_path'], 'r', encoding='utf-8') as f: # Specify encoding
                test_data = f.read().splitlines()
        except FileNotFoundError:
            print(f"[WARNING] prepare_data: Could not find generated file: {self.config['test_path']}")

        # Ensure a tuple is always returned
        return train_data, test_data
    

    def w2v_DA(self):
        da_train_data = self.train_data.copy()
        model = self.method(self.config['train_path'], dim=self.config['vector_dim'],
                                            minCount=self.config['minCount'], minn=0, maxn=0, epoch=self.config['epoch'])
        random.seed(0)
        for anomaly_type in self.anomaly_types:
            for node in self.nodes:
                sample_count = len([
                    text for text in self.train_data
                    if text.split('__label__')[-1] == str(self.node_labels[node])+str(self.anomaly_type_labels[anomaly_type])])
                if sample_count == 0:
                    continue
                anomaly_texts = [
                    text for text in self.train_data
                    if text.split('\t')[-1] == f'__label__{self.node_labels[node]}{self.anomaly_type_labels[anomaly_type]}']
                loop = 0
                while sample_count < self.config['sample_count']:
                    loop += 1
                    if loop >= 10*self.config['sample_count']:
                        break
                    # 随机选取相应label的序列进行复制
                    chosen_text, label = anomaly_texts[random.randint(0, len(anomaly_texts) - 1)].split('\t')
                    chosen_text_splits = chosen_text.split()
                    if len(chosen_text_splits) < self.config['minCount']:
                        continue
                    # 随机选取若干事件进行替换
                    edit_event_ids = random.sample(range(len(chosen_text_splits)), self.config['edit_count'])
                    for event_id in edit_event_ids:
                        # 替换被选中的事件，选取离他距离最近的事件用于替换
                        nearest_event = model.get_nearest_neighbors(chosen_text_splits[event_id])[0][-1]
                        chosen_text_splits[event_id] = nearest_event
                    da_train_data.append(
                        ' '.join(chosen_text_splits) + f'\t__label__{self.node_labels[node]}{self.anomaly_type_labels[anomaly_type]}')
                    sample_count += 1
        
#                 words = []
        with open(self.config['train_da_path'], 'w') as f:
            for text in da_train_data:
                f.write(text + '\n')


    def event_embedding_lab(self, data_path):
        model = self.method(data_path, dim=self.config['vector_dim'],
                                          minCount=self.config['minCount'], minn=0, maxn=0, epoch=self.config['epoch'])
        event_dict = dict()
        for event in model.words:
            event_dict[event] = model[event]
        return event_dict


    def save_to_txt(self, data: dict, keys, save_path):
        """ Writes data to a text file in fastText format. (More Robust) """
        fillna = False
        lines_written = 0
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Determine if data keys are integers or strings
        data_keys_are_int = all(isinstance(k, int) for k in data.keys())
        
        with open(save_path, 'w', encoding='utf-8') as f:
            for key_from_list in keys: # These keys come from the CSV index
                
                # --- Adjust key type to match data dictionary keys ---
                if data_keys_are_int:
                    try:
                        lookup_key = int(key_from_list) # Try converting CSV index to int
                    except ValueError:
                        continue # Skip if conversion fails
                else:
                    lookup_key = str(key_from_list) # Assume data keys are strings
                # --- End adjustment ---

                if lookup_key not in data:
                    continue 
                    
                case_data = data[lookup_key]
                if not isinstance(case_data, dict):
                    continue

                for node_info, text in case_data.items(): 
                    if not isinstance(node_info, (tuple, list)) or len(node_info) < 2:
                        continue 
                        
                    node_name = node_info[0]
                    # Ensure anomaly name is treated as string for lookup
                    anomaly_name = str(node_info[1]) 

                    if not isinstance(text, str) or len(text.strip()) == 0:
                        if fillna: text = 'None'
                        else: continue

                    text = text.strip() 

                    node_label_idx = self.node_labels.get(node_name, None)
                    anomaly_label_idx = self.anomaly_type_labels.get(anomaly_name, None)

                    if node_label_idx is not None and anomaly_label_idx is not None:
                        label_str = f"__label__{node_label_idx}{anomaly_label_idx}"
                        f.write(f'{text}\t{label_str}\n') 
                        lines_written += 1

        print(f"DEBUG save_to_txt: Finished writing to {save_path}. Total lines written: {lines_written}")
        return
    
    def do_lab(self):
        self.w2v_DA()
        pf.save(self.config['save_path'], self.event_embedding_lab(self.config['train_da_path']))



def run_fasttext(config, labels):
    # event embedding流程；基于数据增强
    start_ts = time.time()
    lab2 = FastTextLab(config, labels)
    lab2.do_lab()
    end_ts = time.time()
    print('fasttext time used:', end_ts-start_ts, 's')