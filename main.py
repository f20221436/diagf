from transforms.events import metric_trace_log_parse, fasttext_with_DA, sententce_embedding
from models import He_DGL
from public_function import deal_config, get_config
import os
import pandas as pd

from monitor import ResourceMonitor
from plot_resource_usage import plot_resource_usage


if __name__ == '__main__':
    config = get_config()
    label_path = os.path.join(config['base_path'], config['demo_path'],
                            config['label'], config['he_dgl']['run_table'])
    labels = pd.read_csv(label_path, index_col=0)

    # STEP 1: PARSE EVENTS FROM METRIC, TRACE, LOG
    print('\n' + '='*80)
    print('STEP 1: Parsing Events from Metric, Trace, and Log Data')
    print('='*80)
    monitor_parse = ResourceMonitor()
    monitor_parse.start_logging('resource_log_parse.csv', interval=1)
    try:
        print('[parse]')
        metric_trace_log_parse.run_parse(deal_config(config, 'parse'), labels)
    finally:
        monitor_parse.stop_logging()
    print('\n[Parse Resource Usage]')
    plot_resource_usage('resource_log_parse.csv')

    # STEP 2: FASTTEXT EMBEDDING WITH DATA AUGMENTATION
    print('\n' + '='*80)
    print('STEP 2: FastText Event Embedding with Data Augmentation')
    print('='*80)
    monitor_fasttext = ResourceMonitor()
    monitor_fasttext.start_logging('resource_log_fasttext.csv', interval=1)
    try:
        print('[fasttext]')
        fasttext_with_DA.run_fasttext(deal_config(config, 'fasttext'), labels)
    finally:
        monitor_fasttext.stop_logging()
    print('\n[FastText Resource Usage]')
    plot_resource_usage('resource_log_fasttext.csv')

    # STEP 3: SENTENCE EMBEDDING
    print('\n' + '='*80)
    print('STEP 3: Generating Sentence Embeddings')
    print('='*80)
    monitor_sent = ResourceMonitor()
    monitor_sent.start_logging('resource_log_sentence.csv', interval=1)
    try:
        print('[sentence_embedding]')
        sententce_embedding.run_sentence_embedding(deal_config(config, 'sentence_embedding'))
    finally:
        monitor_sent.stop_logging()
    print('\n[Sentence Embedding Resource Usage]')
    plot_resource_usage('resource_log_sentence.csv')

    # STEP 4: GNN TRAINING
    print('\n' + '='*80)
    print('STEP 4: Training GNN Model')
    print('='*80)
    monitor_gnn = ResourceMonitor()
    monitor_gnn.start_logging('resource_log_gnn.csv', interval=1)
    try:
        print('[dgl]')
        lab_id = 1  # Experiment unique ID
        He_DGL.UnircaLab(deal_config(config, 'he_dgl')).do_lab(lab_id)
    finally:
        monitor_gnn.stop_logging()
    print('\n[GNN Training Resource Usage]')
    plot_resource_usage('resource_log_gnn.csv')