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


    # FASTTEXT (commented out)
    # monitor_fasttext = ResourceMonitor()
    # monitor_fasttext.start_logging('resource_log_fasttext.csv', interval=1)
    # try:
    #     print('[fasttext]')
    #     fasttext_with_DA.run_fasttext(deal_config(config, 'fasttext'), labels)
    # finally:
    #     monitor_fasttext.stop_logging()
    # from plot_resource_usage import plot_resource_usage
    # print('\n[FastText Resource Usage]')
    # plot_resource_usage('resource_log_fasttext.csv')

    # SENTENCE EMBEDDING (commented out)
    # monitor_sent = ResourceMonitor()
    # monitor_sent.start_logging('resource_log_sentence.csv', interval=1)
    # try:
    #     print('[sentence_embedding]')
    #     sententce_embedding.run_sentence_embedding(deal_config(config, 'sentence_embedding'))
    # finally:
    #     monitor_sent.stop_logging()
    # print('\n[Sentence Embedding Resource Usage]')
    # plot_resource_usage('resource_log_sentence.csv')

    # GNN TRAINING
    monitor_gnn = ResourceMonitor()
    monitor_gnn.start_logging('resource_log_gnn.csv', interval=1)
    try:
        print('[dgl]')
        lab_id = 9 # 实验唯一编号
        He_DGL.UnircaLab(deal_config(config, 'he_dgl')).do_lab(lab_id)
    finally:
        monitor_gnn.stop_logging()
    print('\n[GNN Training Resource Usage]')
    plot_resource_usage('resource_log_gnn.csv')