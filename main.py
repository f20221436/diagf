from transforms.events import metric_trace_log_parse, fasttext_with_DA, sententce_embedding
from models import He_DGL
from public_function import deal_config, get_config
import os
import pandas as pd


if __name__ == '__main__':
    config = get_config()
    label_path = os.path.join(config['base_path'], config['demo_path'],
                              config['label'], config['he_dgl']['run_table'])
    labels = pd.read_csv(label_path, index_col=0)

    # print('[parse]')
    # metric_trace_log_parse.run_parse(deal_config(config, 'parse'), labels)

    print('[fasttext]')
    fasttext_with_DA.run_fasttext(deal_config(config, 'fasttext'), labels)

    print('[sentence_embedding]')
    sententce_embedding.run_sentence_embedding(deal_config(config, 'sentence_embedding'))

    print('[dgl]')
    lab_id = 9 # 实验唯一编号
    He_DGL.UnircaLab(deal_config(config, 'he_dgl')).do_lab(lab_id)