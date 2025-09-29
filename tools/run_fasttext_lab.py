# tools/run_fasttext_lab.py
import os
import numpy as np
import fasttext
import pickle
import time
import random


# Import your project's helpers
import public_function as pf
from transforms.events.fasttext_with_DA import FastTextLab  # will use the class you posted

# ---- Small monkey-patch helpers to make FastTextLab robust ----
# We'll patch two methods at runtime: event_embedding_lab and a safer get_nearest helper.

def patched_event_embedding_lab(self, data_path):
    """
    Train (or load) a fastText model on data_path and return dict token->np.array(vector).
    Uses model.get_words() and model.get_word_vector() for compatibility.
    Skips tokens that look like fastText labels (__label__*).
    """
    model = self.method(data_path,
                        dim=self.config.get('vector_dim', 100),
                        minCount=self.config.get('minCount', 1),
                        minn=0, maxn=0,
                        epoch=self.config.get('epoch', 5))
    event_dict = {}
    # get list of words in a robust way
    words = []
    try:
        words = model.get_words()
    except Exception:
        # some bindings might expose model.words
        try:
            words = list(model.words)
        except Exception:
            words = []

    for w in words:
        if isinstance(w, str) and w.startswith("__label__"):
            continue
        try:
            vec = np.array(model.get_word_vector(w), dtype=np.float32)
            event_dict[w] = vec
        except Exception:
            # skip problematic tokens
            continue
    return event_dict

def safe_get_nearest_word(model, token):
    """
    Return a nearest neighbor string for token or token itself if no neighbor.
    Handles empty results.
    """
    try:
        neighs = model.get_nearest_neighbors(token)
        if not neighs:
            return token
        # neighs is list of tuples (score, word) or (word, score). Pick the last element of first tuple as word if needed.
        first = neighs[0]
        # try common tuple shapes
        if isinstance(first, tuple) and len(first) >= 2:
            # determine which entry looks like a string word
            if isinstance(first[0], str):
                return first[0]
            else:
                return first[-1]
        # fallback
        return str(first)
    except Exception:
        return token

# Apply patches
FastTextLab.event_embedding_lab = patched_event_embedding_lab

# Patch the w2v_DA function to use safe_get_nearest_word
_orig_w2v = FastTextLab.w2v_DA
def patched_w2v_DA(self):
    # copy of original logic but replacing nearest neighbor retrieval robustly
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
                chosen_text, label = anomaly_texts[random.randint(0, len(anomaly_texts) - 1)].split('\t')
                chosen_text_splits = chosen_text.split()
                if len(chosen_text_splits) < self.config['minCount']:
                    continue
                edit_event_ids = random.sample(range(len(chosen_text_splits)), min(self.config['edit_count'], len(chosen_text_splits)))
                for event_id in edit_event_ids:
                    nearest_event = safe_get_nearest_word(model, chosen_text_splits[event_id])
                    chosen_text_splits[event_id] = nearest_event
                da_train_data.append(
                    ' '.join(chosen_text_splits) + f'\t__label__{self.node_labels[node]}{self.anomaly_type_labels[anomaly_type]}')
                sample_count += 1

    with open(self.config['train_da_path'], 'w') as f:
        for text in da_train_data:
            f.write(text + '\n')

FastTextLab.w2v_DA = patched_w2v_DA

# ---- End patches ----

def generate_fasttext_pickle(config, cases, out_pkl_path=None):
    """
    Create a FastTextLab and run do_lab() to produce the pickle.
    If out_pkl_path is provided, move/rename the produced file to that path.
    """
    lab = FastTextLab(config, cases)
    start = time.time()
    lab.do_lab()
    end = time.time()
    print("fasttext done in %.2f s" % (end-start))
    produced_path = config['save_path']
    if out_pkl_path and produced_path != out_pkl_path:
        os.makedirs(os.path.dirname(out_pkl_path), exist_ok=True)
        os.replace(produced_path, out_pkl_path)
        print("moved", produced_path, "->", out_pkl_path)
    print("Saved embedding to:", out_pkl_path or produced_path)

if __name__ == "__main__":
    # Example usage: loads repo config and labels automatically (adjust paths if needed)
    # You must have the repo's public_function.get_config() / deal_config available
    from public_function import get_config, deal_config
    repo_cfg = get_config()
    cfg = deal_config(repo_cfg, 'fasttext')
    # load your label/cases structure. Adjust this path to where your repo stores labels
    # Typical places: config['labels_path'] or data/...; try pf.load('data/.../labels.pkl') if you know the path
    # We'll try common keys, otherwise try default demo label path
    labels = None
    for k in ['labels_path', 'label_path', 'labels', 'data_path']:
        if k in cfg and os.path.exists(cfg[k]):
            try:
                labels = pf.load(cfg[k])
                break
            except Exception:
                labels = None
    if labels is None:
        # fallback: attempting demo path used in repository
        demo_labels_path = os.path.join("DiagFusion","data","gaia","demo","demo_1100","labels.pkl")
        if os.path.exists(demo_labels_path):
            labels = pf.load(demo_labels_path)
        else:
            raise FileNotFoundError("Could not find labels/cases automatically. Set cfg key 'labels_path' or provide the cases object.")
    # run and save
    generate_fasttext_pickle(cfg, labels, out_pkl_path=cfg.get('save_path'))
