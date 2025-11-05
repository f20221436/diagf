# diagf/tools/create_sentence_embedding.py
import os
import sys
import argparse
import yaml

# --- Add necessary paths ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)    # repo root 'diagf'
sys.path.insert(0, parent_dir)

# --- Imports from project (unchanged logic) ---
try:
    import public_function as pf
    from transforms.events.sentence_embedding import sentence_embedding_main
except ImportError as e:
    print(f"[ERROR] Failed to import required modules: {e}")
    print(f"        Ensure 'public_function.py' is in '{parent_dir}'")
    print(f"        Ensure 'sentence_embedding.py' is in '{os.path.join(parent_dir, 'transforms', 'events')}'")
    sys.exit(1)


def _resolve_path_repo(p):
    """Resolve path relative to repo root (parent_dir) unless already absolute."""
    if not p:
        return None
    return os.path.normpath(p) if os.path.isabs(p) else os.path.normpath(os.path.join(parent_dir, p))


def main():
    parser = argparse.ArgumentParser(description="Create sentence embeddings using fastText outputs and event embeddings.")
    parser.add_argument('--config', type=str, default='gaia_config.yaml',
                        help='Path to the main configuration file (relative to repo root or absolute).')
    parser.add_argument('--use-augmented', action='store_true',
                        help="Prefer the augmented fastText output (train_da.txt) when available.")
    args = parser.parse_args()

    # --- Load config (try project helper, fallback to yaml) ---
    repo_cfg = None
    try:
        from public_function import get_config, deal_config
        repo_cfg = get_config(config_file=args.config)
    except Exception:
        cfg_path = args.config
        if not os.path.isabs(cfg_path):
            cfg_path = os.path.join(parent_dir, cfg_path)
        if not os.path.exists(cfg_path):
            print(f"[ERROR] Config file not found: {cfg_path}")
            sys.exit(1)
        with open(cfg_path, 'r', encoding='utf-8') as fh:
            repo_cfg = yaml.safe_load(fh)

    # --- Extract sections ---
    cfg_sent = repo_cfg.get('sentence_embedding', {}) if repo_cfg else {}
    cfg_fast = repo_cfg.get('fasttext', {}) if repo_cfg else {}

    # --- Candidate paths from config ---
    train_path_cfg = cfg_sent.get('train_path')
    test_path_cfg = cfg_sent.get('test_path')
    source_path_cfg = cfg_sent.get('source_path')
    save_path_cfg = cfg_sent.get('save_path')

    ft_train_da = cfg_fast.get('train_da_path')
    ft_train = cfg_fast.get('train_path')
    ft_test = cfg_fast.get('test_path')
    ft_source = cfg_fast.get('save_path')

    # --- pick train path (prefer augmented when requested) ---
    train_candidate = None
    if args.use_augmented and ft_train_da:
        train_candidate = ft_train_da
    if train_candidate is None:
        train_candidate = ft_train_da or train_path_cfg or ft_train

    test_candidate = ft_test or test_path_cfg
    source_candidate = ft_source or source_path_cfg

    # --- Resolve to absolute paths ---
    train_txt_path = _resolve_path_repo(train_candidate) if train_candidate else None
    test_txt_path = _resolve_path_repo(test_candidate) if test_candidate else None
    event_embedding_path = _resolve_path_repo(source_candidate) if source_candidate else None

    # -----------------------------
    # ✅ Determine save_base (fixed)
    # We now always derive it from the 'train_txt_path' anomalies folder.
    # -----------------------------
    save_base = None
    if train_txt_path:
        # Example: ...\MicroSS\anomalies\fasttext_temp\train_da.txt
        # We take everything up to 'anomalies' and append 'sentence_embedding'
        parts = os.path.normpath(train_txt_path).split(os.sep)
        idx = next((i for i in range(len(parts)-1, -1, -1)
                    if parts[i].lower() == 'anomalies'), None)
        if idx is not None:
            anomalies_dir = os.sep.join(parts[:idx + 1])
            save_base = os.path.join(anomalies_dir, 'sentence_embedding')

    # fallback to event_embedding or repo root if train path missing
    if save_base is None:
        if event_embedding_path and os.path.exists(event_embedding_path):
            anomalies_dir = os.path.dirname(os.path.dirname(event_embedding_path))
            save_base = os.path.join(anomalies_dir, 'sentence_embedding')
        else:
            save_base = os.path.join(parent_dir, 'MicroSS', 'anomalies', 'sentence_embedding')

    save_base = os.path.normpath(save_base)

    # --- Print resolved paths ---
    print("Using these resolved paths:")
    print("  train:", train_txt_path)
    print("  test: ", test_txt_path)
    print("  event_embedding:", event_embedding_path)
    print("  save_base:", save_base)

    # --- Validate existence of required inputs ---
    missing = []
    if not event_embedding_path or not os.path.exists(event_embedding_path):
        missing.append(('event_embedding', event_embedding_path))
    if not train_txt_path or not os.path.exists(train_txt_path):
        missing.append(('train_txt', train_txt_path))
    if not test_txt_path or not os.path.exists(test_txt_path):
        missing.append(('test_txt', test_txt_path))

    if missing:
        print("[ERROR] Required input files missing or not found:")
        for name, path in missing:
            print(f"  - {name}: {path}")
        print("Please check your config file and file locations. Exiting.")
        sys.exit(1)

    # ensure output directory exists
    out_dir = os.path.dirname(save_base)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Number of services
    num_services = cfg_sent.get('K_S') if cfg_sent else None
    if num_services is None:
        if cfg_fast and 'nodes' in cfg_fast:
            num_services = len(str(cfg_fast['nodes']).split())
        else:
            num_services = 10

    # --- Run embedding generation ---
    print("\n🚀 Starting Sentence Embedding Generation...")
    try:
        sentence_embedding_main(
            file_dict_path=event_embedding_path,
            train_path=train_txt_path,
            test_path=test_txt_path,
            save_path=save_base,
            service_num=num_services,
            pf_module=pf
        )
        print("\n🎉 Process completed successfully!")
        print(f"   Chunk files saved in: {save_base}_chunks")
        print(f"   Metadata saved to: {save_base}_metadata.pkl")
    except Exception as e:
        print("\n❌ An error occurred during embedding generation:", e)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
