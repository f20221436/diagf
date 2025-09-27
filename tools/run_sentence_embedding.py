import argparse
from sentence_embedding import run_sentence_embedding

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="Word embedding dictionary (e.g., fasttext.pkl)")
    ap.add_argument("--train", required=True, help="Train text file")
    ap.add_argument("--test", required=True, help="Test text file")
    ap.add_argument("--output", required=True, help="Output sentence_embedding.pkl path")
    ap.add_argument("--k_s", type=int, required=True, help="Service number K_S")
    args = ap.parse_args()

    config = {
        "source_path": args.source,
        "train_path": args.train,
        "test_path": args.test,
        "save_path": args.output,
        "K_S": args.k_s,
    }

    run_sentence_embedding(config)
