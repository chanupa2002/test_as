import argparse
from train import train_model
from evaluate import evaluate_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Facial Emotion Recognition")
    parser.add_argument("--mode", type=str, choices=["train", "evaluate"], required=True)
    args = parser.parse_args()

    if args.mode == "train":
        train_model()
    elif args.mode == "evaluate":
        evaluate_model()
