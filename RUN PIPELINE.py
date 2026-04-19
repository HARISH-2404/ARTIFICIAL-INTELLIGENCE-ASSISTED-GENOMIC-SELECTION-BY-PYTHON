import argparse
import logging
from train_model import train
from predict import predict
from utils import load_config

logging.basicConfig(level=logging.INFO)

def run_pipeline(config_path):
    config = load_config(config_path)

    logging.info("Starting Training Pipeline...")
    model = train(config)

    logging.info("Running Predictions...")
    results = predict(model, config)

    logging.info("Pipeline Completed Successfully")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    run_pipeline(args.config)
