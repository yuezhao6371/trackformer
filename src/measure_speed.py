import torch
import numpy as np
import os
import toml
import logging
import wandb
import argparse
from model import TransformerClassifier
import utils.metrics_calculator_slim as metrics_calculator
import utils.training_utils as training_utils
import utils.data_utils as data_utils
import utils.output_utils as output_utils
import utils.wandb_utils as wandb_utils
import time


def load_config(config_path):
    """
    Load the TOML configuration file and return a dictionary.
    """
    with open(config_path, "r") as config_file:
        config = toml.load(config_file)
    return config


def load_model(config, device):
    model = TransformerClassifier(
        inputfeature_dim=config["model"]["inputfeature_dim"],
        num_classes=config["data"]["num_classes"],
        num_heads=config["model"]["num_heads"],
        embed_dim=config["model"]["embed_dim"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
    ).to(device)

    if (
        "checkpoint_path" not in config["model"]
        or not config["model"]["checkpoint_path"]
    ):
        logging.error("Checkpoint path must be provided for evaluation.")
    else:
        checkpoint = torch.load(config["model"]["checkpoint_path"])
        model.load_state_dict(checkpoint["model_state"])
        epoch = checkpoint["epoch"] + 1
        logging.info(
            f"Loaded model_state of epoch {epoch}. Ignoring optimizer_state and scheduler_state. Starting evaluation from checkpoint."
        )

    model.eval()
    return model


def setup_logging(config, output_dir):
    level = getattr(logging, config["logging"]["level"].upper(), logging.INFO)
    log_file = os.path.join(output_dir, "evaluation.log")
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )


def initialize_wandb(config, output_dir):
    wandb_logger = wandb_utils.WandbLogger(
        config=config["wandb"], output_dir=output_dir, job_type="evaluation"
    )
    wandb_logger.initialize()
    return wandb_logger


def evaluate(model, testloader, helperloader, truths_df, device, config, wandb_logger):
    test_metrics_calculator = metrics_calculator.MetricsCalculator(model.num_classes)

    cpu_time_ms = []
    gpu_time_ms = []
    start_gpu_event = torch.cuda.Event(enable_timing=True)
    end_gpu_event = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for (inputs, labels), (hit_ids, event_ids) in zip(testloader, helperloader):
            start_cpu_time = time.process_time_ns()
            inputs, labels = inputs.to(device), labels.to(device)
            end_cpu_time = time.process_time_ns()
            batch_cpu_time_ms = (end_cpu_time - start_cpu_time) / 1e6
            cpu_time_ms.append(batch_cpu_time_ms)

            start_gpu_event.record()
            outputs = model(inputs).view(-1, model.num_classes)
            labels = labels.view(-1)

            ### adding unsqueeze to compare with Nadia
            # test = torch.unsqueeze(outputs,0)
            # test = torch.unsqueeze(labels,0)
            ### adding unsqueeze to compare with Nadia

            end_gpu_event.record()
            torch.cuda.synchronize()
            batch_gpu_time_ms = start_gpu_event.elapsed_time(end_gpu_event)
            gpu_time_ms.append(batch_gpu_time_ms)
            logging.debug(f"Batch cpu time: {batch_cpu_time_ms:.6f} ms")
            logging.debug(f"Batch gpu time: {batch_gpu_time_ms:.6f} ms")

            test_metrics_calculator.add_true_score(
                hit_ids, event_ids, outputs, truths_df
            )

    avg_cpu_time_ms = sum(cpu_time_ms[1:]) / len(cpu_time_ms[1:])
    avg_gpu_time_ms = sum(gpu_time_ms[1:]) / len(gpu_time_ms[1:])
    logging.info(f"Avg CPU time: {avg_cpu_time_ms:.2f} ms")
    logging.info(f"Avg GPU time: {avg_gpu_time_ms:.2f} ms")
    wandb_logger.log(
        {
            "avg_cpu_time_ms": avg_cpu_time_ms,
            "avg_gpu_time_ms": avg_gpu_time_ms,
        }
    )

    all_true_scores = test_metrics_calculator.get_all_true_scores()
    true_score = np.mean(all_true_scores) if all_true_scores else 0

    logging.info(f"True score: {true_score:.2f}%")
    wandb_logger.log({"true_score": true_score})


def main(config_path):
    config = load_config(config_path)
    output_dir = output_utils.unique_output_dir(config)  # with time stamp
    output_utils.copy_config_to_output(config_path, output_dir)
    setup_logging(config, output_dir)
    logging.info(f"output_dir: {output_dir}%")
    wandb_logger = initialize_wandb(config, output_dir)

    start_wallclock = time.perf_counter_ns()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    model = load_model(config, device)
    loaders = data_utils.load_dataloader(config, device, mode="eval")
    data_loader = loaders["test"]
    helper_loader = loaders["test_helper"]
    truths_df = data_utils.load_truths(config)

    logging.info("Started evaluation.")
    evaluate(model, data_loader, helper_loader, truths_df, device, config, wandb_logger)
    logging.info("Finished evaluation.")
    end_wallclock = time.perf_counter_ns()
    total_wallclock_time_ms = (end_wallclock - start_wallclock) / 1e6
    logging.info(
        f"Total wallclock time (including scoring): {total_wallclock_time_ms:.2f} ms"
    )
    wandb_logger.log({"total_wallclock_time_ms": total_wallclock_time_ms})

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total Trainable Parameters: {total_params}")
    wandb_logger.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Measuring the inference speed of a model with a given config file."
    )
    parser.add_argument(
        "config_path", type=str, help="Path to the configuration TOML file."
    )

    # Parse arguments
    args = parser.parse_args()
    main(args.config_path)
