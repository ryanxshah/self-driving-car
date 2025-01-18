"""
Usage:
    python3 -m homework.train_planner --your_args here
"""

import torch
import argparse
import numpy as np
import torch.utils.tensorboard as tb

from pathlib import Path
from datetime import datetime
from .metrics import PlannerMetric
from homework.models import load_model, save_model
from homework.datasets.road_dataset import load_data

def train(
        exp_dir: str = "logs",
        transform_pipeline = "state_only",
        num_workers = 4,
        model_name: str = "cnn_planner",
        num_epoch: int = 50,
        lr: float = 1e-3,
        batch_size: int = 128,
        seed: int = 2024,
        **kwargs,
):
    
    # SET DEVICE
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # SET SEED
    torch.manual_seed(seed)
    np.random.seed(seed)

    # SET LOGGER
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # LOAD MODEL AND SET MODEL IN TRAINING MODE
    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    # LOAD DATA
    train_data = load_data("drive_data/train", shuffle=True, batch_size=batch_size, num_workers=2)
    val_data = load_data("drive_data/val", shuffle=False)

    # CREATE OPTIMIZER AND LOSS FUNCTION
    loss_func = torch.nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    global_step = 0
    
    # CREATE METRICS
    train_planner_metric = PlannerMetric()
    val_planner_metric = PlannerMetric()


    for epoch in range(num_epoch):

        model.train()
        for batch in train_data:
            image = batch["image"].to(device)
            track_left = batch["track_left"].to(device)
            track_right = batch["track_right"].to(device)
            waypoints = batch["waypoints"].to(device)
            waypoints_mask = batch["waypoints_mask"].to(device)

            preds = model(image)

            loss = loss_func(preds[waypoints_mask], waypoints[waypoints_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_planner_metric.add(preds, waypoints, waypoints_mask)

            global_step += 1

        with torch.inference_mode():

            model.eval()
            for batch in val_data:
                image = batch["image"].to(device)
                track_left = batch["track_left"].to(device)
                track_right = batch["track_right"].to(device)
                waypoints = batch["waypoints"].to(device)
                waypoints_mask = batch["waypoints_mask"].to(device)
                
                preds = model(image)
                val_planner_metric.add(preds, waypoints, waypoints_mask)

        train_metrics = train_planner_metric.compute()
        val_metrics = val_planner_metric.compute()

        # PRINT METRICS EVERY EPOCH
        print(
            f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
            "\n train: "
            f"longitudinal_err={train_metrics['longitudinal_error']:.4f} "
            f"lateral_err={train_metrics['lateral_error']:.4f} "
            f"l1_err={train_metrics['l1_error']:.4f} "
            "\n val: "
            f"longitudinal_err={val_metrics['longitudinal_error']:.4f} "
            f"lateral_err={val_metrics['lateral_error']:.4f} "
            f"l1_err={val_metrics['l1_error']:.4f} "
            "\n"
        )
        

    # save and overwrite the model in the root directory for grading
    save_model(model)

    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)

    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=3)

    # pass all arguments to train
    train(**vars(parser.parse_args()))

