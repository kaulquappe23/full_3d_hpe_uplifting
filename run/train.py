import argparse
import json
import logging
import os
import time
import datetime
import pickle
import random

import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter

import paths
from dataset.fit3d.dataset_generator import create_fit3d_datasets
from dataset.fit3d.keypoint_order import Fit3DOrder26P
from utils import path_utils, time_formatting
from utils.metric_history import MetricHistory

from model import build_model, UpliftPoseConfig, UpliftUpsampleConfig
from utils.load_weights import convert_tf_to_torch_weights, filter_weights

from run.train_utils import validation, train_step, save_best_weight, update_ema, update_loggers
from utils.rotation import RotType


def main(cfg) -> None:

    # set seeds
    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed_all(cfg.SEED)
    np.random.seed(cfg.SEED)
    random.seed(cfg.SEED)

    if hasattr(cfg, "ROT_REP") and cfg.ROT_REP:
        cfg.OUT_DIM = RotType[cfg.ROT_REP].value
        cfg.ROT_REP = RotType[cfg.ROT_REP]

    cur_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = time_formatting.generate_run_name(cfg, cur_time)
    output_dir = os.path.join(cfg.LOG_DIR, run_name)
    path_utils.mkdirs(output_dir)
    cfg.LOG_OUT_DIR = output_dir
    cfg.RUN_NAME = run_name

    checkpoint_template = os.path.join(output_dir, "cp_{:04d}.ckpt")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "training.log")),
            logging.StreamHandler()
        ])
    logging.info(f"Started training `{run_name}`")

    cfg.dump(os.path.join(cfg.LOG_OUT_DIR, "config.json"))

    if cfg.DATASET_NAME == "fit3d":
        cfg.AUGM_FLIP_KEYPOINT_ORDER = Fit3DOrder26P.flip_lr_indices()
        cfg.NUM_KEYPOINTS = Fit3DOrder26P._num_joints
        if cfg.BONES_LOSS_ENABLED:
            cfg.BONES = Fit3DOrder26P.bones()
    else:
        raise RuntimeError("Dataset not supported: " + cfg.DATASET_NAME)

    val_subset_name = cfg.VALIDATION_NAME
    if cfg.DATASET_NAME == "fit3d":
        train_ds, train_loader, val_ds, val_loader, val_batches = (
            create_fit3d_datasets(
                cfg=cfg,
                train_subset=cfg.TRAIN_NAME,
                val_subset=val_subset_name,
            )
        )
    else:
        raise NotImplementedError(f"`{cfg.DATASET_NAME}` is not supported.")


    cfg.VALIDATION_BATCHES = val_batches
    cfg.KEYPOINT_ORDER = eval(cfg.KEYPOINT_ORDER)

    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.DEVICE = device

    # build model
    model = build_model(cfg)
    if cfg.PRETRAINED_WEIGHTS:
        logging.info(f"Loading weights from {cfg.PRETRAINED_WEIGHTS}")
        if cfg.PRETRAINED_WEIGHTS.endswith(".pkl"):
            with open(cfg.PRETRAINED_WEIGHTS, "rb") as f:
                weights = pickle.load(f)
                weights = convert_tf_to_torch_weights(weights)
        else:
            weights = torch.load(cfg.PRETRAINED_WEIGHTS, map_location=device)
            if cfg.WEIGHTS_MODEL_NAME is not None:
                weights = weights[cfg.WEIGHTS_MODEL_NAME].state_dict()
        try:
            model.load_state_dict(weights)
        except RuntimeError:
            weights = filter_weights(weights, model)
            model.load_state_dict(weights, strict=False)
    model = model.to(device)

    # Keep an exponential moving average of the actual model
    ema_model = None
    ema_enabled = cfg.EMA_ENABLED
    if ema_enabled:
        logging.info("Cloning EMA model.")
        ema_model = build_model(config=cfg)
        ema_model.load_state_dict(model.state_dict())
        ema_model = ema_model.to(device)

    # set up optimizer and scheduler
    logging.info(f"Using {cfg.OPTIMIZER} optimizer")
    if cfg.OPTIMIZER == "AdamW":
        logging.info(cfg.OPTIMIZER_PARAMS)
        optimizer = torch.optim.AdamW(
            params=model.parameters(),
            weight_decay=cfg.OPTIMIZER_PARAMS["weight_decay"],
            lr=cfg.SCHEDULE_PARAMS["initial_learning_rate"],
            eps=cfg.SCHEDULE_PARAMS["epsilon"],
        )

        def weight_decay_hook(optimizer, args, kwargs):
            for group in optimizer.param_groups:
                # epoch finished
                change = iteration == cfg.training.steps_per_ep - 1
                if change:
                    group["weight_decay"] = (
                        group["weight_decay"] *
                        cfg.SCHEDULE_PARAMS["decay_rate"]
                    )
        if cfg.SCHEDULE_PARAMS["weight_decay"]:
            optimizer.register_step_post_hook(weight_decay_hook)
    elif cfg.OPTIMIZER == "Adam":
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=cfg.SCHEDULE_PARAMS["initial_learning_rate"],
        )
    else:
        raise ValueError(cfg.optimizer.name)

    # set up scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer, gamma=cfg.SCHEDULE_PARAMS["decay_rate"], last_epoch=-1
    )

    # set up checkpointing
    ckp_dict = {
        "optimizer": optimizer,
        "model": model,
        "epoch": 0,
        "ema_model": ema_model,
        "weights": {
            "pre_best_path": None,
            "pre_best_ema_path": None,
            "last_path": None,
        }
    }

    # restore training
    initial_epoch = 1
    if cfg.RESUME_TRAINING:
        res_train_weights = cfg.RESUME_TRAINING_CKPT
        logging.info(f"Restoring checkpoint from {res_train_weights}")
        checkpoint = torch.load(res_train_weights, map_location=device)
        initial_epoch = checkpoint["epoch"]
        optimizer.load_state_dict(checkpoint["optimizer"].state_dict())
        model.load_state_dict(checkpoint["model"].state_dict())
        if ema_enabled:
            ema_model.load_state_dict(checkpoint["ema_model"].state_dict())

        if cfg.SCHEDULE_PARAMS["weight_decay"]:
            for group in optimizer.param_groups:
                group["weight_decay"] = (
                    group["weight_decay"]
                    * cfg.SCHEDULE_PARAMS["decay_rate"] ** initial_epoch
                )

    # set global step
    global_step = (initial_epoch - 1) * cfg.STEPS_PER_EPOCH

    # Metrics and Tensorboard setup
    tb_writer = SummaryWriter(output_dir)
    metric_hist = MetricHistory() # ? only validation

    # dataset specific metricts, which are not included into train_*.yaml
    add_metrics = []
    higher_better = []
    if ema_enabled:
        for idx, metr in enumerate(cfg.METRIC_NAMES):
            hb = cfg.METRIC_HIGHER_BETTER[idx]
            add_metrics.append(f"{metr}/ema")
            higher_better.append(hb)
    cfg.METRIC_NAMES += add_metrics
    cfg.METRIC_HIGHER_BETTER += higher_better

    # add metrics to metric history
    for m, h in zip(cfg.METRIC_NAMES, cfg.METRIC_HIGHER_BETTER):
        metric_hist.add_metric(m, higher_is_better=h)

    assert cfg.BEST_CHECKPOINT_METRIC in cfg.METRIC_NAMES, \
        "Best checkpoint metric not included in dataset metrics"

    # update tensorboard
    tb_writer.add_scalar("train/LR", scheduler.get_last_lr()[0], 0)
    tb_writer.add_scalar(
        "train/weight_decay", optimizer.param_groups[0]["weight_decay"], 0)

    # Train loop
    train_iter = iter(train_loader)
    ema_decay = 0
    epoch_duration = 0.0
    ckp_interval = cfg.CHECKPOINT_INTERVAL
    val_ds_exist = cfg.VALIDATION_NAME is not None

    # train for n epochs
    epochs = cfg.EPOCHS
    for epoch in range(initial_epoch, epochs + 1):
        epoch_start = time.time()
        logging.info(f"## EPOCH {epoch} / {epochs}")
        ckp_dict["epoch"] = epoch

        epoch_loss = 0
        ep_loss_d = {}

        # (Global) Steps use 0-based index
        model.train()
        steps_per_ep = cfg.STEPS_PER_EPOCH
        for iteration in range(steps_per_ep):
            tick = time.time()

            # update ema decay
            if ema_enabled:
                ema_decay = min(
                    cfg.EMA_DECAY,
                    (1.0 + global_step) / (10.0 + global_step)
                )

            try:
                data = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                data = next(train_iter)

            # one iteration over ds
            model, optimizer, loss_dict = train_step(
                model=model,
                train_ds=train_ds,
                optimizer=optimizer,
                data=data,
                cfg=cfg,
            )
            if cfg.EMA_ENABLED:
                ema_model = update_ema(model, ema_model, ema_decay)

            epoch_loss += loss_dict.pop("main").item()
            tock = time.time()
            step_duration = tock - tick
            epoch_duration = tock - epoch_start

            # track losses of iteration
            for l, val in loss_dict.items():
                if iteration == 0:
                    ep_loss_d[l] = val
                else:
                    ep_loss_d[l] += val.item()

            if iteration % ckp_interval == 0:
                eta = (
                    (steps_per_ep - iteration - 1) / (iteration + 1)
                ) * epoch_duration
                eta_str = time_formatting.format_time(eta)
                logging.info(
                    f"{iteration}/{steps_per_ep} @ Epoch {epoch} "
                    f"(Step {step_duration:.3f}s, ETA {eta_str}): "
                    f"Mean loss {epoch_loss/(iteration+1):.6f}, "
                    f"lr: {scheduler.get_last_lr()[0]:.6f}"
                )

            global_step += 1
        scheduler.step()

        # Checkpoint
        if epoch % ckp_interval == 0:
            torch.save(ckp_dict, checkpoint_template.format(epoch))

        if steps_per_ep > 0:
            epoch_dur_str = time_formatting.format_time(epoch_duration)
            mean_step_dur_str = epoch_duration / steps_per_ep
            logging.info(f"Finished epoch {epoch} in {epoch_dur_str}, "
                         f"{mean_step_dur_str:.3f}s/step")

            def get_epoch_loss(name):
                if name in ep_loss_d.keys():
                    return ep_loss_d[name] / steps_per_ep
                return

            ep_loss_bones = get_epoch_loss("bones")
            ep_loss_angles = get_epoch_loss("angles")
            ep_loss_joints = get_epoch_loss("joints")
            ep_loss_mesh = get_epoch_loss("mesh")

            scalars = {
                "ep": epoch,
                "loss": epoch_loss / steps_per_ep,
                "loss_joints": ep_loss_joints,
                "LR": scheduler.get_last_lr()[0],
                "weight_decay": optimizer.param_groups[0]["weight_decay"],
                "step_duration": epoch_duration / steps_per_ep,
                "loss_bones": ep_loss_bones,
                "loss_angles": ep_loss_angles,
                "loss_mesh": ep_loss_mesh,
            }
            tb_writer, _ = update_loggers(scalars, tb_writer, subset="train")

            if epoch % cfg.VALIDATION_INTERVAL == 0 and val_ds_exist:
                results = validation(
                    model=model,
                    loader=val_loader,
                    ep=epoch,
                    cfg=cfg,
                )
                tb_writer, metric_hist = update_loggers(
                    results, tb_writer, metric_hist, subset="val")
                pre_best_weights_p = save_best_weight(
                    cfg, metric_hist, results["ep"], ckp_dict)
                ckp_dict["weights"]["pre_best_path"] = pre_best_weights_p
                if ema_enabled:
                    results = validation(
                        model=ema_model,
                        loader=val_loader,
                        ep=epoch,
                        prefix="ema",
                        cfg=cfg,
                    )
                    tb_writer, metric_hist = update_loggers(
                        results, tb_writer, metric_hist, "val", "ema")
                    pre_best_w_p_ema = save_best_weight(
                        cfg, metric_hist, results["ep"], ckp_dict, "ema")
                    ckp_dict["weights"]["pre_best_ema_path"] = pre_best_w_p_ema

        last_weights_path = os.path.join(output_dir, f"last_weights.pth")
        torch.save(ckp_dict, last_weights_path)
        ckp_dict["weights"]["last_path"] = last_weights_path

        metric_hist.save_csv(os.path.join(output_dir, "metrics_val.csv"))

    tb_writer.close()

    if val_ds_exist:
        logging.info(f"Best checkpoint results:")
        best_metric = cfg.BEST_CHECKPOINT_METRIC
        if best_metric is not None:
            metric_hist.print_all_for_best_metric(metric=best_metric)
        else:
            metric_hist.print_best()

    logging.info(f"Ended training `{run_name}`")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cfg-path", type=str, default=None,
                        help="path to  config file")
    parser.add_argument("-l", "--log", type=str,
                        help="log directory")
    args = parser.parse_args()

    print(f"Loading configuration from {args.cfg_path}")

    cfg = UpliftPoseConfig(config_file=args.cfg_path)

    if args.log is not None:
        cfg.LOG_DIR = args.log
    else:
        cfg.LOG_DIR = paths.LOG_DIR

    main(cfg)
