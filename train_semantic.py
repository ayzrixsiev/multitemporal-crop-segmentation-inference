"""
Main script for semantic experiments
Author: Vivien Sainte Fare Garnot (github/VSainteuf)
License: MIT
"""

import argparse
import json
import os
import pickle as pkl
import pprint
import time

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

from src import utils, model_utils
from src.dataset import Pastis_Dataset
from src.learning.augment import SpatialAugment
from src.learning.class_weights import class_weights_for_training, keep_mask
from src.learning.meters import AverageValueMeter
from src.learning.metrics import confusion_matrix_analysis
from src.learning.miou import IoU
from src.learning.weight_init import weight_init

parser = argparse.ArgumentParser()
# Model parameters
parser.add_argument(
    "--model",
    default="utae",
    type=str,
    help="Type of architecture to use. Can be one of: (utae/unet3d/fpn/convlstm/convgru/uconvlstm/buconvlstm)",
)
## U-TAE Hyperparameters
parser.add_argument("--encoder_widths", default="[64,64,64,128]", type=str)
parser.add_argument("--decoder_widths", default="[32,32,64,128]", type=str)
parser.add_argument("--out_conv", default="[32, 20]")
parser.add_argument("--str_conv_k", default=4, type=int)
parser.add_argument("--str_conv_s", default=2, type=int)
parser.add_argument("--str_conv_p", default=1, type=int)
parser.add_argument("--agg_mode", default="att_group", type=str)
parser.add_argument("--encoder_norm", default="group", type=str)
parser.add_argument("--n_head", default=16, type=int)
parser.add_argument("--d_model", default=256, type=int)
parser.add_argument("--d_k", default=4, type=int)

# Set-up parameters
parser.add_argument(
    "--dataset_folder",
    default="",
    type=str,
    help="Path to the folder where the results are saved.",
)
parser.add_argument(
    "--res_dir",
    default="./results",
    help="Path to the folder where the results should be stored",
)
parser.add_argument(
    "--num_workers", default=4, type=int, help="Number of data loading workers"
)
parser.add_argument("--rdm_seed", default=1, type=int, help="Random seed")
parser.add_argument(
    "--device",
    default="cuda",
    type=str,
    help="Name of device to use for tensor computations (cuda/cpu)",
)
parser.add_argument(
    "--display_step",
    default=50,
    type=int,
    help="Interval in batches between display of training metrics",
)
parser.add_argument(
    "--cache",
    dest="cache",
    action="store_true",
    help="If specified, the whole dataset is kept in RAM",
)
# Training parameters
parser.add_argument("--epochs", default=100, type=int, help="Number of epochs per fold")
parser.add_argument("--batch_size", default=4, type=int, help="Batch size")
parser.add_argument("--lr", default=0.001, type=float, help="Learning rate")
parser.add_argument(
    "--lr_schedule",
    default="none",
    type=str,
    choices=["none", "cosine"],
    help="Not upstream. 'cosine' anneals the learning rate over --epochs with "
    "CosineAnnealingLR, stepped once per epoch.",
)
parser.add_argument(
    "--class_weights",
    default="none",
    type=str,
    choices=["none", "inverse", "sqrt_inverse"],
    help="Not upstream. Weight the loss by label frequency over the training "
    "folds only.",
)
parser.add_argument(
    "--augment",
    dest="augment",
    action="store_true",
    help="Not upstream. Random flips and quarter-turns of the training patches.",
)
parser.add_argument("--mono_date", default=None, type=str)
parser.add_argument("--ref_date", default="2018-09-01", type=str)
parser.add_argument(
    "--fold",
    default=None,
    type=int,
    help="Do only one of the five fold (between 1 and 5)",
)
parser.add_argument("--num_classes", default=20, type=int)
parser.add_argument("--ignore_index", default=-1, type=int)
parser.add_argument("--pad_value", default=0, type=float)
parser.add_argument("--padding_mode", default="reflect", type=str)
parser.add_argument(
    "--val_every",
    default=1,
    type=int,
    help="Interval in epochs between two validation steps.",
)
parser.add_argument(
    "--val_after",
    default=0,
    type=int,
    help="Do validation only after that many epochs.",
)
parser.add_argument(
    "--resume",
    dest="resume",
    action="store_true",
    help="If specified, continue the run already in res_dir instead of starting over",
)

list_args = ["encoder_widths", "decoder_widths", "out_conv"]
parser.set_defaults(cache=False, resume=False, augment=False)


def iterate(
    model, data_loader, criterion, config, optimizer=None, mode="train", device=None
):
    loss_meter = AverageValueMeter()
    iou_meter = IoU(
        num_classes=config.num_classes,
        ignore_index=config.ignore_index,
        cm_device=config.device,
    )

    t_start = time.time()
    for i, batch in enumerate(data_loader):
        if device is not None:
            batch = recursive_todevice(batch, device)
        (x, dates), y = batch
        y = y.long()

        if mode != "train":
            with torch.no_grad():
                out = model(x, batch_positions=dates)
        else:
            optimizer.zero_grad()
            out = model(x, batch_positions=dates)

        loss = criterion(out, y)
        if mode == "train":
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            pred = out.argmax(dim=1)
        iou_meter.add(pred, y)
        loss_meter.add(loss.item())

        if (i + 1) % config.display_step == 0:
            miou, acc = iou_meter.get_miou_acc()
            print(
                "Step [{}/{}], Loss: {:.4f}, Acc : {:.2f}, mIoU {:.2f}".format(
                    i + 1, len(data_loader), loss_meter.value()[0], acc, miou
                )
            )

    t_end = time.time()
    total_time = t_end - t_start
    print("Epoch time : {:.1f}s".format(total_time))
    miou, acc = iou_meter.get_miou_acc()
    metrics = {
        "{}_accuracy".format(mode): acc,
        "{}_loss".format(mode): loss_meter.value()[0],
        "{}_IoU".format(mode): miou,
        "{}_epoch_time".format(mode): total_time,
    }

    if mode == "test":
        return metrics, iou_meter.conf_metric.value()  # confusion matrix
    else:
        return metrics


def recursive_todevice(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, dict):
        return {k: recursive_todevice(v, device) for k, v in x.items()}
    else:
        return [recursive_todevice(c, device) for c in x]


def prepare_output(config):
    os.makedirs(config.res_dir, exist_ok=True)
    for fold in range(1, 6):
        os.makedirs(os.path.join(config.res_dir, "Fold_{}".format(fold)), exist_ok=True)


def checkpoint(fold, log, config):
    with open(
        os.path.join(config.res_dir, "Fold_{}".format(fold), "trainlog.json"), "w"
    ) as outfile:
        json.dump(log, outfile, indent=4)


def resume_from(fold, model, optimizer, scheduler, config):
    """
    Not upstream. Reload the state written by save_checkpoint_last() so a killed run
    can carry on instead of restarting from random weights, which is what Kaggle's
    12 hour session cap forces you into at 100 epochs.

    Returns (start_epoch, best_mIoU, trainlog). If there is nothing to resume from,
    returns a fresh start so that --resume is always safe to leave switched on.

    The scheduler is restored after the optimizer, which is what carries the
    resumed learning rate. A checkpoint written before --lr_schedule existed has
    no scheduler entry, so the schedule is replayed instead.
    """
    fold_dir = os.path.join(config.res_dir, "Fold_{}".format(fold))
    last_path = os.path.join(fold_dir, "last.pth.tar")
    log_path = os.path.join(fold_dir, "trainlog.json")

    if not (os.path.exists(last_path) and os.path.exists(log_path)):
        print("Nothing to resume in {}, starting from scratch.".format(fold_dir))
        return 1, 0, {}

    state = torch.load(last_path, map_location="cpu")
    model.load_state_dict(state["state_dict"])
    optimizer.load_state_dict(state["optimizer"])

    if scheduler is not None:
        if state.get("scheduler") is not None:
            scheduler.load_state_dict(state["scheduler"])
        else:
            print(
                "Checkpoint holds no schedule state; replaying the "
                "learning-rate schedule over {} epochs.".format(state["epoch"])
            )
            # CosineAnnealingLR.step() multiplies the rate already in the
            # optimizer, so replay has to start from base_lr or it decays twice.
            for group, base_lr in zip(optimizer.param_groups, scheduler.base_lrs):
                group["lr"] = base_lr
            for _ in range(state["epoch"]):
                scheduler.step()
        print("Learning rate resumes at {:.3e}".format(optimizer.param_groups[0]["lr"]))

    # json turns the integer epoch keys into strings on the way out, so undo that
    with open(log_path) as file:
        trainlog = {int(k): v for k, v in json.loads(file.read()).items()}

    start_epoch = state["epoch"] + 1
    best_mIoU = state["best_mIoU"]
    print(
        "Resuming fold {} at epoch {}/{}, best mIoU so far {:.4f}".format(
            fold, start_epoch, config.epochs, best_mIoU
        )
    )
    return start_epoch, best_mIoU, trainlog


def save_checkpoint_last(fold, epoch, best_mIoU, model, optimizer, scheduler, config):
    """
    Not upstream. model.pth.tar only ever holds the *best* epoch, which is the right
    thing to test with but the wrong thing to resume from -- it would silently throw
    away every epoch since the last improvement. So keep a separate latest-epoch
    checkpoint purely for resuming.

    The schedule state is written only when there is one, so that a run at the
    default --lr_schedule none produces the same file it did before.
    """
    state = {
        "epoch": epoch,
        "best_mIoU": best_mIoU,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    torch.save(
        state,
        os.path.join(config.res_dir, "Fold_{}".format(fold), "last.pth.tar"),
    )


def jsonable(obj):
    """
    Not upstream. Convert numpy numbers to plain Python ones, and nan to None.

    A class the model never predicts gives precision 0/0 = nan, which json.dumps
    writes as a bare NaN -- not valid JSON, and rejected by the viewer's
    JSON.parse.
    """
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return None if np.isnan(value) else value
    return obj


def per_class_performance(fold, conf_mat, config):
    """
    Not upstream. Write per_class.json for a single fold.

    overall_performance() needs all five folds, so a one-fold run never got
    per-class numbers. The ignored class is zeroed rather than deleted, which
    gives the same scores but keeps the class ids of
    webapp/pastis_meta.py:CLASS_NAMES.
    """
    conf_mat = np.asarray(conf_mat, dtype=np.float64).copy()
    keep = keep_mask(config.num_classes, config.ignore_index)
    conf_mat[~keep, :] = 0
    conf_mat[:, ~keep] = 0

    # A class with no predictions leaves 0/0 in the precision column.
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class, overall = confusion_matrix_analysis(conf_mat)

    per_class = {k: v for k, v in per_class.items() if keep[int(k)]}
    report = {
        "fold": fold,
        "ignored_class": config.ignore_index,
        "per_class": per_class,
        "overall": overall,
    }
    with open(
        os.path.join(config.res_dir, "Fold_{}".format(fold), "per_class.json"), "w"
    ) as outfile:
        outfile.write(json.dumps(jsonable(report), indent=4))

    print("Per-class IoU (fold {}):".format(fold))
    for class_id in sorted(per_class, key=int):
        print(
            "  class {:>3}  IoU {:>6.2f}  precision {:>6.2f}  recall {:>6.2f}".format(
                class_id,
                100 * per_class[class_id]["IoU"],
                100 * per_class[class_id]["Precision"],
                100 * per_class[class_id]["Recall"],
            )
        )


def save_results(fold, metrics, conf_mat, config):
    with open(
        os.path.join(config.res_dir, "Fold_{}".format(fold), "test_metrics.json"), "w"
    ) as outfile:
        json.dump(metrics, outfile, indent=4)

    if torch.is_tensor(conf_mat):
        conf_mat = conf_mat.cpu().numpy()
    pkl.dump(
        conf_mat,
        open(
            os.path.join(config.res_dir, "Fold_{}".format(fold), "conf_mat.pkl"), "wb"
        ),
    )

    per_class_performance(fold, conf_mat, config)


def overall_performance(config):
    cm = np.zeros((config.num_classes, config.num_classes))
    for fold in range(1, 6):
        cm += pkl.load(
            open(
                os.path.join(config.res_dir, "Fold_{}".format(fold), "conf_mat.pkl"),
                "rb",
            )
        )

    if config.ignore_index is not None:
        cm = np.delete(cm, config.ignore_index, axis=0)
        cm = np.delete(cm, config.ignore_index, axis=1)

    _, perf = confusion_matrix_analysis(cm)

    print("Overall performance:")
    print("Acc: {},  IoU: {}".format(perf["Accuracy"], perf["MACRO_IoU"]))

    # jsonable() is not upstream: it keeps a nan out of the written JSON.
    with open(os.path.join(config.res_dir, "overall.json"), "w") as file:
        file.write(json.dumps(jsonable(perf), indent=4))


def main(config):
    fold_sequence = [
        [[1, 2, 3], [4], [5]],
        [[2, 3, 4], [5], [1]],
        [[3, 4, 5], [1], [2]],
        [[4, 5, 1], [2], [3]],
        [[5, 1, 2], [3], [4]],
    ]

    np.random.seed(config.rdm_seed)
    torch.manual_seed(config.rdm_seed)
    prepare_output(config)
    device = torch.device(config.device)

    fold_sequence = (
        fold_sequence if config.fold is None else [fold_sequence[config.fold - 1]]
    )
    for fold, (train_folds, val_fold, test_fold) in enumerate(fold_sequence):
        if config.fold is not None:
            fold = config.fold - 1

        # Dataset definition
        dt_args = dict(
            folder=config.dataset_folder,
            norm=True,
            reference_date=config.ref_date,
            mono_date=config.mono_date,
            target="semantic",
            sats=["S2"],
        )

        dt_train = Pastis_Dataset(**dt_args, folds=train_folds, cache=config.cache)
        dt_val = Pastis_Dataset(**dt_args, folds=val_fold, cache=config.cache)
        dt_test = Pastis_Dataset(**dt_args, folds=test_fold)

        # Not upstream. Training set only; val and test stay unaugmented.
        train_set = dt_train
        if config.augment:
            train_set = SpatialAugment(train_set)
            print("Spatial augmentation ON: random flips and quarter-turns.")

        collate_fn = lambda x: utils.pad_collate(x, pad_value=config.pad_value)
        train_loader = data.DataLoader(
            train_set,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
            num_workers=config.num_workers,
        )
        val_loader = data.DataLoader(
            dt_val,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
            num_workers=config.num_workers,
        )
        test_loader = data.DataLoader(
            dt_test,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
            num_workers=config.num_workers,
        )

        print(
            "Train {}, Val {}, Test {}".format(len(dt_train), len(dt_val), len(dt_test))
        )

        # Model definition
        model = model_utils.get_model(config, mode="semantic")
        config.N_params = utils.get_ntrainparams(model)
        with open(os.path.join(config.res_dir, "conf.json"), "w") as file:
            file.write(json.dumps(vars(config), indent=4))
        print(model)
        print("TOTAL TRAINABLE PARAMETERS :", config.N_params)
        print("Trainable layers:")
        for name, p in model.named_parameters():
            if p.requires_grad:
                print(name)
        model = model.to(device)
        model.apply(weight_init)

        # Optimizer and Loss
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

        # Not upstream. At --class_weights none this is upstream's
        # torch.ones with a 0 at the ignored class.
        weights = class_weights_for_training(
            config, dataset=dt_train, folds=train_folds, device=device
        )
        criterion = nn.CrossEntropyLoss(weight=weights)

        # Not upstream. See --lr_schedule.
        scheduler = None
        if config.lr_schedule == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.epochs
            )

        # Training loop
        trainlog = {}
        best_mIoU = 0
        start_epoch = 1
        if config.resume:
            start_epoch, best_mIoU, trainlog = resume_from(
                fold + 1, model, optimizer, scheduler, config
            )
        for epoch in range(start_epoch, config.epochs + 1):
            print("EPOCH {}/{}".format(epoch, config.epochs))

            model.train()
            train_metrics = iterate(
                model,
                data_loader=train_loader,
                criterion=criterion,
                config=config,
                optimizer=optimizer,
                mode="train",
                device=device,
            )
            # Not upstream. The rate this epoch ran at, read before the step below.
            if scheduler is not None:
                train_metrics["train_lr"] = optimizer.param_groups[0]["lr"]
            if epoch % config.val_every == 0 and epoch > config.val_after:
                print("Validation . . . ")
                model.eval()
                val_metrics = iterate(
                    model,
                    data_loader=val_loader,
                    criterion=criterion,
                    config=config,
                    optimizer=optimizer,
                    mode="val",
                    device=device,
                )

                print(
                    "Loss {:.4f},  Acc {:.2f},  IoU {:.4f}".format(
                        val_metrics["val_loss"],
                        val_metrics["val_accuracy"],
                        val_metrics["val_IoU"],
                    )
                )

                trainlog[epoch] = {**train_metrics, **val_metrics}
                checkpoint(fold + 1, trainlog, config)
                if val_metrics["val_IoU"] >= best_mIoU:
                    best_mIoU = val_metrics["val_IoU"]
                    torch.save(
                        {
                            "epoch": epoch,
                            "state_dict": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                        },
                        os.path.join(
                            config.res_dir, "Fold_{}".format(fold + 1), "model.pth.tar"
                        ),
                    )
            else:
                trainlog[epoch] = {**train_metrics}
                checkpoint(fold + 1, trainlog, config)

            # Not upstream. Step before the checkpoint is written, so the saved
            # state is the one the next epoch starts from.
            if scheduler is not None:
                scheduler.step()

            save_checkpoint_last(
                fold + 1, epoch, best_mIoU, model, optimizer, scheduler, config
            )

        print("Testing best epoch . . .")
        model.load_state_dict(
            torch.load(
                os.path.join(
                    config.res_dir, "Fold_{}".format(fold + 1), "model.pth.tar"
                )
            )["state_dict"]
        )
        model.eval()

        test_metrics, conf_mat = iterate(
            model,
            data_loader=test_loader,
            criterion=criterion,
            config=config,
            optimizer=optimizer,
            mode="test",
            device=device,
        )
        print(
            "Loss {:.4f},  Acc {:.2f},  IoU {:.4f}".format(
                test_metrics["test_loss"],
                test_metrics["test_accuracy"],
                test_metrics["test_IoU"],
            )
        )
        save_results(fold + 1, test_metrics, conf_mat, config)

    if config.fold is None:
        overall_performance(config)


if __name__ == "__main__":
    config = parser.parse_args()
    for k, v in vars(config).items():
        if k in list_args and v is not None:
            v = v.replace("[", "")
            v = v.replace("]", "")
            config.__setattr__(k, list(map(int, v.split(","))))

    assert config.num_classes == config.out_conv[-1]

    pprint.pprint(config)
    main(config)
