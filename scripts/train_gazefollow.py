import warnings
warnings.simplefilter('ignore')
import argparse
from datetime import datetime
import numpy as np
import os
import random
import sys
from pathlib import Path
import time
import glob
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gazelle.dataloader import GazeDataset, collate_fn
from gazelle.backbone import configure_backbone_finetune, get_backbone_num_blocks
from gazelle.model import get_gazelle_model
from gazelle.utils import gazefollow_auc, gazefollow_l2

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default="gazelle_dinov3_vit_tiny")
parser.add_argument('--data_path', type=str, default='./data/gazefollow')
parser.add_argument('--ckpt_save_dir', type=str, default='./experiments')
parser.add_argument('--exp_name', type=str, default='train_gazefollow')
parser.add_argument('--log_dir', type=str, default='./runs')
parser.add_argument('--log_iter', type=int, default=10, help='how often to log loss during training')
parser.add_argument('--max_epochs', type=int, default=15)
parser.add_argument('--batch_size', type=int, default=60)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--n_workers', type=int, default=8)
parser.add_argument('--use_amp', action='store_true', help='enable mixed precision training')
parser.add_argument('--resume', type=str, default=None, help='path to checkpoint to resume training from')
parser.add_argument('--finetune', action='store_true', help='enable finetuning of the backbone')
parser.add_argument('--finetune_layers', type=int, default=2, help='number of final transformer blocks to finetune (<=0 means all)')
parser.add_argument('--backbone_lr', type=float, default=1e-5, help='learning rate for finetuned backbone parameters')
parser.add_argument('--backbone_weight_decay', type=float, default=0.0, help='weight decay applied to finetuned backbone parameters')
parser.add_argument('--grad_clip_norm', type=float, default=1.0, help='max norm for gradient clipping (<=0 to disable)')
parser.add_argument('--disable_sigmoid', action='store_true', help='predict raw logits and use BCEWithLogitsLoss')
parser.add_argument('--initial_freeze_epochs', type=int, default=10, help='number of epochs to keep initial finetune_layers before unfreezing more backbone layers')
parser.add_argument('--unfreeze_interval', type=int, default=3, help='epoch interval between progressive backbone unfreezing steps after initial_freeze_epochs')
parser.add_argument('--disable_progressive_unfreeze', action='store_true', help='keep backbone finetune_layers fixed for entire training')
parser.add_argument('--distill_teacher', type=str, default='gazelle_dinov3_vits16plus', help='teacher model name for knowledge distillation (only used when distill_weight > 0)')
parser.add_argument('--distill_weight', type=float, default=0.0, help='weight applied to the teacher loss; set <= 0 to disable distillation')
args = parser.parse_args()


def _collect_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state):
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _prepare_model_state_dict(model):
    return {k: v.detach().cpu() for k, v in model.get_gazelle_state_dict().items()}


def _move_optimizer_state_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, train_global_step,
                    best_min_l2, best_epoch, timestamp, exp_dir, log_dir, resume_args):
    resume_args = dict(resume_args)
    checkpoint = {
        "model": _prepare_model_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "train_global_step": train_global_step,
        "best_min_l2": best_min_l2,
        "best_epoch": best_epoch,
        "timestamp": timestamp,
        "exp_dir": exp_dir,
        "log_dir": log_dir,
        "args": resume_args,
        "use_amp": resume_args.get("use_amp"),
        "max_epochs": resume_args.get("max_epochs"),
        "rng_state": _collect_rng_state(),
    }
    torch.save(checkpoint, path)


def prune_epoch_checkpoints(exp_dir, keep=10):
    pattern = os.path.join(exp_dir, "epoch_*.pt")
    checkpoints = sorted(glob.glob(pattern))
    if len(checkpoints) <= keep:
        return
    to_delete = checkpoints[:-keep]
    for ckpt in to_delete:
        try:
            os.remove(ckpt)
            print(f"Removed old checkpoint {ckpt}")
        except OSError as error:
            print(f"WARNING: unable to remove old checkpoint {ckpt}: {error}")


def print_param_summary(rows):
    col1_width = max(len(label) for label, _ in rows + [("Category", 0)])
    value_strs = [f"{count / 1_000_000:.2f}" for _, count in rows]
    col2_width = max(len("Params [M]"), max(len(v) for v in value_strs))

    header = "┏" + "━" * (col1_width + 2) + "┳" + "━" * (col2_width + 2) + "┓"
    divider = "┗" + "━" * (col1_width + 2) + "┻" + "━" * (col2_width + 2) + "┛"
    footer = "└" + "-" * (col1_width + 2) + "┴" + "-" * (col2_width + 2) + "┘"

    print(header)
    print(f"┃ {'Category':<{col1_width}} ┃ {'Params [M]':>{col2_width}} ┃")
    print(divider)
    for (label, _), value in zip(rows, value_strs):
        print(f"| {label:<{col1_width}} | {value:>{col2_width}} |")
    print(footer)


def main():
    resume_checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False) if args.resume else None
    if resume_checkpoint is not None and "model" not in resume_checkpoint:
        raise ValueError(f"The checkpoint at {args.resume} does not contain full training state required for resuming.")

    saved_args = {}
    if resume_checkpoint is not None:
        saved_args = resume_checkpoint.get("args") or {}
        if "use_amp" not in saved_args and "use_amp" in resume_checkpoint:
            saved_args["use_amp"] = resume_checkpoint["use_amp"]

        def restore_arg(name):
            if name not in saved_args:
                return
            saved_value = saved_args[name]
            current_value = getattr(args, name, None)
            if current_value != saved_value:
                print(f"WARNING: resume checkpoint stored {name}={saved_value}, overriding current value {current_value}.")
            setattr(args, name, saved_value)

        for field in ("use_amp", "finetune", "finetune_layers", "backbone_lr", "backbone_weight_decay",
                      "disable_sigmoid", "initial_freeze_epochs", "unfreeze_interval", "disable_progressive_unfreeze",
                      "distill_teacher", "distill_weight"):
            restore_arg(field)

        timestamp = resume_checkpoint.get("timestamp") or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        exp_dir = resume_checkpoint.get("exp_dir") or os.path.dirname(os.path.abspath(args.resume))
        log_dir = resume_checkpoint.get("log_dir") or os.path.join(args.log_dir, args.exp_name, timestamp)
        os.makedirs(exp_dir, exist_ok=True)
        start_epoch = resume_checkpoint.get("epoch", -1) + 1
        train_global_step = resume_checkpoint.get("train_global_step", 0)
        best_min_l2 = resume_checkpoint.get("best_min_l2", 1.0)
        best_epoch = resume_checkpoint.get("best_epoch")
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        exp_dir = os.path.join(args.ckpt_save_dir, args.exp_name, timestamp)
        os.makedirs(exp_dir, exist_ok=True)
        log_dir = os.path.join(args.log_dir, args.exp_name, timestamp)
        start_epoch = 0
        train_global_step = 0
        best_min_l2 = 1.0
        best_epoch = None

    writer_kwargs = {"log_dir": log_dir}
    if resume_checkpoint is not None:
        writer_kwargs["purge_step"] = train_global_step
    writer = SummaryWriter(**writer_kwargs)
    scaler = GradScaler('cuda', enabled=args.use_amp)

    model, transform = get_gazelle_model(
        args.model_name,
        finetune_backbone=args.finetune,
        apply_sigmoid=not args.disable_sigmoid,
    )
    model.cuda()

    teacher_model = None
    distill_loss_fn = nn.MSELoss()
    distill_enabled = args.distill_weight is not None and args.distill_weight > 0
    if distill_enabled:
        if not args.distill_teacher:
            raise ValueError("distill_weight > 0 but no distill_teacher specified.")
        if args.distill_teacher == args.model_name:
            print("WARNING: distill_teacher matches student model; distillation likely ineffective but continuing.")
        teacher_model, _ = get_gazelle_model(
            args.distill_teacher,
            finetune_backbone=False,
            apply_sigmoid=not args.disable_sigmoid,
        )
        teacher_model.cuda()
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad_(False)
        print(f"Knowledge distillation enabled: teacher={args.distill_teacher}, weight={args.distill_weight}")
    else:
        print("Knowledge distillation disabled.")

    total_backbone_blocks = get_backbone_num_blocks(model.backbone) if args.finetune else 0
    base_finetune_layers = 0
    final_unfreeze_epoch = None

    if args.finetune:
        if args.finetune_layers <= 0:
            base_finetune_layers = total_backbone_blocks
        else:
            base_finetune_layers = min(args.finetune_layers, total_backbone_blocks)

    def _target_layers_for_epoch(epoch_index: int) -> int:
        if not args.finetune or total_backbone_blocks == 0:
            return 0
        if base_finetune_layers >= total_backbone_blocks:
            return total_backbone_blocks
        if args.disable_progressive_unfreeze:
            return base_finetune_layers
        if epoch_index < args.initial_freeze_epochs or args.unfreeze_interval <= 0:
            return base_finetune_layers
        steps = (epoch_index - args.initial_freeze_epochs) // max(args.unfreeze_interval, 1) + 1
        return min(total_backbone_blocks, base_finetune_layers + steps)

    current_finetune_layers = _target_layers_for_epoch(start_epoch)

    if args.finetune:
        backbone_trainable_params = configure_backbone_finetune(model.backbone, current_finetune_layers)
    else:
        for param in model.backbone.parameters():
            param.requires_grad = False
        backbone_trainable_params = []

    if args.finetune and total_backbone_blocks:
        extra_needed = max(0, total_backbone_blocks - base_finetune_layers)
        if args.disable_progressive_unfreeze or extra_needed <= 0:
            final_unfreeze_epoch = start_epoch if extra_needed <= 0 else None
        elif args.unfreeze_interval > 0:
            final_unfreeze_epoch = args.initial_freeze_epochs + (extra_needed - 1) * args.unfreeze_interval
        else:
            final_unfreeze_epoch = None

        msg = f"Backbone blocks: {total_backbone_blocks}. Initial trainable blocks: {current_finetune_layers}."
        if final_unfreeze_epoch is None:
            if args.disable_progressive_unfreeze and extra_needed > 0:
                msg += " Progressive unfreeze disabled by flag."
            else:
                msg += " Progressive unfreeze disabled (unable to reach all blocks)."
        else:
            msg += f" Final unfreeze epoch: {final_unfreeze_epoch}."
            if final_unfreeze_epoch >= args.max_epochs:
                msg += " (scheduled beyond max_epochs)"
        print(msg)

        if (not args.disable_progressive_unfreeze and
                base_finetune_layers < total_backbone_blocks and
                args.unfreeze_interval > 0):
            print(f"Progressive unfreeze scheduled after {args.initial_freeze_epochs} epochs, "
                  f"adding one block every {args.unfreeze_interval} epochs.")

    head_params = [param for name, param in model.named_parameters() if not name.startswith("backbone") and param.requires_grad]
    if not head_params:
        raise RuntimeError("No trainable parameters found for model head.")

    param_groups = [{"params": head_params, "lr": args.lr}]
    backbone_group_index = None
    if args.finetune:
        backbone_group_index = len(param_groups)
        param_groups.append({
            "params": backbone_trainable_params,
            "lr": args.backbone_lr,
            "weight_decay": args.backbone_weight_decay,
        })
    optimizer = torch.optim.Adam(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-7)
    loss_fn = nn.BCEWithLogitsLoss() if args.disable_sigmoid else nn.BCELoss()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    backbone_learnable = sum(p.numel() for p in backbone_trainable_params)
    head_learnable = trainable_params - backbone_learnable
    frozen_params = total_params - trainable_params

    table_rows = [
        ("Total params", total_params),
        ("Trainable params", trainable_params),
        ("Backbone trainable", backbone_learnable),
        ("Head trainable", head_learnable),
        ("Frozen params", frozen_params),
    ]
    print_param_summary(table_rows)
    print(f"Learnable parameters: {trainable_params} (backbone finetune: {backbone_learnable})")

    if resume_checkpoint is not None:
        model.load_gazelle_state_dict(resume_checkpoint["model"])
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        if resume_checkpoint.get("scheduler") is not None:
            scheduler.load_state_dict(resume_checkpoint["scheduler"])
        _move_optimizer_state_to_device(optimizer, next(model.parameters()).device)
        if args.use_amp and resume_checkpoint.get("scaler") is not None:
            scaler.load_state_dict(resume_checkpoint["scaler"])
        _restore_rng_state(resume_checkpoint.get("rng_state"))
        if resume_checkpoint.get("max_epochs") and resume_checkpoint["max_epochs"] != args.max_epochs:
            print(f"WARNING: resume checkpoint was created with max_epochs={resume_checkpoint['max_epochs']}, "
                  f"but current run uses max_epochs={args.max_epochs}.")
        print(f"Resuming training from {args.resume} at epoch {start_epoch}")

    train_dataset = GazeDataset('gazefollow', args.data_path, 'train', transform)
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.n_workers)
    eval_dataset = GazeDataset('gazefollow', args.data_path, 'test', transform)
    eval_dl = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.n_workers)

    if start_epoch >= args.max_epochs:
        print(f"Checkpoint epoch {start_epoch} is >= max_epochs {args.max_epochs}. Nothing to train.")
        writer.close()
        return

    for epoch in range(start_epoch, args.max_epochs):
        if args.finetune and backbone_group_index is not None:
            target_layers = _target_layers_for_epoch(epoch)
            if target_layers > current_finetune_layers:
                backbone_trainable_params = configure_backbone_finetune(model.backbone, target_layers)
                current_finetune_layers = target_layers
                optimizer.param_groups[backbone_group_index]["params"] = backbone_trainable_params
                print(f"Progressive unfreeze: last {current_finetune_layers}/{total_backbone_blocks} backbone blocks now trainable.")

        epoch_start_time = time.time()

        # TRAIN EPOCH
        model.train()
        for cur_iter, batch in enumerate(train_dl):
            imgs, bboxes, gazex, gazey, inout, heights, widths, heatmaps = batch

            optimizer.zero_grad()
            imgs_cuda = imgs.cuda()
            bbox_inputs = [[bbox] for bbox in bboxes]
            with autocast('cuda', enabled=args.use_amp):
                preds = model({"images": imgs_cuda, "bboxes": bbox_inputs})
                heatmap_preds = torch.stack(preds['heatmap']).squeeze(dim=1)
            loss_inputs = heatmap_preds.float() if args.use_amp else heatmap_preds
            loss_targets = heatmaps.cuda()
            loss = loss_fn(loss_inputs, loss_targets)

            if distill_enabled:
                with torch.no_grad():
                    with autocast('cuda', enabled=args.use_amp):
                        teacher_preds = teacher_model({"images": imgs_cuda, "bboxes": bbox_inputs})
                        teacher_heatmaps = torch.stack(teacher_preds['heatmap']).squeeze(dim=1)
                if args.use_amp:
                    teacher_heatmaps = teacher_heatmaps.float()
                if args.disable_sigmoid:
                    student_for_distill = torch.sigmoid(loss_inputs)
                    teacher_for_distill = torch.sigmoid(teacher_heatmaps)
                else:
                    student_for_distill = loss_inputs
                    teacher_for_distill = teacher_heatmaps
                distill_loss = distill_loss_fn(student_for_distill, teacher_for_distill)
                loss = loss + args.distill_weight * distill_loss
            scaler.scale(loss).backward()
            if args.grad_clip_norm and args.grad_clip_norm > 0:
                if args.use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            if cur_iter % args.log_iter == 0:
                writer.add_scalar("train/loss", loss.item(), train_global_step)
                if distill_enabled:
                    writer.add_scalar("train/distill_loss", distill_loss.item(), train_global_step)
                print("TRAIN EPOCH {}, iter {}/{}, loss={}".format(epoch, cur_iter, len(train_dl), round(loss.item(), 4)))
            train_global_step += 1

        scheduler.step()

        # EVAL EPOCH
        print("Running evaluation")
        model.eval()
        avg_l2s = []
        min_l2s = []
        aucs = []
        for cur_iter, batch in enumerate(eval_dl):
            imgs, bboxes, gazex, gazey, inout, heights, widths = batch

            with torch.no_grad():
                with autocast('cuda', enabled=args.use_amp):
                    preds = model({"images": imgs.cuda(), "bboxes": [[bbox] for bbox in bboxes]})

            heatmap_preds = torch.stack(preds['heatmap']).squeeze(dim=1)
            if args.disable_sigmoid:
                heatmap_preds = torch.sigmoid(heatmap_preds)
            if args.use_amp:
                heatmap_preds = heatmap_preds.float()
            for i in range(heatmap_preds.shape[0]):
                auc = gazefollow_auc(heatmap_preds[i], gazex[i], gazey[i], heights[i], widths[i])
                avg_l2, min_l2 = gazefollow_l2(heatmap_preds[i], gazex[i], gazey[i])
                aucs.append(auc)
                avg_l2s.append(avg_l2)
                min_l2s.append(min_l2)

        epoch_avg_l2 = np.mean(avg_l2s)
        epoch_min_l2 = np.mean(min_l2s)
        epoch_auc = np.mean(aucs)

        writer.add_scalar("eval/auc", epoch_auc, epoch)
        writer.add_scalar("eval/min_l2", epoch_min_l2, epoch)
        writer.add_scalar("eval/avg_l2", epoch_avg_l2, epoch)
        writer.add_scalar("train/epoch", epoch, epoch)
        print(f"EVAL EPOCH {epoch:03d}: AUC={round(epoch_auc, 4)}, Min L2={round(epoch_min_l2, 4)}, Avg L2={round(epoch_avg_l2, 4)}")

        epoch_elapsed = time.time() - epoch_start_time
        hours, remainder = divmod(epoch_elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"Epoch {epoch:03d} duration (train+eval): {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")

        if epoch_min_l2 < best_min_l2:
            best_min_l2 = epoch_min_l2
            best_epoch = epoch

        ckpt_path = os.path.join(exp_dir, f'epoch_{epoch:03d}.pt')
        save_checkpoint(
            ckpt_path,
            model,
            optimizer,
            scheduler,
            scaler if args.use_amp else None,
            epoch,
            train_global_step,
            best_min_l2,
            best_epoch,
            timestamp,
            exp_dir,
            log_dir,
            vars(args),
        )
        print(f"Saved checkpoint to {ckpt_path}")
        prune_epoch_checkpoints(exp_dir, keep=10)

        if best_epoch == epoch:
            best_filename = f"best_{epoch:03d}_{epoch_auc:.4f}_{best_min_l2:.4f}_{epoch_avg_l2:.4f}.pt"
            best_ckpt_path = os.path.join(exp_dir, best_filename)
            save_checkpoint(
                best_ckpt_path,
                model,
                optimizer,
                scheduler,
                scaler if args.use_amp else None,
                epoch,
                train_global_step,
                best_min_l2,
                best_epoch,
                timestamp,
                exp_dir,
                log_dir,
                vars(args),
            )
            print(f"Saved best checkpoint to {best_ckpt_path}")
            legacy_best = os.path.join(exp_dir, "best.pt")
            if os.path.exists(legacy_best) and legacy_best != best_ckpt_path:
                try:
                    os.remove(legacy_best)
                    print(f"Removed old checkpoint {legacy_best}")
                except OSError as error:
                    print(f"WARNING: unable to remove old checkpoint {legacy_best}: {error}")
            for best_path in glob.glob(os.path.join(exp_dir, "best_*.pt")):
                if best_path == best_ckpt_path:
                    continue
                try:
                    os.remove(best_path)
                    print(f"Removed old checkpoint {best_path}")
                except OSError as error:
                    print(f"WARNING: unable to remove old checkpoint {best_path}: {error}")

    print(f"Completed training. Best Min L2 of {round(best_min_l2, 4)} obtained at epoch {best_epoch:03d}")
    writer.close()

if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    main()
