import warnings
warnings.simplefilter('ignore')
import argparse
from datetime import datetime
import numpy as np
import os
import random
import sys
from pathlib import Path
import glob
import math
from sklearn.metrics import average_precision_score
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gazelle.dataloader import GazeDataset, collate_fn
from gazelle.backbone import configure_backbone_finetune
from gazelle.model import get_gazelle_model
from gazelle.utils import vat_auc, vat_l2

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="gazelle_dinov2_vitb14_inout")
parser.add_argument('--init_ckpt', type=str, default='./checkpoints/gazelle_dinov2_vitb14.pt', help='checkpoint for initialization (trained on GazeFollow)')
parser.add_argument('--data_path', type=str, default='./data/videoattentiontarget')
parser.add_argument('--frame_sample_every', type=int, default=6)
parser.add_argument('--ckpt_save_dir', type=str, default='./experiments')
parser.add_argument('--exp_name', type=str, default='train_vat')
parser.add_argument('--log_dir', type=str, default='./runs')
parser.add_argument('--log_iter', type=int, default=10, help='how often to log loss during training')
parser.add_argument('--max_epochs', type=int, default=8)
parser.add_argument('--batch_size', type=int, default=60)
parser.add_argument('--inout_loss_lambda', type=float, default=1.0)
parser.add_argument('--lr_non_inout', type=float, default=1e-5)
parser.add_argument('--lr_inout', type=float, default=1e-2)
parser.add_argument('--n_workers', type=int, default=8)
parser.add_argument('--resume', type=str, default=None, help='path to checkpoint to resume training from')
parser.add_argument('--finetune', action='store_true', help='enable finetuning of the backbone')
parser.add_argument('--finetune_layers', type=int, default=2, help='number of final transformer blocks to finetune (<=0 means all)')
parser.add_argument('--backbone_lr', type=float, default=1e-5, help='learning rate for finetuned backbone parameters')
parser.add_argument('--backbone_weight_decay', type=float, default=0.0, help='weight decay applied to finetuned backbone parameters')
parser.add_argument('--distill_teacher', type=str, default='gazelle_dinov3_vits16plus',
    help='teacher model name for knowledge distillation (only used when distill_weight > 0)')
parser.add_argument('--distill_weight', type=float, default=0.0,
    help='weight applied to the teacher heatmap loss; set <= 0 to disable distillation')
parser.add_argument('--distill_temp_start', type=float, default=1.0,
    help='initial temperature for distillation soft targets')
parser.add_argument('--distill_temp_end', type=float, default=4.0,
    help='final temperature reached via cosine schedule')
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


def _cosine_anneal(start: float, end: float, step: int, total_steps: int) -> float:
    if total_steps <= 0:
        return end
    progress = min(max(step / total_steps, 0.0), 1.0)
    cosine = 0.5 * (1.0 - math.cos(math.pi * progress))
    return start + (end - start) * cosine


def save_checkpoint(path, model, optimizer, epoch, train_global_step, best_inout_ap,
                    timestamp, exp_dir, log_dir, resume_args):
    resume_args = dict(resume_args)
    checkpoint = {
        "model": _prepare_model_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "train_global_step": train_global_step,
        "best_inout_ap": best_inout_ap,
        "timestamp": timestamp,
        "exp_dir": exp_dir,
        "log_dir": log_dir,
        "args": resume_args,
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
    resume_checkpoint = torch.load(args.resume, map_location='cpu') if args.resume else None
    if resume_checkpoint is not None and "model" not in resume_checkpoint:
        raise ValueError(f"The checkpoint at {args.resume} does not contain full training state required for resuming.")

    saved_args = {}
    if resume_checkpoint is not None:
        saved_args = resume_checkpoint.get("args") or {}

        def restore_arg(name):
            if name not in saved_args:
                return
            saved_value = saved_args[name]
            current_value = getattr(args, name, None)
            if current_value != saved_value:
                print(f"WARNING: resume checkpoint stored {name}={saved_value}, overriding current value {current_value}.")
            setattr(args, name, saved_value)

        for field in ("finetune", "finetune_layers", "backbone_lr", "backbone_weight_decay",
                      "distill_teacher", "distill_weight", "distill_temp_start", "distill_temp_end"):
            restore_arg(field)

        timestamp = resume_checkpoint.get("timestamp") or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        exp_dir = resume_checkpoint.get("exp_dir") or os.path.dirname(os.path.abspath(args.resume))
        log_dir = resume_checkpoint.get("log_dir") or os.path.join(args.log_dir, args.exp_name, timestamp)
        os.makedirs(exp_dir, exist_ok=True)
        start_epoch = resume_checkpoint.get("epoch", -1) + 1
        train_global_step = resume_checkpoint.get("train_global_step", 0)
        best_inout_ap = resume_checkpoint.get("best_inout_ap", 0.0)
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        exp_dir = os.path.join(args.ckpt_save_dir, args.exp_name, timestamp)
        os.makedirs(exp_dir, exist_ok=True)
        log_dir = os.path.join(args.log_dir, args.exp_name, timestamp)
        start_epoch = 0
        train_global_step = 0
        best_inout_ap = 0.0

    writer_kwargs = {"log_dir": log_dir}
    if resume_checkpoint is not None:
        writer_kwargs["purge_step"] = train_global_step
    writer = SummaryWriter(**writer_kwargs)

    model, transform = get_gazelle_model(args.model, finetune_backbone=args.finetune)
    if resume_checkpoint is None:
        print("Initializing from {}".format(args.init_ckpt))
        model.load_gazelle_state_dict(torch.load(args.init_ckpt, weights_only=True)) # initializing from ckpt without inout head
    else:
        print(f"Resuming training from {args.resume} at epoch {start_epoch}")
    model.cuda()

    teacher_model = None
    distill_loss_fn = nn.KLDivLoss(reduction='batchmean')
    distill_enabled = args.distill_weight is not None and args.distill_weight > 0
    if distill_enabled:
        if not args.distill_teacher:
            raise ValueError("distill_weight > 0 but no distill_teacher specified.")
        if args.distill_teacher == args.model:
            print("WARNING: distill_teacher matches student model; distillation likely ineffective but continuing.")
        teacher_model, _ = get_gazelle_model(
            args.distill_teacher,
            finetune_backbone=False,
        )
        teacher_model.cuda()
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad_(False)
        print(f"Knowledge distillation enabled: teacher={args.distill_teacher}, weight={args.distill_weight}, "
              f"temp_start={args.distill_temp_start}, temp_end={args.distill_temp_end} (cosine schedule)")
    else:
        print("Knowledge distillation disabled.")

    if args.finetune:
        backbone_trainable_params = configure_backbone_finetune(model.backbone, args.finetune_layers)
    else:
        for param in model.backbone.parameters():
            param.requires_grad = False
        backbone_trainable_params = []

    inout_params = [param for name, param in model.named_parameters() if "inout" in name and param.requires_grad]
    other_head_params = [param for name, param in model.named_parameters() if "inout" not in name and not name.startswith("backbone") and param.requires_grad]

    param_groups = []
    if inout_params:
        param_groups.append({'params': inout_params, 'lr': args.lr_inout})
    if other_head_params:
        param_groups.append({'params': other_head_params, 'lr': args.lr_non_inout})
    if args.finetune and backbone_trainable_params:
        param_groups.append({
            'params': backbone_trainable_params,
            'lr': args.backbone_lr,
            'weight_decay': args.backbone_weight_decay,
        })
    if not param_groups:
        raise RuntimeError("No trainable parameter groups configured for optimizer.")
    optimizer = torch.optim.Adam(param_groups)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    backbone_learnable = sum(p.numel() for p in backbone_trainable_params)
    head_learnable = trainable_params - backbone_learnable
    frozen_params = total_params - trainable_params

    rows = [
        ("Total params", total_params),
        ("Trainable params", trainable_params),
        ("Backbone trainable", backbone_learnable),
        ("Head trainable", head_learnable),
        ("Frozen params", frozen_params),
    ]
    print_param_summary(rows)
    print(f"Learnable parameters: {trainable_params} (backbone finetune: {backbone_learnable})")

    heatmap_loss_fn = nn.BCELoss()
    inout_loss_fn = nn.BCELoss()

    if resume_checkpoint is not None:
        model.load_gazelle_state_dict(resume_checkpoint["model"])
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        _move_optimizer_state_to_device(optimizer, next(model.parameters()).device)
        _restore_rng_state(resume_checkpoint.get("rng_state"))
        if resume_checkpoint.get("max_epochs") and resume_checkpoint["max_epochs"] != args.max_epochs:
            print(f"WARNING: resume checkpoint was created with max_epochs={resume_checkpoint['max_epochs']}, "
                  f"but current run uses max_epochs={args.max_epochs}.")

    train_dataset = GazeDataset('videoattentiontarget', args.data_path, 'train', transform, in_frame_only=False, sample_rate=args.frame_sample_every)
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.n_workers)
    # Note this eval dataloader samples frames sparsely for efficiency - for final results, run eval_vat.py which uses sample rate 1
    eval_dataset = GazeDataset('videoattentiontarget', args.data_path, 'test', transform, in_frame_only=False, sample_rate=args.frame_sample_every)
    eval_dl = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.n_workers)
    total_train_steps = max(1, len(train_dl) * args.max_epochs) if distill_enabled else 1

    if start_epoch >= args.max_epochs:
        print(f"Checkpoint epoch {start_epoch} is >= max_epochs {args.max_epochs}. Nothing to train.")
        writer.close()
        return

    for epoch in range(start_epoch, args.max_epochs):
        # TRAIN EPOCH
        model.train()
        for cur_iter, batch in enumerate(train_dl):
            imgs, bboxes, gazex, gazey, inout, heights, widths, heatmaps = batch

            optimizer.zero_grad()
            imgs_cuda = imgs.cuda()
            bbox_inputs = [[bbox] for bbox in bboxes]
            preds = model({"images": imgs_cuda, "bboxes": bbox_inputs})
            heatmap_preds = torch.stack(preds['heatmap']).squeeze(dim=1)
            inout_preds = torch.stack(preds['inout']).squeeze(dim=1)

            # compute heatmap loss only for in-frame gaze targets
            heatmap_loss = heatmap_loss_fn(heatmap_preds[inout.bool()], heatmaps[inout.bool()].cuda())
            inout_loss = inout_loss_fn(inout_preds, inout.float().cuda())
            loss = heatmap_loss + args.inout_loss_lambda * inout_loss

            if distill_enabled:
                temperature = None
                distill_loss = None
                current_step = epoch * len(train_dl) + cur_iter
                temperature = _cosine_anneal(
                    args.distill_temp_start,
                    args.distill_temp_end,
                    current_step,
                    total_train_steps - 1,
                )
                temperature = max(1e-6, temperature)
                with torch.no_grad():
                    teacher_preds = teacher_model({"images": imgs_cuda, "bboxes": bbox_inputs})
                    teacher_heatmaps = torch.stack(teacher_preds['heatmap']).squeeze(dim=1)
                teacher_heatmaps = teacher_heatmaps.float()
                student_logits = torch.logit(heatmap_preds, eps=1e-6)
                teacher_logits = torch.logit(teacher_heatmaps, eps=1e-6)
                student_logits_flat = (student_logits / temperature).view(student_logits.shape[0], -1)
                teacher_logits_flat = (teacher_logits / temperature).view(teacher_logits.shape[0], -1)
                student_log_probs = torch.log_softmax(student_logits_flat, dim=1)
                teacher_probs = torch.softmax(teacher_logits_flat, dim=1)
                distill_loss = distill_loss_fn(student_log_probs, teacher_probs) * (temperature ** 2)
                loss = loss + args.distill_weight * distill_loss
            loss.backward()
            optimizer.step()

            if cur_iter % args.log_iter == 0:
                writer.add_scalar("train/loss", loss.item(), train_global_step)
                writer.add_scalar("train/heatmap_loss", heatmap_loss.item(), train_global_step)
                writer.add_scalar("train/inout_loss", inout_loss.item(), train_global_step)
                if distill_enabled:
                    if distill_loss is not None:
                        writer.add_scalar("train/distill_loss", distill_loss.item(), train_global_step)
                    if temperature is not None:
                        writer.add_scalar("train/distill_temperature", temperature, train_global_step)
                print("TRAIN EPOCH {}, iter {}/{}, loss={}".format(epoch, cur_iter, len(train_dl), round(loss.item(), 4)))
            train_global_step += 1

        # EVAL EPOCH
        print("Running evaluation")
        model.eval()
        l2s = []
        aucs = []
        all_inout_preds = []
        all_inout_gts = []
        for cur_iter, batch in enumerate(eval_dl):
            imgs, bboxes, gazex, gazey, inout, heights, widths = batch

            with torch.no_grad():
                preds = model({"images": imgs.cuda(), "bboxes": [[bbox] for bbox in bboxes]})

            heatmap_preds = torch.stack(preds['heatmap']).squeeze(dim=1)
            inout_preds = torch.stack(preds['inout']).squeeze(dim=1)
            for i in range(heatmap_preds.shape[0]):
                if inout[i] == 1: # in-frame
                    auc = vat_auc(heatmap_preds[i], gazex[i][0], gazey[i][0])
                    l2 = vat_l2(heatmap_preds[i], gazex[i][0], gazey[i][0])
                    aucs.append(auc)
                    l2s.append(l2)
                all_inout_preds.append(inout_preds[i].item())
                all_inout_gts.append(inout[i])

        epoch_l2 = np.mean(l2s)
        epoch_auc = np.mean(aucs)
        epoch_inout_ap = average_precision_score(all_inout_gts, all_inout_preds)

        writer.add_scalar("eval/auc", epoch_auc, epoch)
        writer.add_scalar("eval/l2", epoch_l2, epoch)
        writer.add_scalar("eval/inout_ap", epoch_inout_ap, epoch)
        print(f"EVAL EPOCH {epoch:03d}: AUC={round(epoch_auc, 4)}, L2={round(epoch_l2, 4)}, Inout AP={round(epoch_inout_ap, 4)}")

        is_best = epoch_inout_ap > best_inout_ap
        if is_best:
            best_inout_ap = epoch_inout_ap

        ckpt_path = os.path.join(exp_dir, f'epoch_{epoch:03d}.pt')
        save_checkpoint(
            ckpt_path,
            model,
            optimizer,
            epoch,
            train_global_step,
            best_inout_ap,
            timestamp,
            exp_dir,
            log_dir,
            vars(args),
        )
        print(f"Saved checkpoint to {ckpt_path}")
        prune_epoch_checkpoints(exp_dir, keep=10)

        if is_best:
            best_filename = f"best_{epoch:03d}_{epoch_auc:.4f}_{epoch_l2:.4f}_{best_inout_ap:.4f}.pt"
            best_ckpt_path = os.path.join(exp_dir, best_filename)
            save_checkpoint(
                best_ckpt_path,
                model,
                optimizer,
                epoch,
                train_global_step,
                best_inout_ap,
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

    writer.close()


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    main()
