import warnings
warnings.simplefilter('ignore')
import argparse
from datetime import datetime
import numpy as np
import os
import random
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gazelle.dataloader import GazeDataset, collate_fn
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


def main():
    resume_checkpoint = torch.load(args.resume, map_location='cpu') if args.resume else None
    if resume_checkpoint is not None and "model" not in resume_checkpoint:
        raise ValueError(f"The checkpoint at {args.resume} does not contain full training state required for resuming.")

    if resume_checkpoint is not None:
        args.use_amp = resume_checkpoint.get("use_amp", args.use_amp)
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

    model, transform = get_gazelle_model(args.model_name)
    model.cuda()

    for param in model.backbone.parameters(): # freeze backbone
        param.requires_grad = False
    print(f"Learnable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-7)

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
        # TRAIN EPOCH
        model.train()
        for cur_iter, batch in enumerate(train_dl):
            imgs, bboxes, gazex, gazey, inout, heights, widths, heatmaps = batch

            optimizer.zero_grad()
            with autocast('cuda', enabled=args.use_amp):
                preds = model({"images": imgs.cuda(), "bboxes": [[bbox] for bbox in bboxes]})
                heatmap_preds = torch.stack(preds['heatmap']).squeeze(dim=1)
            loss_inputs = heatmap_preds.float() if args.use_amp else heatmap_preds
            loss_targets = heatmaps.cuda()
            loss = loss_fn(loss_inputs, loss_targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if cur_iter % args.log_iter == 0:
                writer.add_scalar("train/loss", loss.item(), train_global_step)
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

        if best_epoch == epoch:
            best_ckpt_path = os.path.join(exp_dir, 'best.pt')
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

    print(f"Completed training. Best Min L2 of {round(best_min_l2, 4)} obtained at epoch {best_epoch:03d}")
    writer.close()

if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    main()
