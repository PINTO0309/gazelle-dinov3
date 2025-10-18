import warnings
warnings.simplefilter('ignore')
import argparse
from datetime import datetime
import numpy as np
import os
import random
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler

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
args = parser.parse_args()


def main():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = os.path.join(args.ckpt_save_dir, args.exp_name, timestamp)
    os.makedirs(exp_dir)
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.exp_name, timestamp))
    scaler = GradScaler('cuda', enabled=args.use_amp)

    model, transform = get_gazelle_model(args.model_name)
    model.cuda()

    for param in model.backbone.parameters(): # freeze backbone
        param.requires_grad = False
    print(f"Learnable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    train_dataset = GazeDataset('gazefollow', args.data_path, 'train', transform)
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.n_workers)
    eval_dataset = GazeDataset('gazefollow', args.data_path, 'test', transform)
    eval_dl = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.n_workers)

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-7)

    best_min_l2 = 1.0
    best_epoch = None
    train_global_step = 0

    for epoch in range(args.max_epochs):
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

        ckpt_path = os.path.join(exp_dir, f'epoch_{epoch:03d}.pt')
        torch.save(model.get_gazelle_state_dict(), ckpt_path)
        print("Saved checkpoint to {}".format(ckpt_path))

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
            ckpt_path = os.path.join(exp_dir, f'best.pt')
            torch.save(model.get_gazelle_state_dict(), ckpt_path)

    print("Completed training. Best Min L2 of {} obtained at epoch {}".format(round(best_min_l2, 4), best_epoch))
    writer.close()

if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    main()
