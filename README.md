# gazelle-dinov3

```bash
git clone https://github.com/PINTO0309/gazelle-dinov3.git && cd gazelle-dinov3
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
source .venv/bin/activate
```
```bash
uv run python data_prep/preprocess_gazefollow.py \
--data_path ./data/gazefollow_extended
```

Dwonloads Distill-DINOv3 pretrain pt to `ckpts`.
- https://github.com/PINTO0309/DEIMv2/releases/download/weights/vitt_distill.pt
- https://github.com/PINTO0309/DEIMv2/releases/download/weights/vittplus_distill.pt

Dwonloads DINOv3 pretrain pth: From https://github.com/facebookresearch/dinov3 to `ckpts`.
- `dinov3_vits16_pretrain_lvd1689m-08c60483.pth`
- `dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth`

```bash
uv run python scripts/train_gazefollow.py \
--data_path data/gazefollow_extended \
--model_name gazelle_dinov3_vit_tiny \
--exp_name gazelle_dinov3_s \
--log_iter 10 \
--max_epochs 15 \
--batch_size 64 \
--lr 1e-3 \
--n_workers 16 \
--use_amp
```
