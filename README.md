# gazelle-dinov3

```bash
git clone --recurse-submodules https://github.com/PINTO0309/gazelle-dinov3.git
cd gazelle-dinov3
git submodule update --init --recursive
```
```bash
uv run python data_prep/preprocess_gazefollow.py \
--data_path ./data/gazefollow_extended
```
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
