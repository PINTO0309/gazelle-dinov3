# gazelle-dinov3

## Installation
```bash
git clone https://github.com/PINTO0309/gazelle-dinov3.git && cd gazelle-dinov3
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
source .venv/bin/activate
```
## Data Preprocessing

- Downloads
  - GazeFollow dataset: https://github.com/ejcgt/attention-target-detection?tab=readme-ov-file#dataset
  - VideoAttentionTarget dataset: https://github.com/ejcgt/attention-target-detection?tab=readme-ov-file#dataset-1

- Preprocessing
  ```bash
  uv run python data_prep/preprocess_gazefollow.py \
  --data_path ./data/gazefollow_extended
  ```

## Download pre-trained backbones

Dwonloads Distill-DINOv3 pretrain pt to `ckpts`.
- https://github.com/PINTO0309/DEIMv2/releases/download/weights/vitt_distill.pt
- https://github.com/PINTO0309/DEIMv2/releases/download/weights/vittplus_distill.pt

Dwonloads DINOv3 pretrain pth: From https://github.com/facebookresearch/dinov3 to `ckpts`.
- `dinov3_vits16_pretrain_lvd1689m-08c60483.pth`
- `dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth`

## Training

```bash
uv run python scripts/train_gazefollow.py \
--data_path data/gazefollow_extended \
--model_name gazelle_dinov3_vit_tiny \
--exp_name gazelle_dinov3_s \
--log_iter 10 \
--max_epochs 60 \
--batch_size 64 \
--lr 1e-3 \
--n_workers 16 \
--use_amp

uv run python scripts/train_gazefollow.py \
--data_path data/gazefollow_extended \
--model_name gazelle_dinov3_vit_tinyplus \
--exp_name gazelle_dinov3_m \
--log_iter 10 \
--max_epochs 40 \
--batch_size 32 \
--lr 1e-3 \
--n_workers 16 \
--use_amp

uv run python scripts/train_gazefollow.py \
--data_path data/gazefollow_extended \
--model_name gazelle_dinov3_vits16 \
--exp_name gazelle_dinov3_l \
--log_iter 10 \
--max_epochs 20 \
--batch_size 8 \
--lr 1e-3 \
--n_workers 8 \
--use_amp

uv run python scripts/train_gazefollow.py \
--data_path data/gazefollow_extended \
--model_name gazelle_dinov3_vits16plus \
--exp_name gazelle_dinov3_x \
--log_iter 10 \
--max_epochs 20 \
--batch_size 8 \
--lr 1e-3 \
--n_workers 8 \
--use_amp
```

## Acknowledgments
- https://github.com/fkryan/gazelle
  ```bibtex
  @inproceedings{ryan2025gazelle,
      author = {Ryan, Fiona and Bati, Ajay and Lee, Sangmin and Bolya, Daniel and Hoffman, Judy and Rehg, James M.},
      title = {Gaze-LLE: Gaze Target Estimation via Large-Scale Learned Encoders},
      year = {2025},
      booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition}
  }
  ```
- https://github.com/Intellindust-AI-Lab/DEIMv2
  ```bibtex
  @article{huang2025deimv2,
    title={Real-Time Object Detection Meets DINOv3},
    author={Huang, Shihua and Hou, Yongjie and Liu, Longfei and Yu, Xuanlong and Shen, Xi},
    journal={arXiv},
    year={2025}
  }
  ```
- https://github.com/facebookresearch/dinov3
  ```bibtex
  @misc{simeoni2025dinov3,
    title={{DINOv3}},
    author={Sim{\'e}oni, Oriane and Vo, Huy V. and Seitzer, Maximilian and Baldassarre, Federico and Oquab, Maxime and Jose, Cijo and Khalidov, Vasil and Szafraniec, Marc and Yi, Seungeun and Ramamonjisoa, Micha{\"e}l and Massa, Francisco and Haziza, Daniel and Wehrstedt, Luca and Wang, Jianyuan and Darcet, Timoth{\'e}e and Moutakanni, Th{\'e}o and Sentana, Leonel and Roberts, Claire and Vedaldi, Andrea and Tolan, Jamie and Brandt, John and Couprie, Camille and Mairal, Julien and J{\'e}gou, Herv{\'e} and Labatut, Patrick and Bojanowski, Piotr},
    year={2025},
    eprint={2508.10104},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2508.10104},
  }
  ```
