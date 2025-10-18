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

```
################################### S
### backbone no-finetune
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

┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Category           ┃ Params [M] ┃
┗━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━┛
| Total params       |     8.17   |
| Trainable params   |     2.68   |
| Backbone trainable |     0.00   |
| Head trainable     |     2.68   |
| Frozen params      |     5.49   |
└--------------------┴------------┘

### backbone finetune - GH200
python scripts/train_gazefollow.py \
--data_path data/gazefollow_extended \
--model_name gazelle_dinov3_vit_tiny \
--exp_name gazelle_dinov3_s \
--log_iter 50 \
--max_epochs 60 \
--batch_size 128 \
--lr 1e-3 \
--n_workers 50 \
--use_amp \
--finetune

┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Category           ┃ Params [M] ┃
┗━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━┛
| Total params       |       8.17 |
| Trainable params   |       3.57 |
| Backbone trainable |       0.89 |
| Head trainable     |       2.68 |
| Frozen params      |       4.60 |
└--------------------┴------------┘

################################### M
### backbone no-finetune
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

┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Category           ┃ Params [M] ┃
┗━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━┛
| Total params       |      12.37 |
| Trainable params   |       2.70 |
| Backbone trainable |       0.00 |
| Head trainable     |       2.70 |
| Frozen params      |       9.67 |
└--------------------┴------------┘

### backbone finetune - GH200
uv run python scripts/train_gazefollow.py \
--data_path data/gazefollow_extended \
--model_name gazelle_dinov3_vit_tinyplus \
--exp_name gazelle_dinov3_m \
--log_iter 10 \
--max_epochs 40 \
--batch_size 32 \
--lr 1e-3 \
--n_workers 16 \
--use_amp \
--finetune

┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Category           ┃ Params [M] ┃
┗━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━┛
| Total params       |      12.37 |
| Trainable params   |       4.28 |
| Backbone trainable |       1.58 |
| Head trainable     |       2.70 |
| Frozen params      |       8.09 |
└--------------------┴------------┘

################################### L
### backbone no-finetune
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

┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Category           ┃ Params [M] ┃
┗━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━┛
| Total params       |      24.33 |
| Trainable params   |       2.73 |
| Backbone trainable |       0.00 |
| Head trainable     |       2.73 |
| Frozen params      |      21.60 |
└--------------------┴------------┘

### backbone finetune - GH200
uv run python scripts/train_gazefollow.py \
--data_path data/gazefollow_extended \
--model_name gazelle_dinov3_vits16 \
--exp_name gazelle_dinov3_l \
--log_iter 10 \
--max_epochs 20 \
--batch_size 8 \
--lr 1e-3 \
--n_workers 8 \
--use_amp \
--finetune

┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Category           ┃ Params [M] ┃
┗━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━┛
| Total params       |      24.33 |
| Trainable params   |       6.28 |
| Backbone trainable |       3.55 |
| Head trainable     |       2.73 |
| Frozen params      |      18.05 |
└--------------------┴------------┘

################################### X
### backbone no-finetune
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

┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Category           ┃ Params [M] ┃
┗━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━┛
| Total params       |      31.43 |
| Trainable params   |       2.73 |
| Backbone trainable |       0.00 |
| Head trainable     |       2.73 |
| Frozen params      |      28.70 |
└--------------------┴------------┘

### backbone finetune - GH200
uv run python scripts/train_gazefollow.py \
--data_path data/gazefollow_extended \
--model_name gazelle_dinov3_vits16plus \
--exp_name gazelle_dinov3_l \
--log_iter 10 \
--max_epochs 20 \
--batch_size 8 \
--lr 1e-3 \
--n_workers 8 \
--use_amp \
--finetune

┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Category           ┃ Params [M] ┃
┗━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━┛
| Total params       |      31.43 |
| Trainable params   |       7.46 |
| Backbone trainable |       4.73 |
| Head trainable     |       2.73 |
| Frozen params      |      23.96 |
└--------------------┴------------┘
```

|Value|Note|
|:-|:-|
|AUC|An index that measures how highly the gaze position within the prediction heat map is ranked. It is the area of ​​the ROC curve when the area including the correct coordinates is considered a "positive example" and the rest is considered a "negative example." The closer the value is to 1.0, the higher the evaluation of areas near the correct answer.|
|Min L2|The Euclidean distance (L2 distance) between the highest-scoring point on the prediction heat map and the correct gaze label. A smaller value means that the "most confident location" is closer to the correct answer.|
|Avg L2|When the entire heat map is considered a probability distribution, this is the Euclidean distance between the prediction center of gravity (expected value) and the correct gaze. It indicates how closely the model's probability mass is to the correct answer; the smaller the value, the better.|
|Inout AP|Average Precision represents the performance of the in/out head in classifying whether the gaze is within the frame (in) or outside the frame (out).<br>By looking at all combinations of precision and recall while moving the score threshold between 0 and 1, we measure threshold-independent discrimination ability.<br>The closer the score is to 1.0, the higher the score is for frames where the gaze is within the screen, and the lower the score is for frames where the gaze is outside the screen, indicating a stable distinction.|

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
