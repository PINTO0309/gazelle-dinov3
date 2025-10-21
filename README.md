# gazelle-dinov3

> [!warning]
> **October 19, 2025 :** I am continuing to experiment to achieve better accuracy, so this repository is still a work in progress.

A model for activating human gaze regions using heat maps. Built with DINOv3

As I have mastered hell annotation for person detection, I can say with confidence that the minimum required resolution to properly classify the "eyes" and "ears" of the human body, which are important for estimating the direction of the head and gaze, is VGA or higher.

I manually annotated numerous human body parts, including those as small as 3 pixels or less, and benchmarked the performance with the DINOv3-based object detection model DEIMv2. The results clearly show that input resolutions below VGA lack sufficient context. The total number of body parts I have carefully hand-labeled is `1,034,735`.

https://github.com/PINTO0309/PINTO_model_zoo/tree/main/472_DEIMv2-Wholebody34

<img width="600" alt="image" src="https://github.com/user-attachments/assets/00fbfa14-b0b7-442d-8ccb-9152a7a8245e" />


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
  - Download and unzip the above dataset into the `data` folder.

- Preprocessing
  ```bash
  uv run python data_prep/preprocess_gazefollow.py \
  --data_path ./data/gazefollow_extended
  ```

## Download pre-trained backbones

Dwonloads Distill-DINOv3 pretrain pt to `ckpts`. The weights were borrowed from [Intellindust-AI-Lab/DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2).
- https://github.com/PINTO0309/DEIMv2/releases/download/weights/vitt_distill.pt
- https://github.com/PINTO0309/DEIMv2/releases/download/weights/vittplus_distill.pt

Dwonloads DINOv3 pretrain pth: From https://github.com/facebookresearch/dinov3 to `ckpts`.
- `dinov3_vits16_pretrain_lvd1689m-08c60483.pth`
- `dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth`
- `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`

## Training

<details><summary>Training Scripts</summary>

```
############################################# S
### backbone finetune - GH200
uv run python scripts/train_gazefollow.py \
--data_path data/gazefollow_extended \
--model_name gazelle_dinov3_vit_tiny \
--exp_name gazelle_dinov3_s_ft_bcelogits_prog \
--log_iter 50 \
--max_epochs 45 \
--batch_size 64 \
--lr 1e-3 \
--n_workers 50 \
--use_amp \
--finetune \
--finetune_layers 2 \
--grad_clip_norm 1.0 \
--disable_sigmoid \
--initial_freeze_epochs 10 \
--unfreeze_interval 3

### distillation - GH200
uv run python scripts/train_gazefollow.py \
--data_path data/gazefollow_extended \
--model_name gazelle_dinov3_vit_tiny \
--exp_name gazelle_dinov3_s_ft_bcelogits_prog_distill \
--log_iter 50 \
--max_epochs 45 \
--batch_size 64 \
--lr 1e-3 \
--n_workers 50 \
--use_amp \
--finetune \
--finetune_layers 2 \
--grad_clip_norm 1.0 \
--disable_sigmoid \
--initial_freeze_epochs 5 \
--unfreeze_interval 2 \
--distill_teacher gazelle_dinov3_vitb16 \
--distill_weight 0.3 \
--distill_temp_end 4.0

┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Category           ┃ Params [M] ┃
┗━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━┛
| Total params       |       8.17 |
| Trainable params   |       3.57 |
| Backbone trainable |       0.89 |
| Head trainable     |       2.68 |
| Frozen params      |       4.60 |
└--------------------┴------------┘

############################################# M
### backbone finetune - GH200
uv run python scripts/train_gazefollow.py \
--data_path data/gazefollow_extended \
--model_name gazelle_dinov3_vit_tinyplus \
--exp_name gazelle_dinov3_m_ft_bcelogits_prog \
--log_iter 50 \
--max_epochs 45 \
--batch_size 64 \
--lr 1e-3 \
--n_workers 50 \
--use_amp \
--finetune \
--finetune_layers 2 \
--grad_clip_norm 1.0 \
--disable_sigmoid \
--initial_freeze_epochs 10 \
--unfreeze_interval 3

┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Category           ┃ Params [M] ┃
┗━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━┛
| Total params       |      12.37 |
| Trainable params   |       4.28 |
| Backbone trainable |       1.58 |
| Head trainable     |       2.70 |
| Frozen params      |       8.09 |
└--------------------┴------------┘

############################################# L
### backbone finetune - GH200
uv run python scripts/train_gazefollow.py \
--data_path data/gazefollow_extended \
--model_name gazelle_dinov3_vits16 \
--exp_name gazelle_dinov3_l_ft_bcelogits_prog \
--log_iter 50 \
--max_epochs 40 \
--batch_size 64 \
--lr 1e-3 \
--n_workers 50 \
--use_amp \
--finetune \
--finetune_layers 2 \
--grad_clip_norm 1.0 \
--disable_sigmoid \
--initial_freeze_epochs 10 \
--unfreeze_interval 3

┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Category           ┃ Params [M] ┃
┗━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━┛
| Total params       |      24.33 |
| Trainable params   |       6.28 |
| Backbone trainable |       3.55 |
| Head trainable     |       2.73 |
| Frozen params      |      18.05 |
└--------------------┴------------┘

############################################# X
### backbone finetune - GH200
uv run python scripts/train_gazefollow.py \
--data_path data/gazefollow_extended \
--model_name gazelle_dinov3_vits16plus \
--exp_name gazelle_dinov3_x_ft_bcelogits_prog \
--log_iter 50 \
--max_epochs 35 \
--batch_size 64 \
--lr 1e-3 \
--n_workers 50 \
--use_amp \
--finetune \
--finetune_layers 2 \
--grad_clip_norm 1.0 \
--disable_sigmoid \
--initial_freeze_epochs 10 \
--unfreeze_interval 3

┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Category           ┃ Params [M] ┃
┗━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━┛
| Total params       |      31.43 |
| Trainable params   |       7.46 |
| Backbone trainable |       4.73 |
| Head trainable     |       2.73 |
| Frozen params      |      23.96 |
└--------------------┴------------┘

############################################# XL
### backbone finetune - GH200
uv run python scripts/train_gazefollow.py \
--data_path data/gazefollow_extended \
--model_name gazelle_dinov3_vitb16 \
--exp_name gazelle_dinov3_xl_ft_bcelogits_prog \
--log_iter 50 \
--max_epochs 20 \
--batch_size 64 \
--lr 1e-3 \
--n_workers 50 \
--use_amp \
--finetune \
--finetune_layers 2 \
--grad_clip_norm 1.0 \
--disable_sigmoid \
--initial_freeze_epochs 5 \
--unfreeze_interval 2

┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Category           ┃ Params [M] ┃
┗━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━┛
| Total params       |      88.50 |
| Trainable params   |      31.19 |
| Backbone trainable |      28.36 |
| Head trainable     |       2.83 |
| Frozen params      |      57.31 |
└--------------------┴------------┘
```

</details>

|Value|Note|
|:-|:-|
|AUC|An index that measures how highly the gaze position within the prediction heat map is ranked. It is the area of ​​the ROC curve when the area including the correct coordinates is considered a "positive example" and the rest is considered a "negative example." The closer the value is to 1.0, the higher the evaluation of areas near the correct answer.|
|Min L2|The Euclidean distance (L2 distance) between the highest-scoring point on the prediction heat map and the correct gaze label. A smaller value means that the "most confident location" is closer to the correct answer.|
|Avg L2|When the entire heat map is considered a probability distribution, this is the Euclidean distance between the prediction center of gravity (expected value) and the correct gaze. It indicates how closely the model's probability mass is to the correct answer; the smaller the value, the better.|
|Inout AP|Average Precision represents the performance of the in/out head in classifying whether the gaze is within the frame (in) or outside the frame (out).<br>By looking at all combinations of precision and recall while moving the score threshold between 0 and 1, we measure threshold-independent discrimination ability.<br>The closer the score is to 1.0, the higher the score is for frames where the gaze is within the screen, and the lower the score is for frames where the gaze is outside the screen, indicating a stable distinction.|

## Benchmark results for each model
High accuracy is not important to me at all. I'm only interested in whether the model has a realistic computational cost. `S`, `M`, `L`, `X`, `XL` are the performance of the models generated in this repository.

- GazeFollow

  |Variant|Param<br>Backbone+Head|AUC ⬆️|Min L2 ⬇️|Avg L2 ⬇️|
  |:-:|:-:|-:|-:|-:|
  |[Gaze-LLE (ViT-B)](https://arxiv.org/pdf/2412.09586)|88.8 M|0.956|0.045|0.104|
  |[Gaze-LLE (ViT-L)](https://arxiv.org/pdf/2412.09586)|302.9 M|0.958|0.041|0.099|
  |S (No distillation)|8.17 M|0.9477|0.0598|0.1221|
  |M (No distillation)|12.37 M||||
  |L (No distillation)|24.33 M||||
  |X (No distillation)|31.43 M||||
  |XL (No distillation)|88.50 M|0.9593|0.0405|0.0973|

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
- https://github.com/ejcgt/attention-target-detection
  ```bibtex
  @inproceedings{Chong_2020_CVPR,
    title={Detecting Attended Visual Targets in Video},
    author={Chong, Eunji and Wang, Yongxin and Ruiz, Nataniel and Rehg, James M.},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2020}
  }
  ```
- https://github.com/PINTO0309/PINTO_model_zoo/tree/main/462_Gaze-LLE
