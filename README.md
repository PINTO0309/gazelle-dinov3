# gazelle-dinov3

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17413165.svg)](https://doi.org/10.5281/zenodo.17413165) ![GitHub License](https://img.shields.io/github/license/pinto0309/gazelle-dinov3) [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/PINTO0309/gazelle-dinov3)


> [!Note]
> **October 26, 2025 :** A checkpoint file `Atto`, `Femto`, `Pico`, `N` containing `GazeFollow`, `VideoAttentionTarget` trained weights and statistical information has been released.
>
> **October 23, 2025 :** A checkpoint file `.pt` containing `VideoAttentionTarget`'s trained weights and statistical information has been released.
>
> **October 22, 2025 :** A checkpoint file `.pt` containing `GazeFollow`'s trained weights and statistical information has been released.
>
> **October 19, 2025 :** I am continuing to experiment to achieve better accuracy, so this repository is still a work in progress.

A model for activating human gaze regions using heat maps. Built with DINOv3.

Real-time demo with RTX3070. The inference speed displayed in the upper left corner of the screen is the total processing time for object detection, gaze area estimation, and all pre-processing and post-processing.

https://github.com/user-attachments/assets/bfc3f569-31c4-4bd0-b539-0fade9782c8f

As I have mastered hell annotation for person detection, I can say with confidence that the minimum required resolution to properly classify the "eyes" and "ears" of the human body, which are important for estimating the direction of the head and gaze, is VGA or higher.

I manually annotated numerous human body parts, including those as small as 3 pixels or less, and benchmarked the performance with the DINOv3-based object detection model DEIMv2. The results clearly show that input resolutions below VGA lack sufficient context. The total number of body parts I have carefully hand-labeled is `1,034,735`.

https://github.com/PINTO0309/PINTO_model_zoo/tree/main/472_DEIMv2-Wholebody34

<img width="600" alt="image" src="https://github.com/user-attachments/assets/00fbfa14-b0b7-442d-8ccb-9152a7a8245e" />

Therefore, the resolutions of `224x224` and `448x448` proposed in previous papers are far too small to obtain sufficient context from the vision. When designing this model, I changed the input resolution to `640x640`. Also, unlike previous papers, I use a pipeline that [progressive_unfreezing](./README_progressive_unfreezing.md) the backbone and fine-tunes all layers. This is important in terms of preventing catastrophic forgetting without self-distillation, preventing gradient collapse, and fine-tuning optimal weights for the task. Note that because `GazeFollow` and `VideoAttentionTarget` have too large class imbalances, I use `BCEWithLogitsLoss` instead of the loss function defined by `BCELoss + Sigmoid`.

The full learning pipeline can be seen [here](./README_architecture.md).

I'm not a researcher but a hobbyist programmer, so I don't write papers.

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

  uv run python data_prep/preprocess_vat.py \
  --data_path ./data/videoattentiontarget
  ```

## Download pre-trained backbones

Downloads Distill-DINOv3 pretrain pt to `ckpts`. The weights were borrowed from [Intellindust-AI-Lab/DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2).
- https://github.com/PINTO0309/DEIMv2/releases/download/weights/vitt_distill.pt
- https://github.com/PINTO0309/DEIMv2/releases/download/weights/vittplus_distill.pt

Downloads PPHGNetV2 pretrain pt to `ckpts`. The weights were borrowed from [Peterande/D-FINE](https://github.com/Peterande/storage).
- https://github.com/PINTO0309/DEIMv2/releases/download/weights/PPHGNetV2_B0_stage1.pth

Downloads DINOv3 pretrain pth: From https://github.com/facebookresearch/dinov3 to `ckpts`.
- `dinov3_vits16_pretrain_lvd1689m-08c60483.pth`
- `dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth`
- `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`

## Training

<details><summary>Training Scripts</summary>

### GazeFollow

```
############################################# Atto
### distillation - GH200
uv run python scripts/train_gazefollow.py \
--data_path data/gazefollow_extended \
--model_name gazelle_hgnetv2_atto \
--exp_name gazelle_hgnetv2_atto_distill \
--log_iter 50 \
--max_epochs 60 \
--batch_size 128 \
--lr 1e-3 \
--n_workers 60 \
--use_amp \
--finetune \
--finetune_layers 2 \
--grad_clip_norm 1.0 \
--disable_sigmoid \
--initial_freeze_epochs 10 \
--unfreeze_interval 2 \
--distill_teacher gazelle_dinov3_vits16plus \
--distill_weight 0.3 \
--distill_temp_end 4.0

┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Category           ┃ Params [M] ┃
┗━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━┛
| Total params       |       2.93 |
| Trainable params   |       2.93 |
| Backbone trainable |       0.23 |
| Head trainable     |       2.70 |
| Frozen params      |       0.00 |
└--------------------┴------------┘

############################################# Femto
### distillation - GH200
uv run python scripts/train_gazefollow.py \
--data_path data/gazefollow_extended \
--model_name gazelle_hgnetv2_femto \
--exp_name gazelle_hgnetv2_femto_distill \
--log_iter 50 \
--max_epochs 55 \
--batch_size 128 \
--lr 1e-3 \
--n_workers 60 \
--use_amp \
--finetune \
--finetune_layers 2 \
--grad_clip_norm 1.0 \
--disable_sigmoid \
--initial_freeze_epochs 10 \
--unfreeze_interval 2 \
--distill_teacher gazelle_dinov3_vits16plus \
--distill_weight 0.3 \
--distill_temp_end 4.0

┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Category           ┃ Params [M] ┃
┗━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━┛
| Total params       |       3.15 |
| Trainable params   |       3.15 |
| Backbone trainable |       0.39 |
| Head trainable     |       2.76 |
| Frozen params      |       0.00 |
└--------------------┴------------┘

############################################# Pico
### distillation - GH200
uv run python scripts/train_gazefollow.py \
--data_path data/gazefollow_extended \
--model_name gazelle_hgnetv2_pico \
--exp_name gazelle_hgnetv2_pico_distill \
--log_iter 50 \
--max_epochs 50 \
--batch_size 128 \
--lr 1e-3 \
--n_workers 60 \
--use_amp \
--finetune \
--finetune_layers 2 \
--grad_clip_norm 1.0 \
--disable_sigmoid \
--initial_freeze_epochs 10 \
--unfreeze_interval 2 \
--distill_teacher gazelle_dinov3_vits16plus \
--distill_weight 0.3 \
--distill_temp_end 4.0

┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Category           ┃ Params [M] ┃
┗━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━┛
| Total params       |       3.51 |
| Trainable params   |       3.51 |
| Backbone trainable |       0.75 |
| Head trainable     |       2.76 |
| Frozen params      |       0.00 |
└--------------------┴------------┘

############################################# N
### distillation - GH200
uv run python scripts/train_gazefollow.py \
--data_path data/gazefollow_extended \
--model_name gazelle_hgnetv2_n \
--exp_name gazelle_hgnetv2_n_distill \
--log_iter 50 \
--max_epochs 50 \
--batch_size 128 \
--lr 1e-3 \
--n_workers 60 \
--use_amp \
--finetune \
--finetune_layers 2 \
--grad_clip_norm 1.0 \
--disable_sigmoid \
--initial_freeze_epochs 10 \
--unfreeze_interval 2 \
--distill_teacher gazelle_dinov3_vits16plus \
--distill_weight 0.3 \
--distill_temp_end 4.0

┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Category           ┃ Params [M] ┃
┗━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━┛
| Total params       |       4.61 |
| Trainable params   |       4.61 |
| Backbone trainable |       1.85 |
| Head trainable     |       2.76 |
| Frozen params      |       0.00 |
└--------------------┴------------┘

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
--max_epochs 40 \
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

### distillation - GH200
uv run python scripts/train_gazefollow.py \
--data_path data/gazefollow_extended \
--model_name gazelle_dinov3_vit_tinyplus \
--exp_name gazelle_dinov3_m_ft_bcelogits_prog_distill \
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
--initial_freeze_epochs 5 \
--unfreeze_interval 2 \
--distill_teacher gazelle_dinov3_vitb16 \
--distill_weight 0.3 \
--distill_temp_end 4.0

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

### distillation - GH200
uv run python scripts/train_gazefollow.py \
--data_path data/gazefollow_extended \
--model_name gazelle_dinov3_vits16 \
--exp_name gazelle_dinov3_l_ft_bcelogits_prog_distill \
--log_iter 50 \
--max_epochs 30 \
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

### distillation - GH200
uv run python scripts/train_gazefollow.py \
--data_path data/gazefollow_extended \
--model_name gazelle_dinov3_vits16plus \
--exp_name gazelle_dinov3_x_ft_bcelogits_prog_distill \
--log_iter 50 \
--max_epochs 30 \
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

### VideoAttentionTarget
- `--init_ckpt` loads the student model’s starting weights before VAT training.
This is always executed even if distillation is not used,
and the backbone + head (In/Out heads can be untrained) pre-trained with
GazeFollow is transplanted into the student model to improve initial performance.

- `--distill_teacher` is an argument that specifies the architecture name of
the teacher model to use during distillation. This is only read when distill_weight
is set to a positive value, and a teacher network is constructed with a separate
get_gazelle_model call.

```
############################################# Atto
### distillation - GH200
uv run python scripts/train_vat.py \
--data_path data/videoattentiontarget \
--model_name gazelle_hgnetv2_atto_inout \
--exp_name gazelle_hgnetv2_atto_inout_distill \
--init_ckpt ckpts/gazelle_hgnetv2_atto_distill.pt \
--frame_sample_every 6 \
--log_iter 50 \
--max_epochs 65 \
--batch_size 128 \
--n_workers 60 \
--lr_non_inout 1e-5 \
--lr_inout 1e-2 \
--inout_loss_lambda 1.0 \
--use_amp \
--grad_clip_norm 1.0 \
--disable_sigmoid \
--disable_progressive_unfreeze \
--distill_teacher gazelle_dinov3_vitb16_inout \
--distill_weight 0.3 \
--distill_temp_end 4.0

############################################# Femto
### distillation - GH200
uv run python scripts/train_vat.py \
--data_path data/videoattentiontarget \
--model_name gazelle_hgnetv2_femto_inout \
--exp_name gazelle_hgnetv2_femto_inout_distill \
--init_ckpt ckpts/gazelle_hgnetv2_femto_distill.pt \
--frame_sample_every 6 \
--log_iter 50 \
--max_epochs 60 \
--batch_size 128 \
--n_workers 60 \
--lr_non_inout 1e-5 \
--lr_inout 1e-2 \
--inout_loss_lambda 1.0 \
--use_amp \
--grad_clip_norm 1.0 \
--disable_sigmoid \
--disable_progressive_unfreeze \
--distill_teacher gazelle_dinov3_vitb16_inout \
--distill_weight 0.3 \
--distill_temp_end 4.0

############################################# Pico
### distillation - GH200
uv run python scripts/train_vat.py \
--data_path data/videoattentiontarget \
--model_name gazelle_hgnetv2_pico_inout \
--exp_name gazelle_hgnetv2_pico_inout_distill \
--init_ckpt ckpts/gazelle_hgnetv2_pico_distill.pt \
--frame_sample_every 6 \
--log_iter 50 \
--max_epochs 50 \
--batch_size 128 \
--n_workers 60 \
--lr_non_inout 1e-5 \
--lr_inout 1e-2 \
--inout_loss_lambda 1.0 \
--use_amp \
--grad_clip_norm 1.0 \
--disable_sigmoid \
--disable_progressive_unfreeze \
--distill_teacher gazelle_dinov3_vitb16_inout \
--distill_weight 0.3 \
--distill_temp_end 4.0

############################################# N
### distillation - GH200
uv run python scripts/train_vat.py \
--data_path data/videoattentiontarget \
--model_name gazelle_hgnetv2_n_inout \
--exp_name gazelle_hgnetv2_n_inout_distill \
--init_ckpt ckpts/gazelle_hgnetv2_n_distill.pt \
--frame_sample_every 6 \
--log_iter 50 \
--max_epochs 50 \
--batch_size 128 \
--n_workers 60 \
--lr_non_inout 1e-5 \
--lr_inout 1e-2 \
--inout_loss_lambda 1.0 \
--use_amp \
--grad_clip_norm 1.0 \
--disable_sigmoid \
--disable_progressive_unfreeze \
--distill_teacher gazelle_dinov3_vitb16_inout \
--distill_weight 0.3 \
--distill_temp_end 4.0

############################################# S
### distillation - GH200
uv run python scripts/train_vat.py \
--data_path data/videoattentiontarget \
--model_name gazelle_dinov3_vit_tiny_inout \
--exp_name gazelle_dinov3_s_inout_distill \
--init_ckpt ckpts/gazelle_dinov3_vit_tiny.pt \
--frame_sample_every 6 \
--log_iter 50 \
--max_epochs 40 \
--batch_size 64 \
--n_workers 50 \
--lr_non_inout 1e-5 \
--lr_inout 1e-2 \
--inout_loss_lambda 1.0 \
--use_amp \
--grad_clip_norm 1.0 \
--disable_sigmoid \
--disable_progressive_unfreeze \
--distill_teacher gazelle_dinov3_vitb16_inout \
--distill_weight 0.3 \
--distill_temp_end 4.0

############################################# M
### distillation - GH200
uv run python scripts/train_vat.py \
--data_path data/videoattentiontarget \
--model_name gazelle_dinov3_vit_tinyplus_inout \
--exp_name gazelle_dinov3_m_inout_distill \
--init_ckpt ckpts/gazelle_dinov3_vit_tinyplus.pt \
--frame_sample_every 6 \
--log_iter 50 \
--max_epochs 20 \
--batch_size 64 \
--n_workers 50 \
--lr_non_inout 1e-5 \
--lr_inout 1e-2 \
--inout_loss_lambda 1.0 \
--use_amp \
--grad_clip_norm 1.0 \
--disable_sigmoid \
--disable_progressive_unfreeze \
--distill_teacher gazelle_dinov3_vitb16_inout \
--distill_weight 0.3 \
--distill_temp_end 4.0

############################################# L
### distill GH200
uv run python scripts/train_vat.py \
--data_path data/videoattentiontarget \
--model_name gazelle_dinov3_vits16_inout \
--exp_name gazelle_dinov3_l_inout_distill \
--init_ckpt ckpts/gazelle_dinov3_vits16.pt \
--frame_sample_every 6 \
--log_iter 50 \
--max_epochs 20 \
--batch_size 64 \
--n_workers 50 \
--lr_non_inout 1e-5 \
--lr_inout 1e-2 \
--inout_loss_lambda 1.0 \
--use_amp \
--grad_clip_norm 1.0 \
--disable_sigmoid \
--disable_progressive_unfreeze \
--distill_teacher gazelle_dinov3_vitb16_inout \
--distill_weight 0.3 \
--distill_temp_end 4.0


############################################# X
### no-distill GH200
uv run python scripts/train_vat.py \
--data_path data/videoattentiontarget \
--model_name gazelle_dinov3_vits16plus_inout \
--exp_name gazelle_dinov3_x_inout \
--init_ckpt ckpts/gazelle_dinov3_vits16plus.pt \
--frame_sample_every 6 \
--log_iter 50 \
--max_epochs 20 \
--batch_size 64 \
--n_workers 50 \
--lr_non_inout 1e-5 \
--lr_inout 1e-2 \
--inout_loss_lambda 1.0 \
--use_amp \
--grad_clip_norm 1.0 \
--disable_sigmoid \
--disable_progressive_unfreeze

### distill GH200
uv run python scripts/train_vat.py \
--data_path data/videoattentiontarget \
--model_name gazelle_dinov3_vits16plus_inout \
--exp_name gazelle_dinov3_x_inout_distill \
--init_ckpt ckpts/gazelle_dinov3_vits16plus.pt \
--frame_sample_every 6 \
--log_iter 50 \
--max_epochs 20 \
--batch_size 64 \
--n_workers 50 \
--lr_non_inout 1e-5 \
--lr_inout 1e-2 \
--inout_loss_lambda 1.0 \
--use_amp \
--grad_clip_norm 1.0 \
--disable_sigmoid \
--disable_progressive_unfreeze \
--distill_teacher gazelle_dinov3_vitb16_inout \
--distill_weight 0.3 \
--distill_temp_end 4.0

############################################# XL
### no-distill GH200
uv run python scripts/train_vat.py \
--data_path data/videoattentiontarget \
--model_name gazelle_dinov3_vitb16_inout \
--exp_name gazelle_dinov3_xl_inout \
--init_ckpt ckpts/gazelle_dinov3_vitb16.pt \
--frame_sample_every 6 \
--log_iter 50 \
--max_epochs 15 \
--batch_size 64 \
--n_workers 50 \
--lr_non_inout 1e-5 \
--lr_inout 1e-2 \
--inout_loss_lambda 1.0 \
--use_amp \
--grad_clip_norm 1.0 \
--disable_sigmoid \
--disable_progressive_unfreeze
```

</details>

|Value|Note|
|:-|:-|
|AUC|An index that measures how highly the gaze position within the prediction heat map is ranked. It is the area of ​​the ROC curve when the area including the correct coordinates is considered a "positive example" and the rest is considered a "negative example." The closer the value is to 1.0, the higher the evaluation of areas near the correct answer.|
|Min L2|The Euclidean distance (L2 distance) between the highest-scoring point on the prediction heat map and the correct gaze label. A smaller value means that the "most confident location" is closer to the correct answer.|
|Avg L2|When the entire heat map is considered a probability distribution, this is the Euclidean distance between the prediction center of gravity (expected value) and the correct gaze. It indicates how closely the model's probability mass is to the correct answer; the smaller the value, the better.|
|Inout AP|Average Precision represents the performance of the in/out head in classifying whether the gaze is within the frame (in) or outside the frame (out).<br>By looking at all combinations of precision and recall while moving the score threshold between 0 and 1, we measure threshold-independent discrimination ability.<br>The closer the score is to 1.0, the higher the score is for frames where the gaze is within the screen, and the lower the score is for frames where the gaze is outside the screen, indicating a stable distinction.|

## Benchmark results for each model
High accuracy is not important to me at all. I'm only interested in whether the model has a realistic computational cost. `Atto`, `Femto`, `Pico`, `N`, `S`, `M`, `L`, `X`, `XL` are the performance of the models generated in this repository. Variants below `N` are CNN, and variants above `S` are ViT.

- GazeFollow

  |Variant|Param<br>(Backbone+Head)|AUC ⬆️|Min L2 ⬇️|Avg L2 ⬇️|Weight|ONNX|
  |:-:|:-:|-:|-:|-:|:-:|:-:|
  |[Gaze-LLE (ViT-B)](https://arxiv.org/pdf/2412.09586)|88.80 M|0.9560|0.0450|0.1040|[Download](https://github.com/fkryan/gazelle/releases/download/v1.0.0/gazelle_dinov2_vitb14.pt)<br>Head only|---|
  |[Gaze-LLE (ViT-L)](https://arxiv.org/pdf/2412.09586)|302.90 M|0.9580|0.0410|0.0990|[Download](https://github.com/fkryan/gazelle/releases/download/v1.0.0/gazelle_dinov2_vitl14.pt)<br>Head only|---|
  |Atto-distillation|2.93 M|0.9267|0.0826|0.1482|[Download](https://github.com/PINTO0309/gazelle-dinov3/releases/download/weights/gazelle_hgnetv2_atto_distill.pt)|[Download](https://github.com/PINTO0309/gazelle-dinov3/releases/download/weights/gazelle_hgnetv2_atto_distill_1x3x320x320_1xNx4.onnx)|
  |Femto-distillation|3.15 M|0.9391|0.0656|0.1289|[Download](https://github.com/PINTO0309/gazelle-dinov3/releases/download/weights/gazelle_hgnetv2_femto_distill.pt)|[Download](https://github.com/PINTO0309/gazelle-dinov3/releases/download/weights/gazelle_hgnetv2_femto_distill_1x3x416x416_1xNx4.onnx)|
  |Pico-distillation|3.51 M|0.9491|0.0544|0.1149|[Download](https://github.com/PINTO0309/gazelle-dinov3/releases/download/weights/gazelle_hgnetv2_pico_distill.pt)|[Download](https://github.com/PINTO0309/gazelle-dinov3/releases/download/weights/gazelle_hgnetv2_pico_distill_1x3x640x640_1xNx4.onnx)|
  |N-distillation|4.61 M|0.9481|0.0549|0.1158|[Download](https://github.com/PINTO0309/gazelle-dinov3/releases/download/weights/gazelle_hgnetv2_n_distill.pt)|[Download](https://github.com/PINTO0309/gazelle-dinov3/releases/download/weights/gazelle_hgnetv2_n_distill_1x3x640x640_1xNx4.onnx)|
  |S-distillation|8.17 M|0.9545|0.0484|0.1118|[Download](https://github.com/PINTO0309/gazelle-dinov3/releases/download/weights/gazelle_dinov3_vit_tiny.pt)|[Download](https://github.com/PINTO0309/gazelle-dinov3/releases/download/weights/gazelle_dinov3_vit_tiny_1x3x640x640_1xNx4.onnx)|
  |M-distillation|12.37 M|0.9564|0.0462|0.1042|[Download](https://github.com/PINTO0309/gazelle-dinov3/releases/download/weights/gazelle_dinov3_vit_tinyplus.pt)|[Download](https://github.com/PINTO0309/gazelle-dinov3/releases/download/weights/gazelle_dinov3_vit_tinyplus_1x3x640x640_1xNx4.onnx)|
  |L-distillation|24.33 M|0.9593|0.0418|0.0992|[Download](https://github.com/PINTO0309/gazelle-dinov3/releases/download/weights/gazelle_dinov3_vits16.pt)|[Download](https://github.com/PINTO0309/gazelle-dinov3/releases/download/weights/gazelle_dinov3_vits16_1x3x640x640_1xNx4.onnx)|
  |X-distillation|**31.43 M**|**0.9604**|**0.0395**|**0.0966**|[Download](https://github.com/PINTO0309/gazelle-dinov3/releases/download/weights/gazelle_dinov3_vits16plus.pt)|[Download](https://github.com/PINTO0309/gazelle-dinov3/releases/download/weights/gazelle_dinov3_vits16plus_1x3x640x640_1xNx4.onnx)|
  |XL (Teacher)|88.50 M|0.9593|0.0405|0.0973|[Download](https://github.com/PINTO0309/gazelle-dinov3/releases/download/weights/gazelle_dinov3_vitb16.pt)|[Download](https://github.com/PINTO0309/gazelle-dinov3/releases/download/weights/gazelle_dinov3_vitb16_1x3x640x640_1xNx4.onnx)|

  RTX3070 + TensorRT inference speed benchmark. Average of 1000 inferences. Below is the inference speed of the entire model integrating backbone and head.

  <img width="700" alt="benchmark_times_combined" src="https://github.com/user-attachments/assets/741d0ae4-ba21-4e59-a755-8ae8d97124dc" />

  |S|M|
  |:-:|:-:|
  |<img width="1280" height="800" alt="benchmark_times_gazelle_dinov3_vit_tiny_1x3x640x640_1xNx4" src="https://github.com/user-attachments/assets/ec1fcbf9-70d8-4b6c-aa8a-d2efbd6079ae" />|<img width="1280" height="800" alt="benchmark_times_gazelle_dinov3_vit_tinyplus_1x3x640x640_1xNx4" src="https://github.com/user-attachments/assets/14ba0f97-247a-4d48-90cf-cff91c1b9b20" />|

  |L|X|
  |:-:|:-:|
  |<img width="1280" height="800" alt="benchmark_times_gazelle_dinov3_vits16_1x3x640x640_1xNx4" src="https://github.com/user-attachments/assets/c51e3c81-65ba-4216-8907-087d505eeaea" />|<img width="1280" height="800" alt="benchmark_times_gazelle_dinov3_vits16plus_1x3x640x640_1xNx4" src="https://github.com/user-attachments/assets/e59b053f-10e8-4b59-abe7-76b8858fc14f" />|

  <img width="700" alt="benchmark_times_combined_2" src="https://github.com/user-attachments/assets/cb876564-f776-43c4-9547-6c2de220c2e1" />

  |N|Pico|
  |:-:|:-:|
  |<img width="1280" height="800" alt="benchmark_times_gazelle_hgnetv2_n_distill_1x3x640x640_1xNx4" src="https://github.com/user-attachments/assets/cbef40a6-937f-4213-89b4-6403d9dd4b27" />|<img width="1280" height="800" alt="benchmark_times_gazelle_hgnetv2_pico_distill_1x3x640x640_1xNx4" src="https://github.com/user-attachments/assets/f5ddf1e5-25b2-4589-9cdb-727a59120620" />|

  |Femto|Atto|
  |:-:|:-:|
  |<img width="1280" height="800" alt="benchmark_times_gazelle_hgnetv2_femto_distill_1x3x416x416_1xNx4" src="https://github.com/user-attachments/assets/233239dc-c35f-4285-bfed-f02a51fe511c" />|<img width="1280" height="800" alt="benchmark_times_gazelle_hgnetv2_atto_distill_1x3x320x320_1xNx4" src="https://github.com/user-attachments/assets/137a961b-6027-4ddc-88c8-25f8b74c55fa" />|

- VideoAttentionTarget

  |Variant|Param<br>(Backbone+Head)|AUC ⬆️|Avg L2 ⬇️|AP IN/OUT ⬆️|Weight|ONNX|
  |:-:|:-:|-:|-:|-:|:-:|:-:|
  |[Gaze-LLE (ViT-B)](https://arxiv.org/pdf/2412.09586)|88.80 M|0.9330|0.1070|0.8970|[Download](https://github.com/fkryan/gazelle/releases/download/v1.0.0/gazelle_dinov2_vitb14_inout.pt)<br>Head only|---|
  |[Gaze-LLE (ViT-L)](https://arxiv.org/pdf/2412.09586)|302.90 M|0.9370|0.1030|0.9030|[Download](https://github.com/fkryan/gazelle/releases/download/v1.0.0/gazelle_dinov2_vitl14_inout.pt)<br>Head only|---|
  |Atto-distillation|2.93 M|0.9055|0.1523|0.8749|[Download](https://github.com/PINTO0309/gazelle-dinov3/releases/download/weights/gazelle_hgnetv2_atto_inout_distill.pt)|[Download](https://github.com/PINTO0309/gazelle-dinov3/releases/download/weights/gazelle_hgnetv2_atto_inout_distill_1x3x320x320_1xNx4.onnx)|
  |Femto-distillation|3.15 M|0.9166|0.1372|0.8779|[Download](https://github.com/PINTO0309/gazelle-dinov3/releases/download/weights/gazelle_hgnetv2_femto_inout_distill.pt)|[Download](https://github.com/PINTO0309/gazelle-dinov3/releases/download/weights/gazelle_hgnetv2_femto_inout_distill_1x3x416x416_1xNx4.onnx)|
  |Pico-distillation|3.51 M|0.9247|0.1245|0.8861|[Download](https://github.com/PINTO0309/gazelle-dinov3/releases/download/weights/gazelle_hgnetv2_pico_inout_distill.pt)|[Download](https://github.com/PINTO0309/gazelle-dinov3/releases/download/weights/gazelle_hgnetv2_pico_inout_distill_1x3x640x640_1xNx4.onnx)|
  |N-distillation|4.61 M|0.9218|0.1258|0.9012|[Download](https://github.com/PINTO0309/gazelle-dinov3/releases/download/weights/gazelle_hgnetv2_n_inout_distill.pt)|[Download](https://github.com/PINTO0309/gazelle-dinov3/releases/download/weights/gazelle_hgnetv2_n_inout_distill_1x3x640x640_1xNx4.onnx)|
  |S-distillation|8.17 M|0.9286|0.1155|0.8945|[Download](https://github.com/PINTO0309/gazelle-dinov3/releases/download/weights/gazelle_dinov3_vit_tiny_inout.pt)|[Download](https://github.com/PINTO0309/gazelle-dinov3/releases/download/weights/gazelle_dinov3_vit_tiny_inout_1x3x640x640_1xNx4.onnx)|
  |M-distillation|12.37 M|0.9325|0.1133|0.8953|[Download](https://github.com/PINTO0309/gazelle-dinov3/releases/download/weights/gazelle_dinov3_vit_tinyplus_inout.pt)|[Download](https://github.com/PINTO0309/gazelle-dinov3/releases/download/weights/gazelle_dinov3_vit_tinyplus_inout_1x3x640x640_1xNx4.onnx)|
  |L-distillation|24.33 M|0.9347|0.1026|0.9011|[Download](https://github.com/PINTO0309/gazelle-dinov3/releases/download/weights/gazelle_dinov3_vits16_inout.pt)|[Download](https://github.com/PINTO0309/gazelle-dinov3/releases/download/weights/gazelle_dinov3_vits16_inout_1x3x640x640_1xNx4.onnx)|
  |X-distillation|31.43 M|0.9366|0.1050|**0.9118**|[Download](https://github.com/PINTO0309/gazelle-dinov3/releases/download/weights/gazelle_dinov3_vits16plus_inout.pt)|[Download](https://github.com/PINTO0309/gazelle-dinov3/releases/download/weights/gazelle_dinov3_vits16plus_inout_1x3x640x640_1xNx4.onnx)|
  |XL (Teacher)|**88.50 M**|**0.9399**|**0.0943**|0.9051|[Download](https://github.com/PINTO0309/gazelle-dinov3/releases/download/weights/gazelle_dinov3_vitb16_inout.pt)|[Download](https://github.com/PINTO0309/gazelle-dinov3/releases/download/weights/gazelle_dinov3_vitb16_inout_1x3x640x640_1xNx4.onnx)|

## Validation
- GazeFollow
  ```bash
  uv run python scripts/eval_gazefollow.py \
  --data_path data/gazefollow_extended \
  --model_name gazelle_dinov3_vit_tiny \
  --ckpt_path ckpts/gazelle_dinov3_vit_tiny.pt \
  --batch_size 64
  ```
- VideoAttentionTarget
  ```bash
  uv run python scripts/eval_vat.py \
  --data_path data/videoattentiontarget \
  --model_name gazelle_dinov3_vit_tiny_inout \
  --ckpt_path ckpts/gazelle_dinov3_vit_tiny_inout.pt \
  --batch_size 64
  ```

## ONNX
- An ultra-lightweight CNN model dedicated to human detection

  https://github.com/PINTO0309/gazelle-dinov3/releases/download/weights/deimv2_hgnetv2_pico_wholebody34_340query_n_batch_640x640.onnx

  or
  
  https://github.com/PINTO0309/PINTO_model_zoo/tree/main/472_DEIMv2-Wholebody34

- Gaze-LLE-DINOv3 models

  https://github.com/PINTO0309/gazelle-dinov3/releases/tag/weights

- Inputs
  |Name|Note|
  |:-|:-|
  |image_bgr|`float32[1, 3, H, W]`. BGR image (not RGB). 640x640 or 416x416 or 320x320.<br>If you don't like BGR image input, rewrite the code below and re-run `export_onnx.py` yourself. Just comment out the line `torch.cat()`. https://github.com/PINTO0309/gazelle-dinov3/blob/07144c2eee14b26422581561afd04c30b4c2e659/gazelle/model.py#L200-L208|
  |bboxes_x1y1x2y2|`float32[1, heads, 4]`. `heads`=`Number of heads detected by the object detection model`. `4`=`[x1,y1,x2,y2]`. `x1`, `y1`, `x2`, `y2` are coordinates normalized to 0.0-1.0.<br>https://github.com/PINTO0309/gazelle-dinov3/blob/a2d21711f57b46c0c9b9ddb580e3eaf9483b6816/demo_deimv2_onnx_wholebody34_with_edges.py#L894-L906|
- Outputs
  |Name|Note|
  |:-|:-|
  |heatmap|`float32[heads, H, W]`. 64x64 or 48x48 or 32x32. Heatmap of the gaze region relative to the input head. It is scaled to the same size as the input image and then used to render it as a heatmap.<br>https://github.com/PINTO0309/gazelle-dinov3/blob/a2d21711f57b46c0c9b9ddb580e3eaf9483b6816/demo_deimv2_onnx_wholebody34_with_edges.py#L954-L977|
  |inout|`float32[heads]`. This is a score that indicates whether the gaze area associated with the head is looking inside or outside the image. It ranges from 0.0 to 1.0, and the closer it is to 1.0, the higher the probability that the gaze area is looking inside the image.|

## Inference
```
uv run python demo_deimv2_onnx_wholebody34_with_edges.py -h

usage: demo_deimv2_onnx_wholebody34_with_edges.py
[-h] [-m MODEL] [--gazelle_model GAZELLE_MODEL] (-v VIDEO | -i IMAGES_DIR)
[-ep {cpu,cuda,tensorrt}] [-it {fp16,int8}] [-dvw] [-dwk]
[-ost OBJECT_SOCRE_THRESHOLD] [-ast ATTRIBUTE_SOCRE_THRESHOLD]
[-kst KEYPOINT_THRESHOLD] [-kdm {dot,box,both}] [-ebm] [-dnm] [-dgm] [-dlr]
[-dhm] [-drc [DISABLE_RENDER_CLASSIDS ...]] [-efm] [-dtk] [-dti] [-dhd]
[-oyt] [-bblw BOUNDING_BOX_LINE_WIDTH] [-chf CAMERA_HORIZONTAL_FOV]
[--gazelle_heatmap_alpha GAZELLE_HEATMAP_ALPHA] [--disable_gazelle_heatmap]

options:
  -h, --help
    show this help message and exit
  -m MODEL, --model MODEL
    ONNX/TFLite file path for DEIMv2.
  --gazelle_model GAZELLE_MODEL
    Gazelle gaze estimation ONNX file path.
  -v VIDEO, --video VIDEO
    Video file path or camera index.
  -i IMAGES_DIR, --images_dir IMAGES_DIR
    jpg, png images folder path.
  -ep {cpu,cuda,tensorrt}, --execution_provider {cpu,cuda,tensorrt}
    Execution provider for ONNXRuntime.
  -it {fp16,int8}, --inference_type {fp16,int8}
    Inference type. Default: fp16
  -dvw, --disable_video_writer
    Disable video writer. Eliminates the file I/O load associated with
    automatic recording to MP4. Devices that use a MicroSD card or
    similar for main storage can speed up overall processing.
  -dwk, --disable_waitKey
    Disable cv2.waitKey(). When you want to process a batch of still images,
    disable key-input wait and process them continuously.
  -ost OBJECT_SOCRE_THRESHOLD,
    --object_socre_threshold OBJECT_SOCRE_THRESHOLD
    The detection score threshold for object detection. Default: 0.35
  -ast ATTRIBUTE_SOCRE_THRESHOLD,
    --attribute_socre_threshold ATTRIBUTE_SOCRE_THRESHOLD
    The attribute score threshold for object detection. Default: 0.70
  -kst KEYPOINT_THRESHOLD, --keypoint_threshold KEYPOINT_THRESHOLD
    The keypoint score threshold for object detection. Default: 0.30
  -kdm {dot,box,both}, --keypoint_drawing_mode {dot,box,both}
    Key Point Drawing Mode. Default: dot
  -ebm, --enable_bone_drawing_mode
    Enable bone drawing mode.
    (Press B on the keyboard to switch modes)
  -dnm, --disable_generation_identification_mode
    Disable generation identification mode.
    (Press N on the keyboard to switch modes)
  -dgm, --disable_gender_identification_mode
    Disable gender identification mode.
    (Press G on the keyboard to switch modes)
  -dlr, --disable_left_and_right_hand_identification_mode
    Disable left and right hand identification mode.
    (Press H on the keyboard to switch modes)
  -dhm, --disable_headpose_identification_mode
    Disable HeadPose identification mode.
    (Press P on the keyboard to switch modes)
  -drc [DISABLE_RENDER_CLASSIDS ...],
    --disable_render_classids [DISABLE_RENDER_CLASSIDS ...]
    Class ID to disable bounding box drawing. List[int].
    e.g. -drc 17 18 19
  -efm, --enable_face_mosaic
    Enable face mosaic.
  -dtk, --disable_tracking
    Disable instance tracking.
    (Press R on the keyboard to switch modes)
  -dti, --disable_trackid_overlay
    Disable TrackID overlay.
    (Press T on the keyboard to switch modes)
  -dhd, --disable_head_distance_measurement
    Disable Head distance measurement.
    (Press M on the keyboard to switch modes)
  -oyt, --output_yolo_format_text
    Output YOLO format texts and images.
  -bblw BOUNDING_BOX_LINE_WIDTH,
    --bounding_box_line_width BOUNDING_BOX_LINE_WIDTH
    Bounding box line width. Default: 2
  -chf CAMERA_HORIZONTAL_FOV,
    --camera_horizontal_fov CAMERA_HORIZONTAL_FOV
    Camera horizontal FOV. Default: 90
  --gazelle_heatmap_alpha GAZELLE_HEATMAP_ALPHA
    Alpha blending ratio for Gazelle heatmap overlay.
    Range: 0.0-1.0. Default: 0.35
  --disable_gazelle_heatmap
    Disable Gazelle gaze heatmap rendering.
```

## Inference speed benchmark
```bash
./benchmark.sh -h
Usage: ./benchmark.sh [--runs N] [--oep PROVIDER] <onnx_model> [<onnx_model> ...]
```
- CPU
  ```bash
  ./benchmark.sh --oep cpu \
  gazelle_hgnetv2_atto_distill_1x3x320x320_1xNx4.onnx \
  gazelle_hgnetv2_femto_distill_1x3x416x416_1xNx4.onnx \
  gazelle_hgnetv2_pico_distill_1x3x640x640_1xNx4.onnx \
  gazelle_hgnetv2_n_distill_1x3x640x640_1xNx4.onnx
  ```
- CUDA
  ```bash
  ./benchmark.sh --oep cuda \
  gazelle_hgnetv2_atto_distill_1x3x320x320_1xNx4.onnx \
  gazelle_hgnetv2_femto_distill_1x3x416x416_1xNx4.onnx \
  gazelle_hgnetv2_pico_distill_1x3x640x640_1xNx4.onnx \
  gazelle_hgnetv2_n_distill_1x3x640x640_1xNx4.onnx
  ```
- TensorRT
  ```bash
  ./benchmark.sh --oep tensorrt \
  gazelle_hgnetv2_atto_distill_1x3x320x320_1xNx4.onnx \
  gazelle_hgnetv2_femto_distill_1x3x416x416_1xNx4.onnx \
  gazelle_hgnetv2_pico_distill_1x3x640x640_1xNx4.onnx \
  gazelle_hgnetv2_n_distill_1x3x640x640_1xNx4.onnx
  ```

## Citation
If you find this project useful, please consider citing:
```bibtex
@software{Hyodo_2025_gazelle_dinov3,
  author    = {Katsuya Hyodo},
  title     = {gazelle-dinov3: Gaze-LLE-DINOv3},
  year      = {2025},
  month     = {oct},
  publisher = {Zenodo},
  version   = {1.0.0},
  doi       = {10.5281/zenodo.17413165},
  url       = {https://github.com/PINTO0309/gazelle-dinov3},
  abstract  = {A model for activating human gaze regions using heat maps, built with DINOv3.},
}
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
- https://github.com/Peterande/D-FINE
  ```bibtex
  @misc{peng2024dfine,
    title={D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution Refinement},
    author={Yansong Peng and Hebei Li and Peixi Wu and Yueyi Zhang and Xiaoyan Sun and Feng Wu},
    year={2024},
    eprint={2410.13842},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
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
- https://github.com/PINTO0309/PINTO_model_zoo/tree/main/472_DEIMv2-Wholebody34
  ```bibtex
  @software{DEIMv2-Wholebody34,
    author={Katsuya Hyodo},
    title={Lightweight human detection models generated on high-quality human data sets. It can detect objects with high accuracy and speed in a total of 28 classes: body, adult, child, male, female, body_with_wheelchair, body_with_crutches, head, front, right-front, right-side, right-back, back, left-back, left-side, left-front, face, eye, nose, mouth, ear, collarbone, shoulder, solar_plexus, elbow, wrist, hand, hand_left, hand_right, abdomen, hip_joint, knee, ankle, foot.},
    url={https://github.com/PINTO0309/PINTO_model_zoo/tree/main/472_DEIMv2-Wholebody34},
    year={2025},
    month={10},
    doi={10.5281/zenodo.10229410}
  }
  ```
- https://github.com/PINTO0309/PINTO_model_zoo/tree/main/462_Gaze-LLE
