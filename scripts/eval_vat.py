import argparse
import torch
from PIL import Image
import json
import os
import numpy as np
from sklearn.metrics import average_precision_score
from tqdm import tqdm

from gazelle.model import get_gazelle_model
from gazelle.utils import vat_auc, vat_l2

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="./data/videoattentiontarget")
parser.add_argument("--model_name", type=str, default="gazelle_dinov2_vitl14_inout")
parser.add_argument("--ckpt_path", type=str, default="./checkpoints/gazelle_dinov2_vitl14_inout.pt")
parser.add_argument("--batch_size", type=int, default=64)
args = parser.parse_args()

class VideoAttentionTarget(torch.utils.data.Dataset):
    def __init__(self, path, img_transform):
        self.sequences = json.load(open(os.path.join(path, "test_preprocessed.json"), "rb"))
        self.frames = []
        for i in range(len(self.sequences)):
            for j in range(len(self.sequences[i]['frames'])):
                self.frames.append((i, j))
        self.path = path
        self.transform = img_transform

    def __getitem__(self, idx):
        seq_idx, frame_idx = self.frames[idx]
        seq = self.sequences[seq_idx]
        frame = seq['frames'][frame_idx]
        image = self.transform(Image.open(os.path.join(self.path, frame['path'])).convert("RGB"))
        bboxes = [head['bbox_norm'] for head in frame['heads']]
        gazex = [head['gazex_norm'] for head in frame['heads']]
        gazey = [head['gazey_norm'] for head in frame['heads']]
        inout = [head['inout'] for head in frame['heads']]

        return image, bboxes, gazex, gazey, inout

    def __len__(self):
        return len(self.frames)
    
def collate(batch):
    images, bboxes, gazex, gazey, inout = zip(*batch)
    return torch.stack(images), list(bboxes), list(gazex), list(gazey), list(inout)


@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running on {}".format(device))

    model, transform = get_gazelle_model(args.model_name)
    model.load_gazelle_state_dict(torch.load(args.ckpt_path, weights_only=True))
    model.to(device)
    model.eval()

    dataset = VideoAttentionTarget(args.data_path, transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate)

    aucs = []
    l2s = []
    inout_preds = []
    inout_gts = []

    for _, (images, bboxes, gazex, gazey, inout) in tqdm(enumerate(dataloader), desc="Evaluating", total=len(dataloader)):
        preds = model.forward({"images": images.to(device), "bboxes": bboxes})
        
        # eval each instance (head)
        for i in range(images.shape[0]): # per image
            for j in range(len(bboxes[i])): # per head
                if inout[i][j] == 1: # in frame
                    auc = vat_auc(preds['heatmap'][i][j], gazex[i][j][0], gazey[i][j][0])
                    l2 = vat_l2(preds['heatmap'][i][j], gazex[i][j][0], gazey[i][j][0])
                    aucs.append(auc)
                    l2s.append(l2)
                inout_preds.append(preds['inout'][i][j].item())
                inout_gts.append(inout[i][j])

    
    print("AUC: {}".format(np.array(aucs).mean()))
    print("Avg L2: {}".format(np.array(l2s).mean()))
    print("Inout AP: {}".format(average_precision_score(inout_gts, inout_preds)))

        
if __name__ == "__main__":
    main()