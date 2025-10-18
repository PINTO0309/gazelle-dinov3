import argparse
import glob
from functools import reduce
import os
import pandas as pd
import json
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="./data/videoattentiontarget")
args = parser.parse_args()

# preprocessing adapted from https://github.com/ejcgt/attention-target-detection/blob/master/dataset.py

def merge_dfs(ls):
    for i, df in enumerate(ls): # give columns unique names
        df.columns = [col if col == "path" else f"{col}_df{i}" for col in df.columns]
    merged_df = reduce(
        lambda left, right: pd.merge(left, right, on=["path"], how="outer"), ls
    )
    merged_df = merged_df.sort_values(by=["path"])
    merged_df = merged_df.reset_index(drop=True)
    return merged_df

def smooth_by_conv(window_size, df, col):
    """Temporal smoothing on labels to match original VideoAttTarget evaluation.
    Adapted from https://github.com/ejcgt/attention-target-detection/blob/acd264a3c9e6002b71244dea8c1873e5c5818500/utils/myutils.py"""
    values = df[col].values
    padded_track = np.concatenate([values[0].repeat(window_size // 2), values, values[-1].repeat(window_size // 2)])
    smoothed_signals = np.convolve(
        padded_track.squeeze(), np.ones(window_size) / window_size, mode="valid"
    )
    return smoothed_signals

def smooth_df(window_size, df):
    df["xmin"] = smooth_by_conv(window_size, df, "xmin")
    df["ymin"] = smooth_by_conv(window_size, df, "ymin")
    df["xmax"] = smooth_by_conv(window_size, df, "xmax")
    df["ymax"] = smooth_by_conv(window_size, df, "ymax")
    return df


def main(PATH):
    # preprocess by sequence and person track
    splits = ["train", "test"]

    for split in splits:
        sequences = []
        max_num_ppl = 0
        seq_idx = 0
        for seq_path in glob.glob(
            os.path.join(PATH, "annotations", split, "*", "*")
        ):
            seq_img_path = os.path.join("images", *seq_path.split("/")[-2:]
            )
            sample_image = os.path.join(PATH, seq_img_path, os.listdir(os.path.join(PATH, seq_img_path))[0])
            width, height = Image.open(sample_image).size
            seq_dict = {"path": seq_img_path, "width": width, "height": height}
            frames = []
            person_files = glob.glob(os.path.join(seq_path, "*"))
            num_ppl = len(person_files)
            if num_ppl > max_num_ppl:
                max_num_ppl = num_ppl
            person_dfs = [
                pd.read_csv(
                    file,
                    header=None,
                    index_col=False,
                    names=["path", "xmin", "ymin", "xmax", "ymax", "gazex", "gazey"],
                )
                for file in person_files
            ]
            # moving-avg smoothing to match original benchmark's evaluation
            window_size = 11
            person_dfs = [smooth_df(window_size, df) for df in person_dfs]
            merged_df = merge_dfs(person_dfs) # merge annotations per person for same frames
            for frame_idx, row in merged_df.iterrows():
                frame_dict = {
                    "path": os.path.join(seq_img_path, row["path"]),
                    "heads": [],
                }
                p_idx = 0
                for i in range(1, num_ppl * 6 + 1, 6):
                    if not np.isnan(row.iloc[i]): # if it's nan lack of continuity (one person leaving the frame for a period of time)
                        xmin, ymin, xmax, ymax, gazex, gazey = row[i: i+6].values.tolist()
                        # match original benchmark's preprocessing of annotations
                        if gazex >=0 and gazey < 0:
                            gazey = 0
                        elif gazey >=0 and gazex < 0:
                            gazex = 0
                        inout = int(gazex >= 0 and gazey >= 0)

                        # move bboxes within frame if necessary
                        xmin = max(xmin, 0)
                        ymin = max(ymin, 0)
                        xmax = min(xmax, width)
                        ymax = min(ymax, height)

                        frame_dict["heads"].append({
                            "bbox": [xmin, ymin, xmax, ymax],
                            "bbox_norm": [xmin / float(width), ymin / float(height), xmax / float(width), ymax / float(height)],
                            "gazex": [gazex],
                            "gazex_norm": [gazex / float(width)],
                            "gazey": [gazey],
                            "gazey_norm": [gazey / float(height)],
                            "inout": inout
                        })
                    p_idx = p_idx + 1

                frames.append(frame_dict)
            seq_dict["frames"] = frames
            sequences.append(seq_dict)
            seq_idx += 1

        print("{} max people per image {}".format(split, max_num_ppl))
        print("{} num unique video sequences {}".format(split, len(sequences)))

        out_file = open(os.path.join(PATH, "{}_preprocessed.json".format(split)), "w")
        json.dump(sequences, out_file)

if __name__ == "__main__":
    main(args.data_path)