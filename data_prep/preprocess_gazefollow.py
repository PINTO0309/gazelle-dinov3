import os
import pandas as pd
import json
from PIL import Image
import argparse

# preprocessing adapted from https://github.com/ejcgt/attention-target-detection/blob/master/dataset.py

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="./data/gazefollow_extended")
args = parser.parse_args()


def main(data_path: str):

    # TRAIN
    train_csv_path = os.path.join(data_path, "train_annotations_release.txt")
    column_names = [
        'path',
        'idx',
        'body_bbox_x',
        'body_bbox_y',
        'body_bbox_w',
        'body_bbox_h',
        'eye_x',
        'eye_y',
        'gaze_x',
        'gaze_y',
        'bbox_x_min',
        'bbox_y_min',
        'bbox_x_max',
        'bbox_y_max',
        'inout',
        'source',
        'meta',
    ]
    df = pd.read_csv(train_csv_path, header=None, names=column_names, index_col=False)
    df = df[df['inout'] != -1]
    df = df.groupby("path").agg(list) # aggregate over frames

    multiperson_ex = 0
    TRAIN_FRAMES = []
    for path, row in df.iterrows():
        img_path = os.path.join(data_path, path)
        img = Image.open(img_path)
        width, height = img.size

        num_people = len(row['idx'])
        if num_people > 1:
            multiperson_ex += 1
        heads = []

        for i in range(num_people):
            xmin, ymin, xmax, ymax = row['bbox_x_min'][i], row['bbox_y_min'][i], row['bbox_x_max'][i], row['bbox_y_max'][i]
            gazex = row['gaze_x'][i] * float(width)
            gazey = row['gaze_y'][i] * float(height)
            gazex_norm = row['gaze_x'][i]
            gazey_norm = row['gaze_y'][i]


            if xmin > xmax:
                temp = xmin
                xmin = xmax
                xmax = temp
            if ymin > ymax:
                temp = ymin
                ymin = ymax
                ymax = temp

            # move in out of frame bbox annotations
            xmin = max(xmin, 0)
            ymin = max(ymin, 0)
            xmax = min(xmax, width)
            ymax = min(ymax, height)

            heads.append({
                'bbox': [xmin, ymin, xmax, ymax],
                'bbox_norm': [xmin / float(width), ymin / float(height), xmax / float(width), ymax / float(height)],
                'inout': row['inout'][i],
                'gazex': [gazex], # convert to list for consistency with multi-annotation format
                'gazey': [gazey],
                'gazex_norm': [gazex_norm],
                'gazey_norm': [gazey_norm],
                'head_id': i
            })
        TRAIN_FRAMES.append({
            'path': path,
            'heads': heads,
            'num_heads': num_people,
            'width': width,
            'height': height,
        })

    print("Train set: {} frames, {} multi-person".format(len(TRAIN_FRAMES), multiperson_ex))
    out_file = open(os.path.join(data_path, "train_preprocessed.json"), "w")
    json.dump(TRAIN_FRAMES, out_file)

    # TEST
    test_csv_path = os.path.join(data_path, "test_annotations_release.txt")
    column_names = [
        'path',
        'idx',
        'body_bbox_x',
        'body_bbox_y',
        'body_bbox_w',
        'body_bbox_h',
        'eye_x',
        'eye_y',
        'gaze_x',
        'gaze_y',
        'bbox_x_min',
        'bbox_y_min',
        'bbox_x_max',
        'bbox_y_max',
        'source',
        'meta',
    ]
    df = pd.read_csv(test_csv_path, header=None, names=column_names, index_col=False)

    TEST_FRAME_DICT = {}
    df = df.groupby(["path", "eye_x"]).agg(list) # aggregate over frames
    for id, row in df.iterrows(): # aggregate by frame
        path, _ = id
        if path in TEST_FRAME_DICT.keys():
            TEST_FRAME_DICT[path].append(row)
        else:
            TEST_FRAME_DICT[path] = [row]

    multiperson_ex = 0
    TEST_FRAMES = []
    for path in TEST_FRAME_DICT.keys():
        img_path = os.path.join(data_path, path)
        img = Image.open(img_path)
        width, height = img.size

        item = TEST_FRAME_DICT[path]
        num_people = len(item)
        heads = []

        for i in range(num_people):
            row = item[i]
            assert(row['bbox_x_min'].count(row['bbox_x_min'][0]) == len(row['bbox_x_min'])) # quick check that all bboxes are equivalent
            xmin, ymin, xmax, ymax = row['bbox_x_min'][0], row['bbox_y_min'][0], row['bbox_x_max'][0], row['bbox_y_max'][0]

            if xmin > xmax:
                temp = xmin
                xmin = xmax
                xmax = temp
            if ymin > ymax:
                temp = ymin
                ymin = ymax
                ymax = temp

            # move in out of frame bbox annotations
            xmin = max(xmin, 0)
            ymin = max(ymin, 0)
            xmax = min(xmax, width)
            ymax = min(ymax, height)

            gazex_norm = [x for x in row['gaze_x']]
            gazey_norm = [y for y in row['gaze_y']]
            gazex = [x * float(width) for x in row['gaze_x']]
            gazey = [y * float(height) for y in row['gaze_y']]

            heads.append({
                'bbox': [xmin, ymin, xmax, ymax],
                'bbox_norm': [xmin / float(width), ymin / float(height), xmax / float(width), ymax / float(height)],
                'gazex': gazex,
                'gazey': gazey,
                'gazex_norm': gazex_norm,
                'gazey_norm': gazey_norm,
                'inout': 1, # all test frames are in frame
                'num_annot': len(gazex),
                'head_id': i
            })

        TEST_FRAMES.append({
            'path': path,
            'heads': heads,
            'num_heads': num_people,
            'width': width,
            'height': height,
        })
        if num_people > 1:
            multiperson_ex += 1

    print("Test set: {} frames, {} multi-person".format(len(TEST_FRAMES), multiperson_ex))
    out_file = open(os.path.join(data_path, "test_preprocessed.json"), "w")
    json.dump(TEST_FRAMES, out_file)



if __name__ == "__main__":
    main(data_path=args.data_path)