#!/usr/bin/env python
# encoding: utf-8

# Usage:
#
# 1. Basic Usage
#       PYTHONPATH="$(dirname -- $0)/..":$PYTHONPATH \
#           python tools/dataset_converters/tappy_keystroke/tappy_keystroke_to_common_language.py \
#           -i /home1/zjin8285/03_gits/handling_imbalanced_time_series_data/data \
#           -o /home1/zjin8285/00_Data/tappy_keystroke \
#           --train_test_split_ratio 0.8 \
#           --min_lines_in_data 100

import argparse
import os
import os.path as osp
import random
import subprocess
from tqdm import tqdm

CLASSES = {
    "identity": {"class_info": [dict(id=0, name='Negative'),
                                dict(id=1, name='Positive')],
                 "class_map": {i:i for i in range(2)}}
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert Tappy KeyStroke to CommonLanguage Style.')
    parser.add_argument('-i', '--input', help='path of VisDrone MOT data')
    parser.add_argument(
        '-o', '--output', help='path to save coco formatted label file')
    parser.add_argument(
        '--train_test_split_ratio',
        default=0.7,
        type=float,
        help='percentage of trainset verses whole dataset')
    parser.add_argument(
        '--min_lines_in_data',
        default=100,
        type=int,
        help='minumum lines in one data file')
    return parser.parse_args()

def get_label(label_path):

    f = open(label_path)
    lines = f.readlines()
    for line in lines:
        lineSplit = line.strip().split(': ')

        if lineSplit[0] == 'Parkinsons':
            return lineSplit[-1] == 'True'

    return False

def main():
    args = parse_args()
    assert not osp.exists(args.output), f"{args.output} is already exists, please make sure its empty."
    if not osp.isdir(args.output):
        os.makedirs(args.output)

    # set in folder
    in_folder = args.input
    in_data_folder = osp.join(in_folder, "TappyData")
    in_label_folder = osp.join(in_folder, "Users")

    # set out folder
    out_folder = args.output

    # split user to train or test
    user_train_test_state_dict = {}

    # go through all data folder
    data_names = sorted(os.listdir(in_data_folder))
    for data_name in tqdm(data_names):

        # get src data path
        src_data_path = osp.join(in_data_folder, data_name)

        # filter number of lines
        if sum(1 for line in open(src_data_path)) < args.min_lines_in_data:
            continue

        # get label path
        user_name = data_name.split('_', 1)[0]
        label_path = osp.join(in_label_folder, "User_" + user_name + ".txt")
        if not osp.isfile(label_path):
            continue

        # get label
        label = get_label(label_path)

        # split train or test
        if user_name in user_train_test_state_dict:
            train_test_state = user_train_test_state_dict[user_name]
        else:
            if random.uniform(0,1) <= args.train_test_split_ratio:
                train_test_state = "train"
            else:
                train_test_state = "test"
            user_train_test_state_dict[user_name] = train_test_state

        # set dst data path
        dst_data_path = osp.join(out_folder, CLASSES['identity']['class_info'][label]['name'], train_test_state, data_name)
        os.makedirs(osp.dirname(dst_data_path), exist_ok=True)

        # soft link
        os.symlink(src_data_path, dst_data_path)

    # statistic
    print("# original data: ", len(data_names),
          "\n# original user: ", len(os.listdir(in_label_folder)),
          "\n# train neg data: ", subprocess.check_output("find {} -type l| wc -l".format(osp.join(out_folder, "Negative", "train")), shell=True).decode('utf-8').strip(),
          "\n# train pos data: ", subprocess.check_output("find {} -type l| wc -l".format(osp.join(out_folder, "Positive", "train")), shell=True).decode('utf-8').strip(),
          "\n# train total data: ", subprocess.check_output("find {} -type l|grep train | wc -l".format(out_folder), shell=True).decode('utf-8').strip(),
          "\n# test neg data: ", subprocess.check_output("find {} -type l| wc -l".format(osp.join(out_folder, "Negative", "test")), shell=True).decode('utf-8').strip(),
          "\n# test pos data: ", subprocess.check_output("find {} -type l| wc -l".format(osp.join(out_folder, "Positive", "test")), shell=True).decode('utf-8').strip(),
          "\n# test total data", subprocess.check_output("find {} -type l|grep test| wc -l".format(out_folder), shell=True).decode('utf-8').strip(),
          "\n# filtered total data", subprocess.check_output("find {} -type l| wc -l".format(out_folder), shell=True).decode('utf-8').strip(),
          "\n# train user: ", len([k for k, v in user_train_test_state_dict.items() if v == 'train']),
          "\n# test user: ", len([k for k, v in user_train_test_state_dict.items() if v == 'test']),
          "\n# filtered total user: ", len(user_train_test_state_dict))

if __name__ == '__main__':
    main()

