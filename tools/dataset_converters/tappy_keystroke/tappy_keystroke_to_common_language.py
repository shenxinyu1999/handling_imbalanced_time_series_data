#!/usr/bin/env python
# encoding: utf-8

# Usage:
#
# 1. Basic Usage
#       PYTHONPATH="$(dirname -- $0)/..":$PYTHONPATH \
#           python tools/dataset_converters/tappy_keystroke/tappy_keystroke_to_common_language.py \
#           -i /home1/zjin8285/00_Data/tappy_keystroke/raw \
#           -o /home1/zjin8285/00_Data/tappy_keystroke/processed \
#           --train_test_split_ratio 0.8 \
#           --min_lines_in_data 50

import argparse
import os
import os.path as osp
import random
import subprocess
import re
from tqdm import tqdm
import pandas as pd
import numpy as np
import termplotlib as tpl

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
        default=None,
        type=float,
        help='percentage of trainset verses whole dataset')
    parser.add_argument(
        '--train_users_path',
        default=None,
        type=str,
        help='path of train users file path.')
    parser.add_argument(
        '--test_users_path',
        default=None,
        type=str,
        help='path of test users file path.')
    parser.add_argument(
        '--split_max_time_interval',
        default=None,
        type=float,
        help='split data by maximum time interval')
    parser.add_argument(
        '--min_lines_in_data',
        default=100,
        type=int,
        help='minumum lines in one data file')
    return parser.parse_args()

def get_data(data_path):
    rows = []
    colNames = ['UserID','Date','Timestamp','Hand','HoldTime','Direction','LatencyTime','FlightTime']

    userID = osp.basename(data_path).rsplit('.',1)[0].split('_')[0]

    f = open(os.path.join(data_path))
    lines = f.readlines()

    for line in lines:
        line = line[:-1]
        line_len = len(line)
        occur = [m.start() for m in re.finditer('(?='+userID+')', line)]
        occur_count = len(occur)
        for idx in range(occur_count):
            occur_idx = occur[idx]
            next_idx = occur[idx+1] if idx < occur_count - 1 else line_len

            cur_line = line[occur_idx:next_idx]
            lineSplit = cur_line.split('\t')

            if lineSplit[0] != userID:
                print(lineSplit)
            elif len(lineSplit) < 8:
                pass
            else:
                try:
                    lineSplit[4] = float(lineSplit[4])
                    lineSplit[6] = float(lineSplit[6])
                    lineSplit[7] = float(lineSplit[7])
                    rows.append(lineSplit[0:8])
                except ValueError:
                    pass
    f.close()
    dataDF = pd.DataFrame(rows, columns=colNames)
    return dataDF

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

    # check train-test-split by code or load from files
    is_split_by_code = args.train_test_split_ratio is not None
    is_split_by_file = (args.train_users_path is not None) and (args.test_users_path is not None)
    assert is_split_by_code != is_split_by_file, \
        f"is_split_by_code={is_split_by_code} should not be same as is_split_by_file={is_split_by_file}."
    if is_split_by_file:
        train_users_list = np.array(pd.read_csv(args.train_users_path, header=None)).flatten().tolist()
        test_users_list = np.array(pd.read_csv(args.test_users_path, header=None)).flatten().tolist()

    # set in folder
    in_folder = args.input
    in_data_folder = osp.join(in_folder, "TappyData")
    in_label_folder = osp.join(in_folder, "Users")

    # set out folder
    out_folder = args.output

    # split user to train or test
    user_train_test_state_dict = {}

    # statistic collect timeinterval
    timeintervals_list = []

    # go through all data folder
    data_names = sorted(os.listdir(in_data_folder))
    for data_name in tqdm(data_names):

        # get src data path
        src_data_path = osp.join(in_data_folder, data_name)
        if not osp.getsize(src_data_path):
            continue
        #src_data_df = pd.read_csv(src_data_path, on_bad_lines='skip', header=None, sep="\s*\t\s*", engine='python')
        #src_data_df.columns = ['UserID','Date','Timestamp','Hand','HoldTime','Direction','LatencyTime','FlightTime']
        src_data_df = get_data(src_data_path)

        # get label path
        user_name = data_name.split('_', 1)[0]
        label_path = osp.join(in_label_folder, "User_" + user_name + ".txt")
        if not osp.isfile(label_path):
            continue
        # get label
        label = get_label(label_path)

        # filter Hand
        src_data_df = src_data_df[src_data_df['Hand'].isin(['L','R','S'])]

        # filter Direction
        src_data_df = src_data_df[src_data_df['Direction'].isin(['LL','LR','LS','RL','RR','RS','SL','SR','SS'])]

        # # filter non float value in holdtime/latency/flytime
        # src_data_df = src_data_df[pd.to_numeric(src_data_df['HoldTime'], errors='coerce').notnull()]
        # src_data_df = src_data_df[pd.to_numeric(src_data_df['LatencyTime'], errors='coerce').notnull()]
        # src_data_df = src_data_df[pd.to_numeric(src_data_df['FlightTime'], errors='coerce').notnull()]

        # filter HoldTime
        src_data_df = src_data_df[src_data_df['HoldTime'] < 10000]

        # drop na
        src_data_df = src_data_df.dropna()

        # delete duplicate rows
        src_data_df = src_data_df.drop_duplicates()

        # group sort by date
        src_data_df = [y for x, y in src_data_df.groupby('Date')]
        src_data_df = [sub_df.sort_values(by='Timestamp') for sub_df in src_data_df]

        # iterate over all dates
        for src_data_subdf in src_data_df:

            # filter number of lines
            if src_data_subdf.shape[0] < args.min_lines_in_data:
                continue

            # split train or test
            if is_split_by_code:
                if user_name in user_train_test_state_dict:
                    train_test_state = user_train_test_state_dict[user_name]
                else:
                    if random.uniform(0,1) <= args.train_test_split_ratio:
                        train_test_state = "train"
                    else:
                        train_test_state = "test"
                    user_train_test_state_dict[user_name] = train_test_state
            elif is_split_by_file:
                if user_name in train_users_list:
                    train_test_state = "train"
                elif user_name in test_users_list:
                    train_test_state = "test"
                else:
                    raise ValueError(f"user_name={user_name} cannot be found in train_users_list or test_users_list.")

            # split base on timeinterval
            if args.split_max_time_interval is not None:

                timestamps = np.array(pd.to_timedelta(src_data_subdf['Timestamp']).dt.total_seconds())
                timediff = timestamps[1:] - timestamps[:-1]
                mask_timediff = timediff > args.split_max_time_interval
                cum_mask_timediff = np.cumsum(mask_timediff)

                # here cum_mask's length is smaller than df by 1, so we pad 0 at the beginning
                cum_mask_timediff = np.pad(cum_mask_timediff, (1,0), 'constant', constant_values=(0,0))

                # iterate over time intervals
                for time_interval_idx in np.unique(cum_mask_timediff).tolist():
                    mask_time_interval = cum_mask_timediff == time_interval_idx
                    src_data_subsubdf = src_data_subdf[mask_time_interval]

                    # filter number of lines
                    if src_data_subsubdf.shape[0] < args.min_lines_in_data:
                        continue

                    # set dst data path
                    dst_data_path = osp.join(out_folder, CLASSES['identity']['class_info'][label]['name'], train_test_state,
                                             user_name + "_" + str(src_data_subdf.iat[1,1]) + "_" + str(time_interval_idx) + ".txt")
                    os.makedirs(osp.dirname(dst_data_path), exist_ok=True)

                    # save sub dataframe into file
                    src_data_subsubdf.to_csv(dst_data_path, sep='\t', header=False, index=False)

                    # statistic time interval
                    sub_timestamps = np.array(pd.to_timedelta(src_data_subsubdf['Timestamp']).dt.total_seconds())
                    sub_timediff = sub_timestamps[1:] - sub_timestamps[:-1]
                    timeintervals_list.append(sub_timediff)

            # split base on date
            else:
                # set dst data path
                dst_data_path = osp.join(out_folder, CLASSES['identity']['class_info'][label]['name'], train_test_state,
                                         user_name + "_" + str(src_data_subdf.iat[1,1]) + ".txt")
                os.makedirs(osp.dirname(dst_data_path), exist_ok=True)

                # save sub dataframe into file
                src_data_subdf.to_csv(dst_data_path, sep='\t', header=False, index=False)

                # statistic time interval
                timestamps = np.array(pd.to_timedelta(src_data_subdf['Timestamp']).dt.total_seconds())
                timediff = timestamps[1:] - timestamps[:-1]
                timeintervals_list.append(timediff)

    # statistic
    print("# original data: ", len(data_names),
          "\n# original user: ", len(os.listdir(in_label_folder)),
          "\n# train neg data: ", subprocess.check_output("find {} -type f| wc -l".format(osp.join(out_folder, "Negative", "train")), shell=True).decode('utf-8').strip(),
          "\n# train pos data: ", subprocess.check_output("find {} -type f| wc -l".format(osp.join(out_folder, "Positive", "train")), shell=True).decode('utf-8').strip(),
          "\n# train total data: ", subprocess.check_output("find {} -type f|rev| cut -d/ -f2| rev|grep train | wc -l".format(out_folder), shell=True).decode('utf-8').strip(),
          "\n# test neg data: ", subprocess.check_output("find {} -type f| wc -l".format(osp.join(out_folder, "Negative", "test")), shell=True).decode('utf-8').strip(),
          "\n# test pos data: ", subprocess.check_output("find {} -type f| wc -l".format(osp.join(out_folder, "Positive", "test")), shell=True).decode('utf-8').strip(),
          "\n# test total data", subprocess.check_output("find {} -type f|rev| cut -d/ -f2| rev|grep test| wc -l".format(out_folder), shell=True).decode('utf-8').strip(),
          "\n# filtered total data", subprocess.check_output("find {} -type f| wc -l".format(out_folder), shell=True).decode('utf-8').strip())
    if is_split_by_code:
        print("# train user: ", len([k for k, v in user_train_test_state_dict.items() if v == 'train']),
              "\n# test user: ", len([k for k, v in user_train_test_state_dict.items() if v == 'test']),
              "\n# filtered total user: ", len(user_train_test_state_dict))
    elif is_split_by_file:
        print("# train user: ", len(train_users_list),
              "\n# test user: ", len(test_users_list),
              "\n# filtered total user: ", len(train_users_list)+len(test_users_list))

    # statistic timeinterval
    print("timeintervals:")
    timeintervals_list = np.concatenate(timeintervals_list, axis=0)
    fig = tpl.figure()
    counts, bin_edges = np.histogram(timeintervals_list, bins='sqrt')
    fig.hist(counts, bin_edges, orientation="horizontal", force_ascii=False)
    fig.show()

if __name__ == '__main__':
    main()

