"""
Data preparation of CommonLangauge dataset for LID.

Download: https://zenodo.org/record/5036977#.YNo1mHVKg5k

Author
------
Pavlo Ruban 2021
"""

import subprocess
import os
import csv
import logging
import torchaudio
from tqdm.contrib import tzip
from speechbrain.utils.data_utils import get_all_files

logger = logging.getLogger(__name__)


PARKINSONS = [
    "Negative",
    "Positive"
]


def prepare_tappy_keystroke(data_folder, save_folder, skip_prep=False):
    """
    Prepares the csv files for the CommonLanguage dataset for LID.
    Download: https://drive.google.com/uc?id=1Vzgod6NEYO1oZoz_EcgpZkUO9ohQcO1F

    Arguments
    ---------
    data_folder : str
        Path to the folder where the CommonLanguage dataset for LID is stored.
        This path should include the multi: /datasets/CommonLanguage
    save_folder : str
        The directory where to store the csv files.
    max_duration : int, optional
        Max duration (in seconds) of training uterances.
    skip_prep: bool
        If True, skip data preparation.

    Example
    -------
    >>> from recipes.CommonLanguage.common_language_prepare import prepare_common_language
    >>> data_folder = '/datasets/CommonLanguage'
    >>> save_folder = 'exp/CommonLanguage_exp'
    >>> prepare_common_language(\
            data_folder,\
            save_folder,\
            skip_prep=False\
        )
    """

    if skip_prep:
        return

    # Setting the save folder
    os.makedirs(save_folder, exist_ok=True)

    # Setting ouput files
    save_csv_train = os.path.join(save_folder, "train.csv")
    save_csv_test = os.path.join(save_folder, "test.csv")

    # If csv already exists, we skip the data preparation
    if skip(save_csv_train, save_csv_test):
        csv_exists = " already exists, skipping data preparation!"
        msg = save_csv_train + csv_exists
        logger.info(msg)
        msg = save_csv_test + csv_exists
        logger.info(msg)

        return

    # Audio files extensions
    extension = [".txt"]

    # Create the signal list of train, dev, and test sets.
    data_split = create_sets(data_folder, extension)

    # Creating csv files for training, dev and test data
    create_csv(txt_list=data_split["train"], csv_file=save_csv_train)
    create_csv(txt_list=data_split["test"], csv_file=save_csv_test)


def skip(save_csv_train, save_csv_test):
    """
    Detects if the CommonLanguage data preparation for LID has been already done.

    If the preparation has been done, we can skip it.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """

    # Checking folders and save options
    skip = (
        os.path.isfile(save_csv_train)
        and os.path.isfile(save_csv_test)
    )

    return skip


def create_sets(data_folder, extension):
    """
    Creates lists for train, dev and test sets with data from the data_folder

    Arguments
    ---------
    data_folder : str
        Path of the CommonLanguage dataset.
    extension: list of file extentions
        List of strings with file extentions that correspond to the audio files
        in the CommonLanguage dataset

    Returns
    -------
    dictionary containing train, dev, and test splits.
    """

    # Datasets initialization
    datasets = {"train", "test"}
    data_split = {dataset: [] for dataset in datasets}

    # Get the list of languages from the dataset folder
    parkinsons = [
        name
        for name in sorted(os.listdir(data_folder))
        if os.path.isdir(os.path.join(data_folder, name))
        and datasets.issubset(os.listdir(os.path.join(data_folder, name)))
    ]

    msg = f"{len(parkinsons)} parkinsons detected!"
    logger.info(msg)

    # Fill the train, dev and test datasets with audio filenames
    for parkinson in parkinsons:
        for dataset in datasets:
            curr_folder = os.path.join(data_folder, parkinson, dataset)
            txt_list = get_all_files(curr_folder, match_and=extension)
            data_split[dataset].extend(txt_list)

    msg = "Data successfully split!"
    logger.info(msg)

    return data_split


def create_csv(txt_list, csv_file):
    """
    Creates the csv file given a list of wav files.

    Arguments
    ---------
    wav_list : list of str
        The list of wav files.
    csv_file : str
        The path of the output json file
    """

    # Adding some Prints
    msg = f"Creating csv lists in {csv_file} ..."
    logger.info(msg)

    csv_lines = []

    # Start processing lines
    total_duration = 0.0

    # Starting index
    idx = 0

    for txt_file in tzip(txt_list):
        txt_file = txt_file[0]

        path_parts = txt_file.split(os.path.sep)
        file_name, txt_format = os.path.splitext(path_parts[-1])

        num_lines = float(subprocess.check_output("cat {}| wc -l".format(txt_file), shell=True).decode('utf-8').strip())
        sample_rate = 733.     # TODO: cannot specify sample rate for event based time series data, so fake one

        audio_duration = num_lines / sample_rate
        total_duration += audio_duration

        # Actual name of the language
        parkinson = path_parts[-3]

        # Create a row with whole utterences
        csv_line = [
            idx,  # ID
            txt_file,  # File name
            txt_format,  # File format
            audio_duration, # TODO: faked audio duration
            parkinson,  # Whether has parkinson
        ]

        # Adding this line to the csv_lines list
        csv_lines.append(csv_line)

        # Increment index
        idx += 1

    # CSV column titles
    csv_header = ["ID", "txt", "txt_format", "duration", "parkinson"]

    # Add titles to the list at indexx 0
    csv_lines.insert(0, csv_header)

    # Writing the csv lines
    with open(csv_file, mode="w", encoding="utf-8") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        for line in csv_lines:
            csv_writer.writerow(line)

    # Final prints
    msg = f"{csv_file} sucessfully created!"
    logger.info(msg)
    msg = f"Number of samples: {len(txt_list)}."
    logger.info(msg)
    msg = f"Total duration: {round(total_duration / 3600, 2)} hours."
    logger.info(msg)

