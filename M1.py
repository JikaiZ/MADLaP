#####################################################
# Functions for Module 1 - report-study matching
#####################################################

import pandas as pd
import numpy as np
import os
import datetime
import json
import collections
from tqdm import tqdm
from datetime import datetime, timedelta

# Folder Names for thyroid studies
diagnostic_folder_names = ['US-SOFT-TISSUE-HEAD-AND-NECK']
biopsy_folder_names = ['US-THYROID-NON-CYST-FINE-NEEDLE-ASPIRATION', 'US-THYROID-BIOPSY', 'US-FNA-THYROID-IMAGES']
pathology_folder_names = ['US-THYROID-IMAGES']


def get_dir_path(df, folder_name, study_folder_names):
    """
    get closest study directory up to pathology date
    
        param:
            df: Input dataframe with columns: MRN, Encounter Identifier, Report Date, Report Text
            folder_name: where data was stored
            study_folder_names: Folder names for specific thyroid studies
        return:
            selected_dir_paths: selected directory paths after matching date and report by datetimes
            selected_folder_datetimes: selected datetimes

    
    """
    selected_dir_paths = []
    selected_folder_datetimes = []
    for i in tqdm(range(len(df))):
        # mrn, date from reports
        mrn = df['MRN'].astype(str).iloc[i]
        report_date = df['Report Date'].iloc[i]
        # info from study folders
        study_folder = os.path.join(folder_name, mrn)
        dirnames = [x[0] for x in os.walk(study_folder)][1:]
        # get all study folder names
        study_names = [x.split('/')[-1].split('_')[0] for x in dirnames]
        # get the study whose names are in the name list
        study_id = np.where([x in study_folder_names for x in study_names])[0]
        if len(study_id) == 0:
            selected_dir_paths.append("")
            selected_folder_datetimes.append("")
            continue
        thyroid_dirnames = [dirnames[x] for x in study_id]
        # parse the dates
        selected_dates = [x.split('/')[-1].split('_')[-2:] for x in thyroid_dirnames]
        selected_dates = [x[0]+x[1] for x in selected_dates]
        selected_dates = [datetime(year=int(s[0:4]), month=int(s[4:6]), day=int(s[6:8]), hour=int(s[8:10]), minute=int(s[10:12]), second=int(s[12:14])) for s in selected_dates]
        selected_datetime = [x.strftime("%Y-%m-%d %H:%M:%S") for x in selected_dates]
        # find the closest study
        diff = np.array([(report_date - x).total_seconds() for x in selected_dates], dtype=np.int32)
        # a pathology report should be created AFTER an exam was done
        diff_6months = (60*60*24)*30*6
        diff[(diff <= 0)] = diff_6months
        closest_idx = np.argmin(diff)
        selected_study_dirname = thyroid_dirnames[closest_idx]
        selected_folder_datetime = selected_datetime[closest_idx]
        selected_folder_datetimes.append(selected_folder_datetime)
        
        if (np.min(diff) >= diff_6months):
            selected_dir_paths.append([])
        else:
            selected_dir_paths.append(selected_study_dirname)
        
    return selected_dir_paths, selected_folder_datetimes

def get_mrnenc(report_df, selected_mrns):
    """
    ger mrn + encounter id

        Parameters:
            report_df: input dataframe weith column: MRN, Encounter Identifier
            selected_mrns: mrns in the analysis

        Returns:
            all_mrn_encounter: list of MRN+Encounter ID. E.g: AA1234567_0987654321
    """
    all_mrn = list(report_df['MRN'][report_df['MRN'].astype(str).isin(selected_mrns)].astype('str'))
    all_encounter = list(report_df['Encounter Identifier'][report_df['MRN'].astype(str).isin(selected_mrns)].astype('str'))
    all_mrn_encounter = [x + '-' + y for x, y in zip(all_mrn, all_encounter)]
    return all_mrn_encounter

def get_number_studies(folder_name, selected_mrns, diagnostic_folder_names, biopsy_folder_names, pathology_folder_names):
    """
        Get number of available thyroid studies for each patient
    """
    nums_diag_study, nums_biopsy_study, nums_path_study, nums_useful_study = [], [], [], []
    for mrn in selected_mrns:
        study_folder = os.path.join(folder_name, str(mrn))
        dirnames = [x[0] for x in os.walk(study_folder)][1:]
        study_names = [x.split('/')[-1].split('_')[0] for x in dirnames]
        num_diag_study = sum([x in diagnostic_folder_names for x in study_names])
        nums_diag_study.append(num_diag_study)
        num_biopsy_study = sum([x in biopsy_folder_names for x in study_names])
        nums_biopsy_study.append(num_biopsy_study)
        num_path_study = sum([x in pathology_folder_names for x in study_names])
        nums_path_study.append(num_path_study)
        num_useful_study = sum([num_diag_study, num_biopsy_study, num_path_study])
        nums_useful_study.append(num_useful_study)
    df = pd.DataFrame({
        'MRN': selected_mrns,
        'NumberDiagStudy': nums_diag_study,
        'NumberBiopsyStudy': nums_biopsy_study,
        'NumberPathStudy': nums_path_study,
        'NumberTotStudy': nums_useful_study
    })
    return df