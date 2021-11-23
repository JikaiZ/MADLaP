#####################################################
# Functions for Module 2 - NLP for searching laterality
#####################################################

import pandas as pd
import numpy as np
import os
import datetime
import collections
from tqdm import tqdm

import re

from utils import *
from nlp_utils import *

from nltk.tokenize import RegexpTokenizer, word_tokenize

def append_laterality(diagnosis_tokens, laterality_flag, diagnosis_word_set):
    """
        Add laterality to list if diagnosis is successfully checked

        Parameters:
            diagnosis_tokens: candidate tokens that could include diagnosis
            laterality_flag: parsed laterality token
            diagnosis_word_set: default diagnosis tokens

        Returns:
            lats: appended list of laterality tokens

    """
    if any([x in diagnosis_tokens for x in diagnosis_word_set]):
        if laterality_flag == 'left':
            lats.append('left')
        elif laterality_flag == 'right':
            lats.append('right')
        elif laterality_flag == 'isthmus':
            lats.append('isthmus')
    return lats

def find_laterality(test_report, laterality_word_set, location_word_set, correct_word_set, opposite_word_set, f):
    """
        Main function for extracting nodule information texts from one input report

        Parameters:
            test_report: input report texts
            laterality_word_set: default laterality tokens
            location_word_set: default location tokens
            correct_word_set: diagnosis tokens for the selected nodule category
            opposite_word_set: diagnosis tokens opposite to the selected nodule category
            f: log file

        Returns:
            locs: extracted location token
            lats: extracted laterality token
            nums: extracted nodule label # token

    """
    # add note if this line was checked for the first time
    for _, line in enumerate(test_report):
        test_report[_] += ' <first_view>'
    tokenizer = RegexpTokenizer(r'\w+')
    location = []
    lats, locs, nums = [], [], []
    for _, line in enumerate(test_report):
        # separate the current line by comma, skip if there is only one 'comma-phrase'
        sep_comma = tokenizer.tokenize(line)
        thyroid_line_num = _
        # if the current line has no more than two tokens, then move to the next line
        if len(sep_comma) < 2:
            continue
        thyroid_token_id = 0
        # get the first token. if it is a thyroid nodule line then the line must have 'Thyroid' in the first two token
        thyroid_tokens = sep_comma[thyroid_token_id]
        # Or, it can be the Specimen line with nodule + diagnosis
        # Check specimen already have nodule + diagnosis
        if ('Specimen' in thyroid_tokens):
            specimen_line = test_report[thyroid_line_num]
            specimen_tokens = tokenizer.tokenize(specimen_line)
            f.write('tokens that may contain laterality in the Specimen section: {}\n'.format( specimen_tokens))
            laterality_flag = return_laterality_flag(specimen_tokens, laterality_word_set)
            location = return_location(specimen_tokens, location_word_set)
            nodule_number = return_number(specimen_line)
            # sometimes a specimen section is near the end of a report
            if thyroid_line_num + 1 >= len(test_report):
                continue
            next_sent = test_report[thyroid_line_num + 1]
            # if a specimen section has more than two nodules or a lymph node, skip
            if ('lymph' in next_sent.lower()) or ('Thyroid' in next_sent):
                f.write('Multiple thyroid nodules in specimen section or lymph node, \n')
                continue
            diagnosis_tokens = tokenizer.tokenize(next_sent.lower())
            f.write('Diagnosis tokens in the Specimen section: {}\n'.format( diagnosis_tokens))
            # if found a diagnosis, then the search is done
            if any([x in diagnosis_tokens for x in correct_word_set]) and ('prev_nodule' not in diagnosis_tokens) and ('bad_nodule' not in diagnosis_tokens):
                f.write('Found the diagnosis in the next line! Record the laterality: {}\n'.format( laterality_flag))
                lats.append(laterality_flag)
                locs.append(location)
                nums.append(nodule_number)
                break
        # No Specimen
        else:
            #  move to the second token, check 'Thyroid' for the first token
            if 'Thyroid' not in thyroid_tokens:
                thyroid_token_id += 1
            thyroid_tokens = sep_comma[thyroid_token_id]
            # First to check if there is a 'Thyroid' or 'thyroid' token in the first or the second token
            # if found, move to check the diagnosis
            if ('Thyroid' in thyroid_tokens) or ('thyroid' in thyroid_tokens):
                lat_token_id = thyroid_token_id + 1
                # check if laterality exists in the line with 'Thyroid'
                lat_line = test_report[thyroid_line_num]
                lat_tokens = tokenizer.tokenize(lat_line)
                f.write('tokens that may contain laterality: {}\n'.format( lat_tokens))
                laterality_flag = return_laterality_flag(lat_tokens, laterality_word_set)
                location = return_location(lat_tokens, location_word_set)
                nodule_number = return_number(lat_line)
                # check the locations
                # prioritize the location from the Specimen section
                if location is None:
                    location = return_location(lat_tokens, location_word_set)
                if laterality_flag is None:
                    f.write('No laterality found! skip to the next line.\n')
                    continue

                diagnosis_line_num = thyroid_line_num + 1
                if diagnosis_line_num >= len(test_report):
                    f.write('Exceed report range, move to the next report.\n')
                    break
                diagnosis_tokens = tokenizer.tokenize(test_report[diagnosis_line_num].lower())
                f.write('candidate laterality: {}\n'.format( laterality_flag))

                # Find the diagnosis in the next line
                if any([x in diagnosis_tokens for x in correct_word_set]) and ('prev_nodule' not in diagnosis_tokens) and ('bad_nodule' not in diagnosis_tokens):
                    f.write('Diagnosis tokens in the next line: {}\n '.format( diagnosis_tokens))
                    f.write('Found the diagnosis in the next line! Record the laterality and location: {}/{}\n'.format( laterality_flag, location))
                    test_report[diagnosis_line_num] = test_report[diagnosis_line_num].replace('<first_view>', '<prev_nodule>')
                    locs.append(location)
                    lats.append(laterality_flag)
                    nums.append(nodule_number)

                # The next line is NOT the diagnosis
                # If the next line has 'Adequate', move on
                elif ('ct' in diagnosis_tokens) or ('md' in diagnosis_tokens) or ('adequacy' in diagnosis_tokens) or ('adequate' in diagnosis_tokens) or ('dr' in diagnosis_tokens):
                    f.write("Process adequate line\n")
                    diagnosis_line_num += 1
                    # if the line moves out of report range, move to the next report
                    if diagnosis_line_num >= len(test_report):
                        f.write('Exceed report range, move to the next report.\n')
                        break
                    diagnosis_tokens = test_report[diagnosis_line_num].lower().split(' ')
                    f.write("The line after the adequate line: {}\n".format( test_report[diagnosis_line_num]))
                    # if it is another thyroid, move to the next line until the search finds the diagnosis line.
                    # mark that line as for the particular thyroid, make sure the next thyroid will get the diagnosis line after it
                    if 'Thyroid' in test_report[diagnosis_line_num]:
                        f.write('Skip the next thyroid until the search finds the diagnosis\n')
                        find_diagnosis_flag = False
                        while (find_diagnosis_flag == False) and (diagnosis_line_num <= len(test_report) -1):
                            diagnosis_tokens = test_report[diagnosis_line_num].lower()
                            if (any([x in diagnosis_tokens for x in correct_word_set])) and ('<prev_nodule>' not in diagnosis_tokens) and ('<bad_nodule>' not in diagnosis_tokens):
                                f.write('Diagnosis tokens: '.format( diagnosis_tokens))
                                f.write('Found the diagnosis in the next line! Record the laterality and location: {}/{}\n'.format( laterality_flag, location))
                                find_diagnosis_flag = True
                                test_report[diagnosis_line_num] = test_report[diagnosis_line_num].replace('<first_view>', '<prev_nodule>')
                                locs.append(location)
                                lats.append(laterality_flag)
                                nums.append(nodule_number)
                            elif (any([x in diagnosis_tokens for x in opposite_word_set])) and ('<first_view>' in diagnosis_tokens):
                                f.write('This is a diagnosis line, but not the correct diagnosis\n')
                                test_report[diagnosis_line_num] = test_report[diagnosis_line_num].replace('<first_view>', '<bad_nodule>')
                                break
                            elif ('<prev_nodule>' in diagnosis_tokens) or ('<bad_nodule>' in diagnosis_tokens):
                                f.write('Found a diagnosis line, but this is for the previous nodule.\n')
                                diagnosis_line_num += 1
                            else:
                                f.write('Not a diagnosis line. Move to the next line.\n')
                                diagnosis_line_num += 1
                    # if it is not a thyroid line, check the following
                    else:
                        # if it is a diagnosis, and was not previously reviewed, then check if it has the correct diagnosis
                        if (any([x in diagnosis_tokens for x in correct_word_set])) and (('<first_view>' in diagnosis_tokens)):
                            f.write("Diagnosis checked! Record the laterality: {}\n".format( laterality_flag))
                            # mark this line as the diagnosis for the previous nodule
                            test_report[diagnosis_line_num] = test_report[diagnosis_line_num].replace('<first_view>', '<prev_nodule>')
                            locs.append(location)
                            lats.append(laterality_flag)
                            nums.append(nodule_number)
                        elif '<first_view>' not in diagnosis_tokens:
                            f.write('Found a diagnosis line, but this is for the previous nodule.\n')
                            diagnosis_line_num += 1
                            find_diagnosis_flag = False
                            while (find_diagnosis_flag == False) and (diagnosis_line_num <= len(test_report) -1):
                                diagnosis_tokens = test_report[diagnosis_line_num].lower()
                                if (any([x in diagnosis_tokens for x in correct_word_set])) and ('<prev_nodule>' not in diagnosis_tokens) and ('<bad_nodule>' not in diagnosis_tokens):
                                    f.write('Diagnosis tokens: {}\n'.format( diagnosis_tokens))
                                    f.write('Found the diagnosis in the next line! Record the laterality and location: {}/{}\n'.format( laterality_flag, location))
                                    find_diagnosis_flag = True
                                    # mark this line as the diagnosis for the previous nodule
#                                     diagnosis_tokens = ['<prev_nodule>' if x=='<first_view>' else x for x in diagnosis_tokens]
                                    test_report[diagnosis_line_num] = test_report[diagnosis_line_num].replace('<first_view>', '<prev_nodule>')
                                    locs.append(location)
                                    lats.append(laterality_flag)
                                    nums.append(nodule_number)
                                elif (any([x in diagnosis_tokens for x in opposite_word_set])) and ('<first_view>' in diagnosis_tokens):
                                    f.write('This is a diagnosis line, but not the correct diagnosis\n')
                                    test_report[diagnosis_line_num] = test_report[diagnosis_line_num].replace('<first_view>', '<bad_nodule>')
                                    break
                                elif ('<prev_nodule>' in diagnosis_tokens) or ('<bad_nodule>' in diagnosis_tokens):
                                    f.write('Found a diagnosis line, but this is for the previous nodule.\n')
                                    diagnosis_line_num += 1
                                else:
                                    f.write('Not a diagnosis line. Move to the next line.\n')
                                    diagnosis_line_num += 1
                        else:
                            f.write("Diagnosis checked! Failed to pass the diagnosis check: {}\n".format( laterality_flag))
                # find the laterality, but no diagnosis
                else: 
                    f.write('No diagnosis was found! Continue to the next line.\n')
    return locs, lats, nums

def get_site(test_report):
    """
        Extract site from one pathology report

        Parameters:
            test_report: input report in free text

        Returns:
            cur_site: site token
    """
    cur_site = "none"
    tokenizer = RegexpTokenizer(r'\w+')
    for line in test_report:
        if 'Ordering Location' in line:
            tokenized = np.array(tokenizer.tokenize(line.lower()))
            location_idx = int(np.where(tokenized == 'location')[0])
            received_idx = int(np.where(tokenized == 'received')[0])
            cur_site = tokenized[(location_idx + 1):received_idx]
            cur_site = ' '.join(cur_site)
        

    return cur_site
        
def get_all_sites(test_reports):
    """
        Get sites for all reports

        Parameters:
            test_reports: list of reports

        Returns:
            sites: list of sites
    """
    sites = []
    for test_report in test_reports: 
        test_report = test_report.split('\r\n')
        site = get_sites(test_report)
        sites.append(site)
    return sites
        
def get_num_nodules(res_lat):
    """
        Return number of nodules given by extracted laterality tokens

        Parameters:
            res_lat: laterality tokens

        Returns:
            counts: number of nodules
    """
    counts = 0
    for _ in res_lat:
        if any([_ in ['left', 'right', 'isthmus']]):
            counts += 1
    return counts

def get_uniq_lat(all_lats_):
    """
        Get unique laterality from all lateralities. E.g. "right left left" returns 'right'
        
        Parameters:
            all_lats_: list of all laterality tokens

        Returns:
            res: list of unique laterality tokens
    """

    res = []
    for _input in all_lats_:
        # print('====================')
        _input = np.array(_input)
        # remove none first
        none_idx = np.argwhere(_input == 'none').flatten()
        input_clean = np.delete(_input, none_idx)
        if len(input_clean) != 0:
            label, counts = np.unique(input_clean, return_counts=True)
            uniq_idx = np.argwhere(counts == 1).flatten()
            if len(uniq_idx)!=0:
                final_res = label[uniq_idx]
            else:
                final_res = 'n/a'
        else:
            final_res = 'n/a'
        res.append(final_res)
    return res

def return_laterality_numnod(report_df, selected_mrns, laterality_word_set, location_word_set, correct_word_set, opposite_word_set, log_file=None):
    """
        Extract nodule information for all reports and collate information into one dataframe

        Parameters:
            report_df: dataframe that contains pathology reports
            selected_mrns: patients included in the study
            laterality_word_set: default laterality tokens
            location_word_set: default location tokens
            correct_word_set: diagnosis tokens for the selected nodule category
            opposite_word_set: diagnosis tokens opposite to the selected nodule category
            log_file: log file
        
        Returns:
            no_pediatric_df: nodule information for non-pediatric patients
            freq_table: frequency table of number of nodules


    """

    f = open(log_file, 'w')
    test_reports = report_df['Report Text'][report_df['MRN'].astype(str).isin(selected_mrns)]
    all_mrn = list(report_df['MRN'][report_df['MRN'].astype(str).isin(selected_mrns)].astype('str'))
    all_encounter = list(report_df['Encounter Identifier'][report_df['MRN'].astype(str).isin(selected_mrns)].astype('str'))
    all_mrn_encounter = [x + '-' + y for x, y in zip(all_mrn, all_encounter)]
    all_dates = list(report_df['Report Date'][report_df['MRN'].astype(str).isin(selected_mrns)])
    all_locs, all_lats, all_sites, all_nums = [], [], [], []
    all_locs_uniq, all_lats_uniq, all_sites_uniq, all_nums_uniq = [], [], [], []
    for _, test_report in enumerate(test_reports):
        # print('='*40, 'Report Number: ', _, all_mrn[_], '='*40)
        f.write('============================================== \n')
        f.write('MRN: {}\n'.format(all_mrn[_]))
        test_report = test_report.replace('_x000D_', '')
        test_report = test_report.split('\n')
        locs, lats, nums = find_laterality(test_report, laterality_word_set, location_word_set, correct_word_set, opposite_word_set, f)
        
        site = get_site(test_report)
        all_sites.append(site)
        all_lats.append(lats)
        all_locs.append(locs)
        all_nums.append(nums)
    
    uniq_lats = get_uniq_lat(all_lats)
    num_nodules = [get_num_nodules(x) for x in all_lats]
    df = pd.DataFrame({
    'MRN': all_mrn,
    'MRN_Enc': all_mrn_encounter,
    'ReportDate': all_dates,
    'ReportSite': all_sites,
    'laterality': all_lats,
    'location': all_locs, 
    'nodule_number': all_nums,
    'num_nodules': num_nodules,
    'uniq_lat': uniq_lats
    })
    f.close()
    c = df.num_nodules.value_counts(sort=False)
    p = df.num_nodules.value_counts(normalize=True, sort=False)
    freq_table = pd.concat([c, p], axis=1, keys=['counts', '%'])
    # exclude pediatric
    # no_pediatric_df = df.copy()
    # no_pediatric_df = no_pediatric_df[~no_pediatric_df.ReportSite.str.contains('child')]
    # print("number of patients from input/exclude pediatric patients: {}/{}".format(len(np.unique(df.MRN)), len(np.unique(no_pediatric_df.MRN))))
    return df, freq_table

def get_lat_res(df):
    """
        Expand previous nodule information dataframe so that each row records one nodule

        Parameters:
            df: dataframe from "return_laterality_numnod"
        
        Returns:
            res: output dataframe

    """
    N = len(df)
    tot_num_nodules = np.sum(df.num_nodules)
    mrns, dates, lats, locs, nums = np.array([], dtype=str), np.array([], dtype=str), np.array([], dtype=str), np.array([], dtype=str), np.array([], dtype=str)
    uniqs, sites = np.array([], dtype=str), np.array([], dtype=str)
    nodule_ids = np.array([], dtype=str)
    word_tokenizer = RegexpTokenizer(r'\#\d|\w+')
    for idx in tqdm(range(N)):
        # print('='*40, 'Report Number: ', idx, '='*40)
        # duplicate mrns
        cur_mrn = df.MRN.iloc[idx]
        cur_date = df.MRN_Enc.iloc[idx]
        cur_num_nodules = df.num_nodules.iloc[idx]
        cur_site = df.ReportSite.iloc[idx]
        cur_uniq = df.uniq_lat.iloc[idx]
        
        if cur_num_nodules == 0:
            mrns = np.concatenate([mrns, [cur_mrn]])
            dates = np.concatenate([dates, [cur_date]])
            sites = np.concatenate([sites, [cur_site]])
            lats = np.concatenate([lats, ['none']])
            locs = np.concatenate([locs, ['none']])
            nums = np.concatenate([nums, ['none']])
            nodule_ids = np.concatenate([nodule_ids, ['none']])
            uniqs = np.concatenate([uniqs, ['none']])
        else:
            # mrns
            mrn = np.repeat(cur_mrn, cur_num_nodules)
            mrns = np.concatenate([mrns, mrn])
            # dates
            date = np.repeat(cur_date, cur_num_nodules)
            dates = np.concatenate([dates, date])
            # sites
            site = np.repeat(cur_site, cur_num_nodules)
            sites = np.concatenate([sites, site])
            # nodule id
            nodule_id = np.arange(cur_num_nodules) 
            nodule_ids = np.concatenate([nodule_ids, nodule_id])

            # laterality
            cur_laterality = df.laterality.iloc[idx]
            cur_laterality = word_tokenizer.tokenize(cur_laterality)
            if pd.isnull(cur_uniq) is False:
                cur_uniq = word_tokenizer.tokenize(cur_uniq)[0]
            else:
                cur_uniq = 'n/a'
            uniq_lat, lat = np.array([], dtype=str), np.array([], dtype=str)
            for _ in range(cur_num_nodules):
                cur_lat = cur_laterality[_]
                if len(cur_lat) == 0:
                    _ = None
                if cur_lat == cur_uniq:
                    uniq_lat = np.append(uniq_lat, 'Yes')
                else:
                    uniq_lat = np.append(uniq_lat, 'No')
                lat = np.append(lat, cur_lat)
            uniqs = np.append(uniqs, uniq_lat)
            lats = np.concatenate([lats, lat])
            # location
            cur_location = df.location.iloc[idx].strip("[]").split(',')
            loc = np.array([], dtype=str)
            for _ in range(cur_num_nodules):
                cur_loc = cur_location[_]
                if len(cur_loc) == 0:
                    _ = None
                loc = np.append(loc, cur_loc)
                # print(loc)
            locs = np.concatenate([locs, loc])
            # nodule number 
            cur_nodule_number = df.nodule_number.iloc[idx]
            cur_nodule_number = word_tokenizer.tokenize(cur_nodule_number)
            num = np.array([], dtype=str)
            for _ in range(cur_num_nodules):
                cur_num = str(cur_nodule_number[_])
                if cur_num is None:
                    _ = None
                num = np.append(num, cur_num)
            nums = np.concatenate([nums, num])
    locs = [word_tokenizer.tokenize(x) for x in locs]
    # save in a dataframe
    res = pd.DataFrame({
        'MRN': mrns,
        "MRN_Enc": dates,
        "ReportSite": sites,
        'NoduleId_lsa': nodule_ids,
        'laterality_lsa': lats,
        'location_lsa': locs,
        'NoduleNumber_lsa': nums,
        'is_unique': uniqs
    })
    
    return res
