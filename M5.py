#################################################################
# Functions for M5 - measurement matching
#################################################################

import pandas as pd
import numpy as np
import os
import datetime
import json
import collections
from tqdm import tqdm
import nltk
import re
import string

from utils import *
from datetime import datetime, timedelta

from nltk.tokenize import RegexpTokenizer, word_tokenize

## OCR

def get_measure_multi_outputs(ocr_df, f):
    idx = ['OCR' in x for x in ocr_df.columns]
    columns = pd.Index(np.array(ocr_df.columns)[idx])
    ocr_outputs = ocr_df[columns].values
    nrow, ncol = ocr_outputs.shape[0], ocr_outputs.shape[1]
    res_measure = []
    for rownum in range(nrow):
        f.write('=== {} ==== \n'.format(ocr_df.MRN.values[rownum]))
        f.write('Img: {} \n'.format(ocr_df.Image.values[rownum]))
        find_measure = False
        _measure_seq = 'none'
        for rowcol in range(ncol):
            cur_outputs = ocr_outputs[rownum, rowcol]
            if pd.isnull(cur_outputs) == True:
                continue
            cur_outputs = cur_outputs.translate(str.maketrans('', '', '!"#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~'))
            f.write("Output {} Input String: {} \n".format(rowcol+1, cur_outputs))
            measure = re.findall(r"(\d+(?:\.\d*)?)(c|cm|mm|em|cem|crm)|(\d+(?:\.\d*)?)\s(c|cm|mm|em|cem|crm)", cur_outputs)
            if len(measure) == 0:
                measure_seq = 'none'
            else:
                measure_seq = np.zeros([len(measure)], dtype=np.float32)
                for _, item in enumerate(measure):
                    res_string = ' '.join(item)
                    float_number = round(float(re.findall(r"\d+(?:\.\d*)?|\d+", ' '.join(item))[0]), 1)
                    if float_number > 15:
                        float_number = 0
                    measure_seq[_] = float_number
            if (_measure_seq != 'none') and (len(measure_seq) > len(_measure_seq)):
                find_measure = False
            if (find_measure == False) and (measure_seq != 'none') and (0 not in measure_seq):
                _measure_seq = measure_seq
                find_measure = True
        if _measure_seq != 'none':
            res_measure.append(_measure_seq)
        else:
            res_measure.append('none')
    return res_measure

def get_measure_df(yield_df, ocr_df, lat_df, f):
    noduleinfos, images, measures = [], [], []
    res_df = pd.DataFrame()
    for _, row in yield_df.iterrows():
        mrnenc, noduleid = row.MRN_Enc, row.NoduleId
        study, image = row.Study, row.Image
        image = image.split(',')
        
        mrn = mrnenc.split('-')[0]
        f.write('======================= \n')
        # get laterality
        tmp_lat_df = lat_df[lat_df.MRN_Enc == mrnenc]
        f.write('Number of nodules: {} \n'.format( len(tmp_lat_df)))

        
        tmp_lat_df = tmp_lat_df[tmp_lat_df.NoduleId_lsa == noduleid]

        lat = tmp_lat_df.laterality_lsa.iloc[0]
        if 'none' in image:
            ocr_image = ['n/a']
            tmp_study = ['n/a']
            measure = ['none']
        elif len(image) == 2:
            ocr_image = image
            tmp_study = study
            measure = ['yield1', 'yield1']
            f.write('yield 1 \n')
        elif len(image) > 2:
            # get images
            tmp_ocr_df = ocr_df[ocr_df.MRN == mrn]
            ocr_image = image
            tmp_study = study
            tmp_ocr_df = tmp_ocr_df[tmp_ocr_df.Image.isin(image)]
            # get measurements
            measure = get_measure_multi_outputs(tmp_ocr_df, f)
        f.write('Extracted Measure: {}\n'.format( measure))
        noduleinfo = mrnenc + '-' + str(noduleid) + '-' + lat

        tmp_noduleinfo = np.repeat(noduleinfo, len(ocr_image))
        
        tmp_df = pd.DataFrame({
            'NoduleInfo': tmp_noduleinfo,
            'Study': tmp_study,
            'Image': ocr_image,
            'Measure': measure
        })
        res_df = pd.concat([res_df, tmp_df])
    return res_df


## Collating rad reports

def opposite_lat(lat):
    if lat == 'right':
        res = ['left', 'isthmus']
    elif lat == 'left':
        res = ['right', 'isthmus']
    elif lat == 'isthmus':
        res = ['left', 'right']
    else:
        res = 'none'
    return res

def get_accn_from_qido(qido_file):
    with open(qido_file, 'r') as f:
        contents = f.readlines()

    mrns, accns, stdess = [], [], []
    for line in contents:
        if line[0] == '#':
            continue
        else:
            status,rpmrn,dterg,mrn,accn,stdte,pnme,dob,sex,mod,stdes,imgnm,stuid,wurl = line.split('|')
            stdes_ = stdes.replace(" ", "-")
            study_dir = "_".join([stdes_, stdte])
            mrns.append(mrn)
            accns.append(accn)
            stdess.append(study_dir)
    df = pd.DataFrame({
        'MRN': mrns,
        'ACCN': accns,
        'Study':  stdess

    })
    return df

def get_rad_report_by_accn(qido_df, root_src, report_df):
    res_df = pd.DataFrame()
    for (dirpath, dirnames, filenames) in os.walk(root_src):
        for dirname in dirnames:
            if ('US-SOFT-TISSUE-HEAD-AND-NECK' in dirname):
                study = '_'.join(dirname.split('_')[0:2])
                study_folder = os.path.join(dirpath, dirname)
                study_mrn = study_folder.split('/')[-2]
                sel_df = qido_df[qido_df.MRN == study_mrn]
                sel_df = sel_df[sel_df.Study == study]
                sel_accn = sel_df.ACCN.values[0]
                sel_accn_ = sel_accn.replace('RH', "")
                sel_accn_ = sel_accn_.replace('B', "")

                sel_report_df = report_df[report_df['MRN'].astype(str) == study_mrn]
                sel_report_df = sel_report_df[sel_report_df['Accession Number'].str.contains(sel_accn_, na=False)]
                sel_report = sel_report_df['Report Text']
                if len(sel_report)>0:
                    sel_report = sel_report.iloc[0]
                    res_df = pd.concat([res_df, sel_report_df])
                    
    return res_df


def collate_rad_report(match_yield1_input, qido_file, report_df, root_src):
    """Prepare spreadsheet containing rad reports with id as nodid"""
    qido_df = get_accn_from_qido(qido_file)
    rad_report_df = get_rad_report_by_accn(qido_df, root_src,report_df)
    match_yield1 = match_yield1_input[match_yield1_input.Decisions.str.contains('yield2')]
    # mrnenc = match_yield1[match_yield1.Decisions.str.contains('yield2')].MRN_Enc.tolist()
    mrns, nods, reports = [], [], []
    for _, row in match_yield1.iterrows():
        decision, study, mrnenc = row.Decisions, row.Study, row.MRN_Enc
        fin_decision = row.DecisionFin
        if (decision == 'yield2') and ('FAIL' not in fin_decision):
            nodid = row.NoduleId
            lat = row.LSA_res.split('-')[0]
            nodinfo = row.MRN_ENC_NODID_LAT
            mrn = str(mrnenc.split('-')[0])
        
            test_report_df = rad_report_df[rad_report_df['MRN'].astype(str) == mrn]
            # ENDO 
            if len(test_report_df) == 0:
                print(mrnenc, ':', study)
                continue
            else:
                report_text = test_report_df['Report Text'].iloc[0]
                mrns.append(mrnenc)
                nods.append(nodinfo)
                reports.append(report_text)
        else:
            continue
            
    df = pd.DataFrame({
        'MRN_Enc': mrns,
        'NoduleInfo': nods,
        'ReportText': reports
    })
    return df

## Get measurements from rad reports
def get_floats(input_string):
    return re.findall(r"[-+]?\d*\.*\d+", input_string)

def find_findings_section(ss):
    """Return findings paragraph
    Input: 
        test_text: radiology text
    Output: 
        findings_section: Section titled Findings
    """
    find_findings, find_impression = False, False
    sentence_line_num, impression_line_num = 0, 0
    for (_, header) in enumerate(ss):
        
        if ("findings:" in header.lower().split(' ')[:3]) and (find_findings == False):       
            sentence_line_num = _
            find_findings = True
        if ("impression:" in header.lower().split(' ')[:3]) and (find_impression == False):        
            impression_line_num = _
            find_impression = True
    findings_section = ss[sentence_line_num+1:impression_line_num]
    findings_section = ('\n').join(findings_section)
    return findings_section


def collate_paragraphs(input_string):
    paragraphs = []
    para_start, para_end = 0, 0
    find_start, find_end = False, False
    _string = input_string.split('\n')
    for _, line in enumerate(_string):
        if (find_start == False) and (len(line) != 0):
            para_start = _
            find_start = True
        if (find_start == True) and (find_end == False) and (len(line) == 0):
            para_end = _
            find_end = True
        if find_end == True:
            paragraph = ' '.join(_string[para_start: para_end])
            find_start, find_end = False, False
            paragraphs.append(paragraph)
    return paragraphs

def get_section(lat, findings,f):
#     lines = findings.split('\n')
    lines = findings
    tot_lines = len(lines)
    linenum_start, linenum_end = 0, 0
    find_start, find_end = False, False
    oppo_lat = opposite_lat(lat)
    for idx, line in enumerate(lines):
        if (find_start == False) and (lat in line.lower().split(' ')):
            linenum_start = idx
            find_start = True
        if (find_start == True) and (find_end == False) and (any([x in oppo_lat for x in line.lower().split(' ')])):
            linenum_end = idx
            find_end = True
        if (find_start == True) and (find_end == False) and (idx == (tot_lines - 1)):
            linenum_end = idx
            find_end = True
        if (find_end == True) and (find_start == True):
            break
    if (find_end == False) or (find_start == False):
        linenum_start, linenum_end = 0, 0
    if linenum_end == linenum_start:
        res = lines[linenum_start]
    else:
        res = '\n'.join(lines[linenum_start:linenum_end])
    num_nodules = len(re.findall(r"Nodule\s\d+|Nodule\s\#\d+", res))
    return res, num_nodules

def get_nodule_section(section):
    lines = section.split('\n')
    tot_lines = len(lines)
    linenum_start = 0
    find_start, find_end = False, False
    for idx, line in enumerate(lines):
        test_line = line.translate(str.maketrans('', '', '!"#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~'))
        if (find_start == False) and ('Nodule' in test_line.split(' ')[:2]):
            linenum_start = idx
            find_start = True
        if (find_start == True) and (find_end == False) and ('lymph' in test_line.lower()[:10]):
            linenum_end = idx
            find_end = True
            break
    if (find_start == False):
        linenum_start, linenum_end = 0, 0
        res = lines[0:0]
    elif find_end == True:
        res = '\n'.join(lines[linenum_start:linenum_end])
    elif find_end == False:
        res = '\n'.join(lines[linenum_start:])
    return res

def find_nodules_paragraphs(nodule_section, lat):
    lines = nodule_section.split('\n')
    res = []
    for line in lines:
        if lat in line.lower():
            res.append(line)
    return res

def get_measurements_from_section(section):
    measures = []
    if isinstance(section, list):
        section = section[0]
    p = re.compile(r"(\d+(?:\.\d*)?)\s*x\s*(\d+(?:\.\d*)?)\s*x\s*(\d+(?:\.\d*)?)\s*(cm|mm)")
    for m in p.finditer(section):
        measure = m.group()
        measures.append(measure)
    return measures

def mask_measurement(input_section, measures):
    res = input_section
    if isinstance(input_section, list):
        res = input_section[0]
    measure_names = {}
    assert len(measures) > 0
    for idx, measure in enumerate(measures):
        measure_name = 'measure' + str(idx)
        measure_names[measure_name] = measure
        res = res.replace(measure, ' ' + measure_name)
    return res, measure_names

def check_measurement_for_nodule(masked, m_names):
    sents = nltk.sent_tokenize(masked)

    res = []
    for sent in sents:
        # find the measurement
        sent_tokenized = nltk.word_tokenize(sent)
        if any([x in m_names for x in sent_tokenized]):
            true_idx = [i for i,x in enumerate([x in sent_tokenized for x in m_names]) if x]
            cur_measure = m_names[true_idx[0]]
            cur_measure_loc = [i for i,x in enumerate([x in m_names for x in sent_tokenized]) if x][0]
            test_sent = sent.translate(str.maketrans('', '', '!"#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~'))
            if ('lobe measure' not in test_sent.lower()) and ('gland measure' not in test_sent.lower()) and ('previous' not in sent_tokenized[cur_measure_loc-1:cur_measure_loc]):
                res.append(cur_measure)
    return res


def get_measurements_from_radreport(candidate_df, log_file=None):
    res = []
    f = open(log_file, 'w')
    for _, row in candidate_df.iterrows():
        nodinfo = row.NoduleInfo
        mrn, enc, noduleid, lat = nodinfo.split('-')
        mrnenc = mrn + '-' + enc
        test_string = row.ReportText.split('\n')
        f.write('{} ==========: \n'.format(nodinfo))
        f.write("Report: \n")
        f.write(row.ReportText)
        findings = find_findings_section(test_string)
        if len(findings) == 0:
            res.append('no findings')
            continue
        f.write("------------------------- \n")
        f.write("Extracted Findings: \n")
        f.write(findings)
        # get paragraphs
        paragraphs = collate_paragraphs(findings)
        f.write('number of paragraphs: {}\n'.format(len(paragraphs)))
        first_line = paragraphs[0].lower()
        
        if ('isthmus' in first_line) and ('right' in first_line) and ('left' in first_line):
            f.write('Alternative pattern for the first paragraph: {} \n'.format( first_line))
            nodule_section = get_nodule_section('\n'.join(paragraphs))
            if len(nodule_section) == 0:
                res.append('bad format')
                f.write('bad format for alternative pattern \n')
                continue
            nodule_section = find_nodules_paragraphs(nodule_section, lat)
            if len(nodule_section) > 1:
                res.append('multiple nodules')
                f.write('multiple nodules in rad report detected, pass! \n')
                continue
            elif len(nodule_section) == 0:
                res.append('bad format')
                f.write('bad format for alternative pattern \n')
                continue
            else:
                sel_section = nodule_section
        else:
            section, num_nodules = get_section(lat, paragraphs, f)
            if len(section) == 0:
                res.append('no section')
                continue
            if num_nodules > 1:
                f.write('multiple nodules in rad report detected, pass! \n')
                res.append('multiple nodules')
                continue
            nodule_section = get_nodule_section(section)
            if len(nodule_section) > 0:
                sel_section = nodule_section
            else:
                sel_section = section
                
        f.write("------------------------- \n")
        f.write("Extracted Section: \n")
        f.write(str(sel_section)+ '\n')
        f.write("************************** \n")
        measures = get_measurements_from_section(sel_section)
        if len(measures) == 0:
            res.append('no measures')
            continue
        masked, m_names_dict = mask_measurement(sel_section, measures)
        m_names = list(m_names_dict.keys())
        checked_measurement_names = check_measurement_for_nodule(masked, m_names)
        if len(checked_measurement_names) == 0:
            res.append('no passed measures')
            continue
        elif len(checked_measurement_names) == 2:
            # cm mm
            # small
            checked_measurements_dict = {x:m_names_dict[x] for x in checked_measurement_names}
            checked_measurements = np.array(list(checked_measurements_dict.values()))
            checked_measurements = checked_measurements[np.argmin([max(get_floats(x)) for x in checked_measurements])]
            res.append(checked_measurements)
        elif len(checked_measurement_names) > 2:
            f.write("extracted multiple measurements: {} \n".format( ",".join([m_names_dict[x] for x in checked_measurement_names])))
            res.append('multiple extracted measures')
            continue
        else:
            checked_measurements = m_names_dict[checked_measurement_names[0]]
            res.append(checked_measurements)
        f.write("------------------------- \n")
        f.write("Extracted Measurements: \n")
        f.write(checked_measurements + "\n")
    f.close()
        
    res_df = pd.DataFrame({
    'MRN_ENC_NODID_LAT': candidate_df.NoduleInfo.values,
    'radreport_output': res
    })
    return res_df

def get_floats(input_string):
    return re.findall(r"[-+]?\d*\.*\d+", input_string)
