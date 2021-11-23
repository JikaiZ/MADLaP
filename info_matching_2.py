################################################################
# Functions to help produce yields from M5
################################################################


import datetime
import collections
from tqdm import tqdm
import re
import string

from utils import *
from datetime import datetime, timedelta
from nltk.tokenize import RegexpTokenizer, word_tokenize

def get_ocr_image_res(tmp_ocr_df):
    assert len(tmp_ocr_df) == 2
    view1, view2 = tmp_ocr_df.ocr_view.iloc[0], tmp_ocr_df.ocr_view.iloc[1]
    lat1, lat2 = tmp_ocr_df.ocr_laterality.iloc[0], tmp_ocr_df.ocr_laterality.iloc[1]
    loc1, loc2 = tmp_ocr_df.ocr_location.iloc[0], tmp_ocr_df.ocr_location.iloc[1]
    num1, num2 = tmp_ocr_df.ocr_nums.iloc[0], tmp_ocr_df.ocr_nums.iloc[1]
    res1 = view1 + '-' + lat1 + '-' + loc1 + '-' + num1
    res2 = view2 + '-' + lat2 + '-' + loc2 + '-' + num2
    return res1, res2

def extract_float_from_string(measure):
    measure_seq = np.zeros([len(measure)], dtype=np.float32)
    for _, item in enumerate(measure):
        res_string = ' '.join(item)
        res_string = res_string.strip('[]')
        float_number = round(float(re.findall(r"\d+(?:\.\d*)?|\d+", ' '.join(item))[0]), 1)
        if float_number > 100:
            float_number = round(float_number * 0.01, 1)
        measure_seq[_] = float_number
    return measure_seq

def match_image_by_measurements(rad_rep_result, candidate_ocr_result, ocr_text_df, f):
    """
        Get yield for M5 based on measurements
        params:
            rad_rep_result: radreport_measure_.csv
            candidate_ocr_result: ocr_outputs_measurements_postprocessed_.csv
            ocr_text_df: ocr_outputs_postprocessed_.csv
        return:
            df: matching information
    """
    studies, mrnencs, res, measures = [], [], [], []
    ocr_results1, ocr_results2 = [], []
    decisions = []
    for _ in range(len(rad_rep_result)):
        f.write('================================= \n')
        ocr_res1, ocr_res2 = 'n/a', 'n/a'
        mrnenc = '-'.join(rad_rep_result.iloc[_].MRN_ENC_NODID_LAT.split('-')[:2])
        mrnencs.append(mrnenc)
        measure = rad_rep_result.iloc[_].radreport_output
        measures.append(measure)
        nodinfo = rad_rep_result.iloc[_].MRN_ENC_NODID_LAT
        mrn = rad_rep_result.iloc[_].MRN_ENC_NODID_LAT.split('-')[0]
        tmp_ocr_result = candidate_ocr_result[candidate_ocr_result.NoduleInfo == nodinfo]
        tmp_ocr_result = tmp_ocr_result[tmp_ocr_result.Measure!= 'none']
        if len(tmp_ocr_result) == 0:
            studies.append('n/a')
            decisions.append('FAIL')
            res.append('no OCR results')
            ocr_results1.append(ocr_res1)
            ocr_results2.append(ocr_res2)
            continue
        
        study = tmp_ocr_result.Study.iloc[0]
        studies.append(study)
        ocr_measure = tmp_ocr_result.Measure.tolist()

        
        rep_measure = rad_rep_result.iloc[_].radreport_output
        f.write('nodid: {}, ocr measure: {}, rep measure: {} \n'.format(nodinfo, ocr_measure, rep_measure))
        if 'multiple nodules' in rep_measure:
            decisions.append('FAIL')
            res.append('multiple nodules in rad report')
            ocr_results1.append(ocr_res1)
            ocr_results2.append(ocr_res2)
            continue
        if 'multiple extracted' in rep_measure:
            decisions.append('FAIL')
            res.append('multiple extracted measures')
            ocr_results1.append(ocr_res1)
            ocr_results2.append(ocr_res2)
            continue
        if 'no passed measures' in rep_measure:
            decisions.append('FAIL')
            res.append('no passed measures')
            ocr_results1.append(ocr_res1)
            ocr_results2.append(ocr_res2)
            continue
        if 'bad format' in rep_measure:
            decisions.append('FAIL')
            res.append('bad format')
            ocr_results1.append(ocr_res1)
            ocr_results2.append(ocr_res2)
            continue
        measure0 = re.findall(r"(\d+(?:\.\d*)?)|(\d+(?:\.\d*)?)|\d+", rep_measure)
        rep_measure_floats = extract_float_from_string(measure0)
    
        selected_image_idx = []
        
        for rep_float in rep_measure_floats:
            find_measure = False
            for (image_idx, me) in enumerate(ocr_measure):
                me = ' '.join(me.strip('[]').split(' '))
                rep_float_str = str(rep_float)
                if (rep_float_str in me) and (find_measure == False):
                    sel_idx_ = image_idx
                    selected_image_idx.append(sel_idx_)
                    find_measure = True
        f.write('sel idx: {} \n'.format(selected_image_idx))
        selected_image_uidx = np.unique(selected_image_idx)
        if len(selected_image_uidx) == 2:
            all_imgs = tmp_ocr_result.Image.tolist()
            sel_images_arr = tmp_ocr_result.Image.iloc[selected_image_uidx].values
            test_ocr_df = ocr_text_df[ocr_text_df.Image.isin(sel_images_arr)]
            ocr_res1, ocr_res2 = get_ocr_image_res(test_ocr_df)
            sel_images = ",".join(sel_images_arr)
            sel_measure = ",".join(tmp_ocr_result.Measure.iloc[selected_image_uidx].values)
            f.write('all_imgs: {}\n'.format(all_imgs))
            f.write('measures on image: {} \n'.format(ocr_measure))
            f.write('sel images: {}\n'.format(sel_images))
            f.write('txt ocr1: {}, ocr2: {}'.format(ocr_res1, ocr_res2))
            f.write('sel measure: {}\n'.format(sel_measure))
            ocr_results1.append(ocr_res1)
            ocr_results2.append(ocr_res2)
            decisions.append('yield2')
            res.append(sel_images)
        else:
            res.append('# of image selected: {}'.format(len(selected_image_uidx)))
            ocr_results1.append(ocr_res1)
            ocr_results2.append(ocr_res2)
            decisions.append('FAIL')
    df = pd.DataFrame({
        'MRN_ENC_NODID_LAT': rad_rep_result.MRN_ENC_NODID_LAT.values,
        'MRN_Enc': mrnencs,
        'Study': studies,
        'Decisions': decisions,
        'Image': res,
        'RadReportMeasure': measures,
        'ocr_res_img1': ocr_results1,
        'ocr_res_img2': ocr_results2
    })
    return df