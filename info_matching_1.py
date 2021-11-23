################################################################
# Functions to help produce yields from M3-M4
################################################################

import pandas as pd
import numpy as np
import os
import datetime
import json
import collections
from tqdm import tqdm

import re

from utils import *
from nlp_utils import *

from nltk.tokenize import RegexpTokenizer, word_tokenize

##################################################################

# helper functions to match nodule info from texts and from images

##################################################################
def get_biopsy_studies(lat_df, study_df):
    studies = []
    for idx in range(len(lat_df)):
        cur_mrnenc = lat_df.MRN_Enc.iloc[idx]
        cur_study_dir = study_df[study_df.MRN_Enc.str.match(cur_mrnenc)].BiopsyStudyDir.values[0]
        if pd.isnull(cur_study_dir) is True:
            cur_study = 'none'
        else:
            cur_study = cur_study_dir.split('/')[-1]
        studies.append(cur_study)
    return studies

def get_diagnostic_studies(lat_df, study_df):
    studies = []
    for idx in range(len(lat_df)):
        cur_mrnenc = lat_df.MRN_Enc.iloc[idx]
        cur_study_dir = study_df[study_df.MRN_Enc.str.match(cur_mrnenc)].DiagnosticStudyDir.values[0]
        if pd.isnull(cur_study_dir) is True:
            cur_study = 'none'
        else:
            cur_study = cur_study_dir.split('/')[-1]
        studies.append(cur_study)
    return studies

def get_endo_studies(lat_df, study_df):
    studies = []
    for idx in range(len(lat_df)):
        cur_mrnenc = lat_df.MRN_Enc.iloc[idx]
        cur_study_dir = study_df[study_df.MRN_Enc.str.match(cur_mrnenc)].PathologyStudyDir.values[0]
        if pd.isnull(cur_study_dir) is True:
            cur_study = 'none'
        else:
            cur_study = cur_study_dir.split('/')[-1]
        studies.append(cur_study)
    return studies

def get_ocr_image_res(tmp_ocr_df):
    assert len(tmp_ocr_df) == 2
    view1, view2 = tmp_ocr_df.ocr_view.iloc[0], tmp_ocr_df.ocr_view.iloc[1]
    lat1, lat2 = tmp_ocr_df.ocr_laterality.iloc[0], tmp_ocr_df.ocr_laterality.iloc[1]
    loc1, loc2 = tmp_ocr_df.ocr_location.iloc[0], tmp_ocr_df.ocr_location.iloc[1]
    num1, num2 = tmp_ocr_df.ocr_nums.iloc[0], tmp_ocr_df.ocr_nums.iloc[1]
    res1 = view1 + '-' + lat1 + '-' + loc1 + '-' + num1
    res2 = view2 + '-' + lat2 + '-' + loc2 + '-' + num2
    return res1, res2

def view_check(view1, view2):
    """check if views are the a pair if ocr returns values"""
    word_tokenizer = RegexpTokenizer(r'\#\d|\w+')
    res = 'unknown'
    if (view1 == 'none') or (view2 == 'none'):
        res = "unknown"
    elif (view1 != 'none') and (view2 != 'none'):
        view1_ = word_tokenizer.tokenize(view1)
        view2_ = word_tokenizer.tokenize(view2)
        if (len(view1_) > 1) or (len(view2_) > 1):
            res = 'multiple views'
        elif view1_ == view2_:
            res = 'same view'
        elif view1_ != view2_:
            res = "diff view"
    return res


def match_ocr_lat(test_ocr_df, tmp_lat_df, cur_study, num_nodules_lsa, f):
    """
        get matching info for one nodule
        param:
            test_ocr_df: ocr_outputs_postprocessed_.csv
            tmp_lat_df: lat_res_df_.csv
            cur_study, num_nodules_lsa: current study, number of nodules from laterality searching
        return:
            res: array of info matching related information 
            ocr_res1, ocr_res2: results for image 1/2, or none for multiple/no images
            decision: various decisions of the yield, such as cd: caliper detection, lat: cd + ocr, FAIL, etc..
    
    """
    word_tokenizer = RegexpTokenizer(r'\#\d|\w+')
    find_img = False
    ####################################################
    # Basic Nodule Info
    ####################################################
    # MRNENC
    cur_mrnenc = tmp_lat_df.MRN_Enc
    # SITE
    cur_site = tmp_lat_df.ReportSite
    # LATERALITY
    cur_lat = tmp_lat_df.laterality_lsa
    # NODULE #
    cur_num = replace_num(tmp_lat_df.NoduleNumber_lsa)
    # LOCATION + REV LOC
    cur_loc_ = tmp_lat_df.location_lsa
    cur_loc_ = cur_loc_.strip('[]')
    cur_loc_ = word_tokenizer.tokenize(cur_loc_)
    cur_loc = '/'.join(cur_loc_)
    cur_loc_rev = '/'.join(cur_loc_[::-1])
    # COMBINE 
    lsa_res = cur_lat + '-' + cur_loc + '-' + cur_num
    
    matched_idx = np.where(test_ocr_df.Study.values == cur_study)[0]
    test_ocr_df = test_ocr_df.iloc[matched_idx]
    res = 'none'
    ocr_res1, ocr_res2 = 'none', 'none'
    # caliper detection for exactly one extracted nodule
    if len(test_ocr_df) < 2:
        decision = "FAIL: ocr insuffcient images"
        f.write('No ocr images \n')
        find_img = False
    elif (len(test_ocr_df) == 2) and (num_nodules_lsa == 1):
        f.write('---------------------------- \n')
        f.write('check caliper detection outputs \n')
        if len(tmp_lat_df) == 0:
            nid = 'none'
        else:
            nid = tmp_lat_df.NoduleId_lsa
        cur_img = np.array(test_ocr_df.Image)
        cur_img = ",".join(cur_img.tolist())
        
        ocr_res1, ocr_res2 = get_ocr_image_res(test_ocr_df)
        ocr_view1, ocr_view2 = ocr_res1.split('-')[0], ocr_res2.split('-')[0]
        # view checks
        view_check_res = view_check(ocr_view1, ocr_view2)
        f.write("view1: {}, view2: {}, view check: {} \n".format(ocr_view1, ocr_view2, view_check_res))
        if (view_check_res == "same view") or (view_check_res == "multiple views"):
            f.write("fail to pass view check! \n")
            decision = "FAIL: view check"
            find_img = False
        else:
            decision = "cd"
            find_img = True
            f.write("pass view check, matched two images by two caliper detection output images \n")
            res = [cur_img, cur_mrnenc, cur_site, nid, decision, find_img]
            
    else:
        cur_num = replace_num(tmp_lat_df.NoduleNumber_lsa)
        nid = str(tmp_lat_df.NoduleId_lsa)
        # find a nodule with laterality
        if cur_lat == 'none':
            decision = "FAIL: no lat in report/no cd results"
            f.write('No lat \n')
            find_img = False
        else:
            f.write('******************** \n')
            f.write('check laterality match... \n')
            tmp_ocr_df = test_ocr_df[test_ocr_df.ocr_laterality == cur_lat]
            f.write("number of matched images from caliper detection: {} \n".format(len(tmp_ocr_df)))
            if len(tmp_ocr_df) == 2:
                lat_check = True
                cur_img = np.array(tmp_ocr_df.Image)
                cur_img = ",".join(cur_img.tolist())
                _loc1, _loc2 = tmp_ocr_df.ocr_location.iloc[0], tmp_ocr_df.ocr_location.iloc[1]
                _num1, _num2 = tmp_ocr_df.ocr_nums.iloc[0], tmp_ocr_df.ocr_nums.iloc[1]
                ocr_res1, ocr_res2 = get_ocr_image_res(tmp_ocr_df)
                ocr_view1, ocr_view2 = ocr_res1.split('-')[0], ocr_res2.split('-')[0]
                # view checks
                view_check_res = view_check(ocr_view1, ocr_view2)
                f.write("view1: {}, view2: {}, view check: {} \n".format(ocr_view1, ocr_view2, view_check_res))
                if (view_check_res == "same view") or (view_check_res == "multiple views"):
                    lat_check = False
                # loc check if loc from lat is not none
                if cur_loc != 'none':
                    if ((_loc1 != 'none') and (not _loc1 in [cur_loc, cur_loc_rev])) or ((_loc2 != 'none') and (not _loc2 in [cur_loc, cur_loc_rev])):
                        lat_check = False
                # num check if num from lat is not none
                if cur_num != 'none':
                    if ((_num1 != 'none') and (_num1 != cur_num)) or ((_num2 != 'none') and (_num2 != cur_num)):
                        lat_check = False
                if lat_check == True:
                    decision = 'lat'
                    find_img = True
                    f.write("pass view check, matched two images by two caliper detection output images. \n")
                    res = [cur_img, cur_mrnenc, cur_site, nid, decision, find_img]
                else:
                    decision = "FAIL: view/loc/num check"
                    f.write('fail to pass view/loc/num checks. \n')
                    find_img = False
            elif len(tmp_ocr_df) < 2:
                decision = "FAIL: lat insufficient images"
                find_img = False   
            elif len(tmp_ocr_df) > 2:
                cur_loc_ = tmp_lat_df.location_lsa
                cur_loc_ = cur_loc_.strip('[]')
                cur_loc_ = word_tokenizer.tokenize(cur_loc_)
                cur_loc = '/'.join(cur_loc_)
                cur_loc_rev = '/'.join(cur_loc_[::-1])
                # print(cur_loc, cur_loc_rev)
                # first check location
                if cur_loc != 'none':
                    f.write('******************** \n')
                    f.write('check location match... \n')
                    f.write("location from lsa: {} \n".format( cur_loc))
                    tmp_ocr_df_byloc = tmp_ocr_df[(tmp_ocr_df.ocr_location.isin([cur_loc, cur_loc_rev]))]
                    f.write("number of matched images from images by lat and loc: {} \n".format( len(tmp_ocr_df_byloc)))
                    if len(tmp_ocr_df_byloc) < 2:
                        decision = 'FAIL: lat/loc insufficient images'
                        find_img = False
                    elif len(tmp_ocr_df_byloc) == 2:
                        f.write('******************** \n')
                        f.write('check lat + location + num match... \n')
                        _num1, _num2 = tmp_ocr_df_byloc.ocr_nums.iloc[0], tmp_ocr_df_byloc.ocr_nums.iloc[1]
                        cur_img = np.array(tmp_ocr_df_byloc.Image, dtype=str)
                        cur_img = ",".join(cur_img.tolist())
                        ocr_res1, ocr_res2 = get_ocr_image_res(tmp_ocr_df_byloc)
                        ocr_view1, ocr_view2 = ocr_res1.split('-')[0], ocr_res2.split('-')[0]
                        # view checks
                        view_check_res = view_check(ocr_view1, ocr_view2)
                        f.write("view1: {}, view2: {}, view check: {} \n".format(ocr_view1, ocr_view2, view_check_res))
                        if (view_check_res == "same view") or (view_check_res == "multiple views"):
                            decision = "FAIL: pass loc check, fail view check"
                            f.write("fail to pass view check! \n")
                            find_img = False
                        elif (cur_num != 'none') and (((_num1 != 'none') and (_num1 != cur_num)) or ((_num2 != 'none') and (_num2 != cur_num))):
                            decision = "FAIL: pass loc check, fail num check"
                            f.write('fail to pass num check \n')
                            find_img = False
                        else:
                            decision = 'lat/loc'
                            find_img = True
                            f.write("pass view check, matched two images by two caliper detection output images \n")
                            res = [cur_img, cur_mrnenc, cur_site, nid, decision, find_img]
                            
                    elif len(tmp_ocr_df_byloc) > 2:
                        # num field is not none, check!
                        if cur_num != 'none':
                            f.write('******************** \n')
                            f.write('check lat + location + num match... \n')
                            tmp_ocr_df_bylocnum = tmp_ocr_df_byloc[tmp_ocr_df_byloc.ocr_nums == cur_num]
                            if len(tmp_ocr_df_bylocnum) == 2:
                                cur_img = np.array(tmp_ocr_df_bylocnum.Image, dtype=str)
                                cur_img = ",".join(cur_img.tolist())
                                ocr_res1, ocr_res2 = get_ocr_image_res(tmp_ocr_df_bylocnum)
                                ocr_view1, ocr_view2 = ocr_res1.split('-')[0], ocr_res2.split('-')[0]
                                # view checks
                                view_check_res = view_check(ocr_view1, ocr_view2)
                                f.write("view1: {}, view2: {}, view check: {} \n".format(ocr_view1, ocr_view2, view_check_res))
                                if (view_check_res == "same view") or (view_check_res == "multiple views"):
                                    decision = "FAIL: pass loc check, multiple images, fail view check"
                                    f.write("fail to pass view check! \n")
                                    find_img = False
                                else:
                                    decision = 'lat/loc/num'
                                    find_img = True
                                    f.write("pass view check, matched two images by two caliper detection output images \n")
                                    res = [cur_img, cur_mrnenc, cur_site, nid, decision, find_img]
                            elif len(tmp_ocr_df_bylocnum) > 2:
                                cur_img = np.array(tmp_ocr_df_bylocnum.Image, dtype=str)
                                cur_img = ",".join(cur_img.tolist())
                                decision = 'yield2'
                                find_img = False
                                res = [cur_img, cur_mrnenc, cur_site, nid, decision, find_img]
                            else:
                                decision = 'FAIL: lat/loc/num insufficient images'
                                find_img = False
                                                   
                        # num field is none, multiple images, check measurements
                        else:
                            cur_img = np.array(tmp_ocr_df_byloc.Image, dtype=str)
                            cur_img = ",".join(cur_img.tolist())
                            decision = 'yield2'
                            find_img = False
                            res = [cur_img, cur_mrnenc, cur_site, nid, decision, find_img]
                        
                # then check numbers, assumably location is none
                elif cur_num != 'none':
                    f.write('******************** \n')
                    f.write('check lat + num match... \n')
                    tmp_ocr_df_bynum = tmp_ocr_df[tmp_ocr_df.ocr_nums == cur_num]
                    if len(tmp_ocr_df_bynum) == 2:
                        cur_img = np.array(tmp_ocr_df_bynum.Image, dtype=str)
                        cur_img = ",".join(cur_img.tolist())
                        ocr_res1, ocr_res2 = get_ocr_image_res(tmp_ocr_df_bynum)
                        ocr_view1, ocr_view2 = ocr_res1.split('-')[0], ocr_res2.split('-')[0]
                        # view checks
                        view_check_res = view_check(ocr_view1, ocr_view2)
                        f.write("view1: {}, view2: {}, view check: {} \n".format(ocr_view1, ocr_view2, view_check_res))
                        if (view_check_res == "same view") or (view_check_res == "multiple views"):
                            decision = "FAIL: pass num check, no loc, fail view check"
                            f.write("fail to pass view check! \n")
                            find_img = False
                        else:
                            decision = 'lat/loc:none/num'
                            find_img = True
                            f.write("pass view check, matched two images by two caliper detection output images \n")
                            res = [cur_img, cur_mrnenc, cur_site, nid, decision, find_img]
                    elif len(tmp_ocr_df_bynum) < 2:
                        decision = 'FAIL: lat/loc:none/num insufficient images'
                        find_img = False
                    else:
                        decision = 'yield2'
                        cur_img = np.array(tmp_ocr_df_bynum.Image, dtype=str)
                        cur_img = ",".join(cur_img.tolist())
                        find_img = False
                        res = [cur_img, cur_mrnenc, cur_site, nid, decision, find_img]
                else:
                    decision = "yield2"
                    cur_img = np.array(tmp_ocr_df.Image, dtype=str)
                    cur_img = ",".join(cur_img.tolist())
                    find_img = False
                    res = [cur_img, cur_mrnenc, cur_site, nid, decision, find_img]
    f.write('END OF SEARCHING....... RESULTS: {} - {}\n'.format( find_img, res))
    return res, ocr_res1, ocr_res2, decision


def get_matched_patient(ocr_df, lat_df, f):
    """
        Get matching information for all candidate nodules
    """
    patients = np.unique(ocr_df['MRN'])
    f.write("number of patients: {}\n".format( len(np.unique(lat_df['MRN']))))
    word_tokenizer = RegexpTokenizer(r'\#\d|\w+')
    counter = 0
    mrns, nids, sites = [], [], []
    nodinfos = []
    studies, imgs = [], []
    decisions = []
    lsa_results = []
    ocr_results1, ocr_results2 = [], []
    for test_patient in patients:
        f.write("============ {} ============ \n".format( test_patient))
        test_lat_df = lat_df[(lat_df.MRN == test_patient)]
        tmp_mrn_enc = test_lat_df.MRN_Enc.values
        tmp_mrn_enc_, tmp_mrn_enc_counts = np.unique(tmp_mrn_enc,return_counts=True)
        tmp_mrn_enc_res = {tmp_mrn_enc_[i]: tmp_mrn_enc_counts[i] for i in range(len(tmp_mrn_enc_))}
        for idx in range(len(test_lat_df)):
            find_img = False
            test_ocr_df = ocr_df[ocr_df.MRN == test_patient]
            # remove duplicated rows in OCR
            tmp_lat_df = test_lat_df.iloc[idx]
            ################################################### 
            # info from df
            ###################################################
            cur_lat = tmp_lat_df.laterality_lsa
            cur_num = replace_num(tmp_lat_df.NoduleNumber_lsa)
            cur_loc_ = tmp_lat_df.location_lsa
            cur_loc_ = cur_loc_.strip('[]')
            cur_loc_ = word_tokenizer.tokenize(cur_loc_)
            cur_loc = '/'.join(cur_loc_)
            cur_loc_rev = '/'.join(cur_loc_[::-1])
            cur_site = tmp_lat_df.ReportSite
            cur_uniq = tmp_lat_df['is_unique']
            lsa_res = cur_lat + '-' + cur_loc + '-' + cur_num + '-' + str(cur_uniq)
            cur_mrnenc = str(tmp_lat_df.MRN_Enc)
            cur_nid = str(tmp_lat_df.NoduleId_lsa)
            num_nodules_lsa = tmp_mrn_enc_res[cur_mrnenc]
            cur_noduleinfo = cur_mrnenc + '-' + cur_nid + '-' + cur_lat
            f.write('lsa result for nodule {} is: {}, lat uniqueness is: {} \n'.format(cur_noduleinfo, lsa_res, cur_uniq))

            # check biopsy study first
            f.write('**********************\n')
            f.write("Checking Images in Biopsy Study....\n")
            cur_study = tmp_lat_df.BiopsyStudy
            if len(cur_study) > 10:
                biopsy_res, biopsy_ocr_res1, biopsy_ocr_res2, biopsy_decision = match_ocr_lat(test_ocr_df, tmp_lat_df, cur_study, num_nodules_lsa, f)
                if biopsy_res != 'none':

                    cur_img, cur_mrnenc, cur_site, nid, decision, find_img = biopsy_res[0], biopsy_res[1], biopsy_res[2], biopsy_res[3], biopsy_res[4], biopsy_res[5]
                    if find_img == True:
                        imgs.append(cur_img)
                        mrns.append(cur_mrnenc)
                        sites.append(cur_site)
                        nids.append(nid)
                        studies.append(cur_study)
                        decisions.append(decision)
                        lsa_results.append(lsa_res)
                        ocr_results1.append(biopsy_ocr_res1)
                        ocr_results2.append(biopsy_ocr_res2)
                        nodinfos.append(cur_noduleinfo)
                        f.write('Find image from biopsy study: {} \n'.format( find_img))
                    else:
                        f.write('No matched images in biopsy study \n')
                else:
                    imgs.append('none')
                    mrns.append(cur_mrnenc)
                    sites.append(cur_site)
                    nids.append(cur_nid)
                    studies.append(cur_study)
                    decisions.append(biopsy_decision)
                    lsa_results.append(lsa_res)
                    ocr_results1.append('none')
                    ocr_results2.append('none')
                    nodinfos.append(cur_noduleinfo)
                    f.write('No matched images in biopsy study \n')
            f.write('******************** \n')
            
            if find_img == False:
                f.write("Checking Images in Diagnostic Study.... \n")
                cur_study = tmp_lat_df.DiagnosticStudy
                if len(cur_study) > 10:
                    diagnostic_res, diagnostic_ocr_res1, diagnostic_ocr_res2, diag_decision = match_ocr_lat(test_ocr_df, tmp_lat_df, cur_study, num_nodules_lsa, f)
                    if diagnostic_res != 'none': 
                        cur_img, cur_mrnenc, cur_site, nid, decision, find_img = diagnostic_res[0], diagnostic_res[1], diagnostic_res[2], diagnostic_res[3], diagnostic_res[4], diagnostic_res[5]
                        # diagnostic study wont have M3 yields
                        if 'cd' in decision:
                            decision = 'FAIL: S2M3 not allowed'
                        imgs.append(cur_img)
                        mrns.append(cur_mrnenc)
                        sites.append(cur_site)
                        nids.append(nid)
                        studies.append(cur_study)
                        decisions.append(decision)
                        lsa_results.append(lsa_res)
                        nodinfos.append(cur_noduleinfo)
                        ocr_results1.append(diagnostic_ocr_res1)
                        ocr_results2.append(diagnostic_ocr_res2)
                        f.write('Find image from diagnostic study: \n'.format( find_img))
                    else:
                        imgs.append('none')
                        mrns.append(cur_mrnenc)
                        sites.append(cur_site)
                        nids.append(cur_nid)
                        studies.append(cur_study)
                        decisions.append(diag_decision)
                        lsa_results.append(lsa_res)
                        ocr_results1.append('none')
                        ocr_results2.append('none')
                        nodinfos.append(cur_noduleinfo)
                        f.write('No matched images in diagnostic study \n')
                else:
                    f.write('not a valid study name \n')
            f.write('******************** \n')
                  
            if find_img == False:
                f.write("Checking Images in Endo Study.... \n")  
                cur_study = tmp_lat_df.EndoStudy
                if len(cur_study) > 10:
                    endo_res, endo_ocr_res1, endo_ocr_res2, endo_decision = match_ocr_lat(test_ocr_df, tmp_lat_df, cur_study, num_nodules_lsa, f)
                    if endo_res != 'none':
                        cur_img, cur_mrnenc, cur_site, nid, decision, find_img = endo_res[0], endo_res[1], endo_res[2], endo_res[3], endo_res[4], endo_res[5]
                        # endo study count as FNA, but prioritize later
                        if 'yield2' in decision:
                            decision = 'FAIL: multiple images in Endo FNA'
                        imgs.append(cur_img)
                        mrns.append(cur_mrnenc)
                        sites.append(cur_site)
                        nids.append(nid)
                        studies.append(cur_study)
                        decisions.append(decision)
                        lsa_results.append(lsa_res)
                        ocr_results1.append(endo_ocr_res1)
                        ocr_results2.append(endo_ocr_res2)
                        nodinfos.append(cur_noduleinfo)
                        f.write('Find image from Endo study: {} \n'.format(find_img))
                    else:
                        imgs.append('none')
                        mrns.append(cur_mrnenc)
                        sites.append(cur_site)
                        nids.append(cur_nid)
                        studies.append(cur_study)
                        decisions.append(endo_decision)
                        lsa_results.append(lsa_res)
                        ocr_results1.append('none')
                        ocr_results2.append('none')
                        nodinfos.append(cur_noduleinfo)
                        f.write('No matched images in endo study \n')
            f.write('******************** \n')

    df = pd.DataFrame({
    'MRN_Enc': mrns,
    'NoduleId': nids,
    'MRN_ENC_NODID_LAT': nodinfos,
    'ReportSite': sites,
    'Study': studies,
    "Image": imgs,
    'Decisions': decisions,
    'LSA_res': lsa_results,
    'ocr_res_img1': ocr_results1,
    'ocr_res_img2': ocr_results2
        })
    return df, imgs

def sanity_check(res_df):
    """
        Additional check of matched results regarding views (long/trans) and nodule information
    """
    final_decision = np.array(res_df.Decisions)
    for idx, row in res_df.iterrows():
        mrn_enc, nodnum, decision = row.MRN_Enc, row.NoduleId, row.Decisions
        lat_res, ocr_res1, ocr_res2 = row.LSA_res, row.ocr_res_img1, row.ocr_res_img2
        uniq_lat = lat_res.split('-')[-1]
        if 'FAIL' in decision:
            continue
        else:
            # caliper detection
            if decision == 'cd':
                # check view
                view1, view2 = ocr_res1.split('-')[0], ocr_res2.split('-')[0]
                if (view1 != 'none') and (view2 != 'none'):
                    if (view1 == view2):
                        final_decision[idx] = 'FAIL: same view'
                # check laterality
                lat1, lat2 = ocr_res1.split('-')[1], ocr_res2.split('-')[1]
                lat_lsa = lat_res.split('-')[0]

                if (lat1 != 'none') and (lat2 != 'none'):
                    ## mismatch ocr results
                    if lat1 != lat2:
                        final_decision[idx] = 'FAIL: diff lat'
                    ## mismatch lsa results
                    elif (lat1 != lat_lsa) and (lat_lsa != 'none'):
                        final_decision[idx] = 'FAIL: not match with LSA'
            # cd + ocr
            if ('lat' in decision) or ('loc' in decision) or ('num' in decision):
                view1, view2 = ocr_res1.split('-')[0], ocr_res2.split('-')[0]
                if (view1 != 'none') and (view2 != 'none'):
                    if (view1 == view2):
                        final_decision[idx] = 'FAIL: same view'
            # yield 2 for measurements
            if decision == 'yield2':
                if uniq_lat != 'Yes':
                    final_decision[idx] = 'FAIL: not unique lat'
    res_df['DecisionFin'] = final_decision
    fin_df = res_df[~res_df['Decisions'].str.contains('FAIL')]
    return fin_df

