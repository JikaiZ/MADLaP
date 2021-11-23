#####################################################
# Functions for M4 - OCR
#####################################################

import os

import cv2
import matplotlib.pyplot as plt
import pytesseract

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from nltk.tokenize import RegexpTokenizer, word_tokenize

from PIL import Image
from nlp_utils import *

from tqdm import tqdm

def read_img(image_name, image_folder):
    """read image by given file"""
    image_path = os.path.join(image_folder, image_name)
    image = Image.open(image_path)
    return image

def crop_img(image, coordinates):
    """crop image by given coordinates: a list containes left, top, right, bottom"""
    left, top, right, bottom = coordinates
    cropped_sample = image.crop((left, top, right, bottom))
    return cropped_sample

def save_img(cropped, image_name, save_folder):
    """save image to the destination"""
    save_path = os.path.join(save_folder, image_name)
    cropped.save(save_path)
    
    
def cropped_bottom_images(image_folder, save_folder, crop_measure=False): 
    """
        Crop images by a fixed proportion

        Parameters:
            image_folder: folder that saves all images
            save_folder: destination folder
            crop_measure: True if cropping measurement in the corner


    """
    for (dirpath, dirnames, filenames) in tqdm(os.walk(image_folder)):
        for filename in filenames:
            image_file = os.path.join(dirpath, filename)
            if '.png' not in image_file:
            	continue
            image_file_tokens = image_file.split('/')
            MRN = image_file_tokens[-3]
            study_name = '-'.join(image_file_tokens[-2].split('-')[:-2])
            image_name = image_file_tokens[-1] 
            # check if MRN folder exists
            PATH_TO_MRN = os.path.join(save_folder, MRN)
            if not os.path.isdir(PATH_TO_MRN):
                os.mkdir(PATH_TO_MRN)
            # check if study folder exists
            PATH_TO_MRN_STUDY = os.path.join(PATH_TO_MRN, study_name)
            if not os.path.isdir(PATH_TO_MRN_STUDY):
                os.mkdir(PATH_TO_MRN_STUDY)
            cropped_image_name = image_name.split('.')[0] + '_cropped.png'
            PATH_TO_CROPPED_IMAGE = os.path.join(PATH_TO_MRN_STUDY, cropped_image_name)
            # read image
            img = Image.open(image_file)
            length_x, width_y = img.size
            factor = min(1, float(1024.0 / length_x))
            size = int(factor * length_x), int(factor * width_y)
            im_resized = img.resize(size, Image.ANTIALIAS)
            width, height = im_resized.size
            # Crop measurement
            if crop_measure == True:
                coord = [0, int(4/5 * height), int(1/4*width), height]
            elif 'US-THYROID-IMAGES' in dirpath:
                coord = [0, 0, int(5/6 * width), int(1/10 * height)]
            else:
                coord = [0, int(4/5 * height), width, height]
            
            cropped_bottom = crop_img(im_resized, coord)
            cropped_bottom.save(PATH_TO_CROPPED_IMAGE, "PNG")

def process_ocr_rawtext(text):
    """Strip text"""
    text = text.strip().split('\n')
    text = ' '.join(text)
    text = text.replace(';', '')
    return text

def ocr_texts(image_folder):
    """Apply OCR to a given image. Five configurations of OCR engine were used to generate outputs. Results were saved in a dataframe
    """
    texts1, texts2, texts3, texts4, texts5 = [], [], [], [], []
    mrns = []
    study_names = []
    image_names = []

    
    for (dirpath, dirnames, filenames) in tqdm(os.walk(image_folder)):
        for filename in filenames:
            if 'png' not in filename:
                continue
            image_file = os.path.join(dirpath, filename)
            image_file_tokens = image_file.split('/')
            MRN = image_file_tokens[-3]
            study_name = '-'.join(image_file_tokens[-2].split('-'))
            image_name = image_file_tokens[-1] 
            mrns.append(MRN)
            study_names.append(study_name)
            image_names.append(image_name)
            img = cv2.imread(image_file, 0)
            # specification 1
            img1 = img
            text1 = process_ocr_rawtext(pytesseract.image_to_string(img1, lang='eng'))
            texts1.append(text1)
            # specification 2
            img2 = cv2.resize(img,(0,0),fx=3,fy=3)
            img2 = cv2.GaussianBlur(img2,(11,11),0)
            img2 = cv2.medianBlur(img2,9)
            text2 = process_ocr_rawtext(pytesseract.image_to_string(img2, lang='eng'))
            texts2.append(text2)
            # specification 3
            img3 = cv2.resize(img,(0,0),fx=13,fy=13)
            img3 = cv2.GaussianBlur(img3,(11,11),0)
            img3 = cv2.medianBlur(img3, 9)
            text3 = process_ocr_rawtext(pytesseract.image_to_string(img3, lang='eng'))
            texts3.append(text3)
            # specification 4
            img4 = cv2.resize(img,(0,0),fx=15,fy=15)
            img4 = cv2.GaussianBlur(img4,(11,11),0)
            img4 = cv2.medianBlur(img4, 9)
            text4 = process_ocr_rawtext(pytesseract.image_to_string(img4, lang='eng'))
            texts4.append(text4)
            # specification 5
            img5 = cv2.resize(img,(0,0),fx=21,fy=21)
            img5 = cv2.GaussianBlur(img5,(11,11),0)
            img5 = cv2.medianBlur(img5, 9)
            text5 = process_ocr_rawtext(pytesseract.image_to_string(img5, lang='eng'))
            texts5.append(text5)
            
    df = pd.DataFrame({
    'MRN': mrns,
    'Study': study_names,
    'Image': image_names,
    'OCR_output1': texts1,
    'OCR_output2': texts2,
    'OCR_output3': texts3,
    'OCR_output4': texts4,
    'OCR_output5': texts5,
    })
    
    return df

def get_ocr_res_one(ocr_output, location_word_set, laterality_word_set, view_word_set, f):
    """
        Extract nodule information from one given OCR output

        Parameters:
            ocr_output: ocr output from an image
            location_word_set: default location tokens
            laterality_word_set: default laterality tokens
            view_word_set: trans/long view tokens

        Returns:
            view, lat, loc, num: corresponds to trans/long view, laterality, location, and nodule label

    """
    word_tokenizer = RegexpTokenizer(r'\#\d|\w+')
    test_string = ocr_output
    f.write('{}\n'.format(test_string))
    view, lat, loc, num = 'none', 'none', 'none', 'none'
    if pd.isnull(test_string) == True:
        test_string = '...'
    tokenized = word_tokenizer.tokenize(test_string)
    # print(tokenized, len(tokenized))
    if len(tokenized) == 0:
        view = 'none'
        lat = 'none'
        loc = 'none'
        num = 'none'
    else:
        items = 'none'
        for x in tokenized:
            match = re.match(r"([a-z]+)([0-9]+)", x, re.I)
            if match:
                items = match.groups()
                beforenum = items[-2]
                if beforenum not in location_word_set:
                    items = 'none'
                    num = 'none'
                else:
                    num = items[-1]

        if items != 'none':    
            tokenized.extend(items)
        f.write("Tokenized string: {}\n".format( tokenized))
    
        lat = return_laterality(tokenized, laterality_word_set)
        loc, loc_id = return_location_M4(tokenized, location_word_set)
        if (loc_id != 'none') and (num == 'none'):
            loc_id = int(loc_id)
            next_id = loc_id + 1
            if next_id >= len(tokenized):
                pass
            else:
                loc_next_token = tokenized[loc_id + 1]
                # print(tokenized[loc_id], loc_next_token)
                num_tmp = re.findall(r'^[0-9]*$', loc_next_token)
                # print(num_tmp)
                if len(num_tmp)!=0:
                    num = num_tmp[0]
        view, _ = return_location_M4(tokenized, view_word_set)
        thyroid, thy_id = return_location_M4(tokenized, ['thyroid', 'thy'])
        if (thy_id != 'none') and (num == 'none'):
            thy_id = int(thy_id)
            next_id = thy_id + 1
            if next_id >= len(tokenized):
                pass
            else:
                thy_next_token = tokenized[thy_id + 1]
                # print(tokenized[thy_id], thy_next_token)
                num_tmp = re.findall(r'^[0-9]*$', thy_next_token)
                # print(num_tmp)
                if len(num_tmp)!=0:
                    num = num_tmp[0]
        if num == 'none':
            num = return_number_M4(tokenized)
    f.write('view: {}; lat: {}, loc: {}, num: {}\n'.format(view, lat, loc, num))
    return view, lat, loc, num


def get_ocr_texts(ocr_df, location_word_set, laterality_word_set, view_word_set, f):
    """
        Summarize 5 OCR outputs for all images.

        Parameters:
            ocr_df: 5 direct ocr outputs for each image. 
            location_word_set: default location tokens
            laterality_word_set: default laterality tokens
            view_word_set: trans/long view tokens

        Returns:
            df: nodule information result

    """
    idx = ['OCR' in x for x in ocr_df.columns]
    columns = pd.Index(np.array(ocr_df.columns)[idx])
    ocr_outputs = ocr_df[columns].values
    nrow, ncol = ocr_outputs.shape[0], ocr_outputs.shape[1]
    meta_cols = ['MRN', 'Study', 'Image', 
                 'ocr_view', 'ocr_laterality', 'ocr_location', 'ocr_nums']
    col_dict = {col: [] for col in meta_cols}
    col_dict['MRN'] = ocr_df.MRN.values
    col_dict['Study'] = ocr_df.Study.values
    col_dict['Image'] = ocr_df.Image.values
    for rownum in range(nrow):
        f.write('==== {} ==== \n'.format(ocr_df.MRN.values[rownum]))
        f.write('Img: {} \n'.format(ocr_df.Image.values[rownum]))
        # print('=' * 30, ocr_df.MRN.values[rownum], '===', ocr_df.Image.values[rownum], '=' * 30)
        find_view, find_lat, find_loc, find_num = False, False, False, False 
        _view, _lat, _loc, _num = 'none', 'none', 'none', 'none'
        for rowcol in range(ncol):
            cur_outputs = ocr_outputs[rownum, rowcol]
            cur_view, cur_lat, cur_loc, cur_num = get_ocr_res_one(cur_outputs, location_word_set, laterality_word_set, view_word_set, f)
            if (cur_lat != 'none') and (find_lat is False):
                _lat = cur_lat
                find_lat = True
            if (cur_view != 'none') and (find_view is False):
                _view = cur_view
                find_view = True
            if (cur_loc != 'none') and (find_loc is False):
                _loc = cur_loc
                find_loc = True
            if (cur_num != 'none') and (find_num is False):
                _num = cur_num
                find_num = True
        col_dict['ocr_view'].append(_view)
        col_dict['ocr_laterality'].append(_lat)
        col_dict['ocr_location'].append(_loc)
        col_dict['ocr_nums'].append(_num)
    return pd.DataFrame(col_dict)
