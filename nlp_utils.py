#####################################################
# Helper functions for NLP applications in MADLaP
#####################################################
import re
import numpy as np


#######################################################
# return_laterality, replace_location, return_location, replace_num, return_number
# are helper functions same as those in M2, M4
#######################################################
def return_laterality(lat_tokens, laterality_word_set):
    lat_tokens = [x.lower() for x in lat_tokens]
    if ('isthmus' in lat_tokens) or ('Isthmus' in lat_tokens) or ('isthmic' in lat_tokens):
        laterality_flag = 'isthmus'
    else:
        if ('left' in lat_tokens) or ('lt' in lat_tokens):
            laterality_flag = 'left'
        elif ('right' in lat_tokens) or ('rt' in lat_tokens):
            laterality_flag = 'right'
        else:
            laterality_flag = 'none'
    return laterality_flag

def replace_location(input_string):
    """
        normalize location string

        Parameters:
            input_string: an input location string

        Returns:
            new_string: a normalized location string
    """
    new_string = 'NA'
    if input_string == 'superior':
        new_string = input_string.replace('superior', 'sup')
    elif input_string == 'inferior':
        new_string = input_string.replace('inferior', 'inf')
    elif input_string == 'middle':
        new_string = input_string.replace('middle', 'mid')
    elif input_string == 'medial':
        new_string = input_string.replace('medial', 'med')
    elif input_string == 'anterior':
        new_string = input_string.replace('anterior', 'ant')
    elif input_string == 'posterior':
        new_string = input_string.replace('posterior', 'post')
    else:
        new_string = input_string
    return new_string


def return_laterality_flag(lat_tokens, laterality_word_set):
    """
        check if laterality is found

        Parameters:
            lat_tokens: candidate tokens may include laterality
            laterality_word_set: default laterality tokens

        Returns:
            laterality_flag: final laterality token

    """
    lat_tokens = [x.lower() for x in lat_tokens]
    if ('isthmus' in lat_tokens) or ('Isthmus' in lat_tokens) or ('isthmic' in lat_tokens):
        laterality_flag = 'isthmus'
    else:
        if ('left' in lat_tokens) or ('lt' in lat_tokens):
            laterality_flag = 'left'
        elif ('right' in lat_tokens) or ('rt' in lat_tokens):
            laterality_flag = 'right'
        else:
            laterality_flag = 'none'
    return laterality_flag

def return_location(loc_tokens, location_word_set):
    """
        Return any location tokens

        Parameters:
            loc_tokens: candidate tokens may include location
            location_word_set: default location tokens

        Returns:
            locs: final location token

    """
    locs = []
    loc_tokens = [x.lower() for x in loc_tokens]
    if any([x in loc_tokens for x in location_word_set]):
        true_list = [x in loc_tokens for x in location_word_set]
        true_idx = [i for i,x in enumerate(true_list) if x]
        locs = [location_word_set[idx] for idx in true_idx]
        locs = [replace_location(x) for x in locs]
        locs = '/'.join(locs)
    else:
        locs = 'none'
    return locs

def return_number(number_tokens, pattern=r'\#\d'):
    """
        Return label number

        Parameters:
            number_tokens: candidate tokens may include numbers
            pattern: regex pattern to extract numbers

        Returns:
            res: final label number token

    """
    res = re.findall(pattern, number_tokens)
    if len(res) == 0:
        res = 'none'
    return res

def return_location_M4(loc_tokens, location_word_set):
    locs = 'none'
    loc_id = ['none']
    loc_tokens = [x.lower() for x in loc_tokens]
    if any([x in loc_tokens for x in location_word_set]):
        true_list = [x in loc_tokens for x in location_word_set]
        true_idx = [i for i,x in enumerate(true_list) if x]
        locs_ = [location_word_set[idx] for idx in true_idx]
        locs_rep = [replace_location(x) for x in locs_]
        locs = '/'.join(locs_rep)
        loc_id = [i for i,x in enumerate(loc_tokens) if x in locs_]
        if len(true_idx) == 0:
            loc_id = ['none']
    else:
        locs = 'none'
    
    return locs, loc_id[-1]

def replace_num(input_string):
    return input_string.replace('#', '')


def return_number_M4(number_tokens, pattern=r'\#\d'):
    for x in number_tokens:
        res = re.findall(pattern, x)
        if len(res) == 0:
            res = ['none']
            continue
        else:
            break
    return replace_num(res[0])