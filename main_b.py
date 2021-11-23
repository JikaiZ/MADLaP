#######################################################
# Main implementation to get benign yields
#######################################################

import pandas as pd 
import numpy as np
import os
from tqdm import tqdm
import time
from datetime import timedelta
import shutil

# M3
import tensorflow as tf
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
import sys
from glob import glob

# import functions from scripts
from utils import *
from M1 import *
from M2 import *
from M3 import *
from M4 import *
from M5 import *
from info_matching_1 import *
from info_matching_2 import *

def main(png_folder, data_folder, results_folder, log_folder, path_report_file, rad_report_file, accn_file, cd_thres):
	timer_file = os.path.join(log_folder, 'time_cost.txt')
	f_time = open(timer_file, 'a')
	# Module 1
	start_m1 = time.time()
	print('Process M1...')
	process_M1(png_folder, data_folder, log_folder)
	end_m1 = time.time()
	elapsed_m1 = end_m1 - start_m1
	f_time.write("M1 Elapsed: {} \n".format(str(timedelta(seconds=elapsed_m1))))
	# Module 2
	start_m2 = time.time()
	print('Process M2...')
	process_M2(data_folder, log_folder)
	end_m2 = time.time()
	elapsed_m2 = end_m2 - start_m2
	f_time.write("M2 Elapsed: {} \n".format(str(timedelta(seconds=elapsed_m2))))
	# Module 3
	start_m3 = time.time()
	print('Process M3...')
	process_M3(png_folder, data_folder, log_folder, cd_thres)
	end_m3 = time.time()
	elapsed_m3 = end_m3 - start_m3
	f_time.write("M3 Elapsed: {} \n".format(str(timedelta(seconds=elapsed_m3))))
	# Module 4
	start_m4 = time.time()
	print('Process M4...')
	process_M4(data_folder, log_folder)
	end_m4 = time.time()
	elapsed_m4 = end_m4 - start_m4
	f_time.write("M4 Elapsed: {} \n".format(str(timedelta(seconds=elapsed_m4))))
	Info matching part 1 
	print('Process info matching part 1 ...')
	start_info1 = time.time()
	info_matching_p1(png_folder, data_folder, results_folder, log_folder)
	end_info1 = time.time()
	elapsed_info1 = end_info1 - start_info1
	f_time.write("Info matching part 1 Elapsed: {} \n".format(str(timedelta(seconds=elapsed_info1))))
	Module 5
	start_m5 = time.time()
	print('Process M5...')
	# process_M5(png_folder, data_folder, results_folder, log_folder)
	end_m5 = time.time()
	elapsed_m5 = end_m5 - start_m5
	f_time.write("M5 Elapsed: {} \n".format(str(timedelta(seconds=elapsed_m5))))
	Info matching part 2
	print('Process info matching part 2 ...')
	start_info2 = time.time()
	info_matching_p2(data_folder, results_folder, log_folder)
	end_info2 = time.time()
	elapsed_info2 = end_info2 - start_info2
	f_time.write("Info matching part 2 Elapsed: {} \n".format(str(timedelta(seconds=elapsed_info2))))
	# Copy selected images
	start_copy = time.time()
	print('Copying images...')
	copy_sel_imgs(png_folder, data_folder, results_folder, log_folder)
	end_copy = time.time()
	elapsed_copy = end_copy - start_copy
	f_time.write("Copying Images Elapsed: {} \n".format(str(timedelta(seconds=elapsed_copy))))
	f_time.close()
	return 0

def copy_sel_imgs(png_folder, data_folder, results_folder, log_folder):
	# specifications
	homedir = os.path.expanduser('~')
	log_file = os.path.join(log_folder, 'copying_imgs.txt')
	f = open(log_file, 'w')
	m1_data_folder = os.path.join(data_folder, 'M1')
	m2_data_folder = os.path.join(data_folder, 'M2')
	m3_data_folder = os.path.join(data_folder, 'M3')
	m4_data_folder = os.path.join(data_folder, 'M4')
	copy_dest_folder = os.path.join(results_folder, 'benign')
	copy_dest_folder_s1 = os.path.join(copy_dest_folder, 'S1')
	copy_dest_folder_s2 = os.path.join(copy_dest_folder, 'S2')
	copy_dest_folder_s1m3 = os.path.join(copy_dest_folder_s1, 'M3')
	copy_dest_folder_s1m4 = os.path.join(copy_dest_folder_s1, 'M4')
	copy_dest_folder_s1m5 = os.path.join(copy_dest_folder_s1, 'M5')
	copy_dest_folder_s2m3 = os.path.join(copy_dest_folder_s2, 'M3')
	copy_dest_folder_s2m4 = os.path.join(copy_dest_folder_s2, 'M4')
	copy_dest_folder_s2m5 = os.path.join(copy_dest_folder_s2, 'M5')
	if not os.path.isdir(copy_dest_folder):
		os.mkdir(copy_dest_folder)
	if not os.path.isdir(copy_dest_folder_s1):
		os.mkdir(copy_dest_folder_s1)
	if not os.path.isdir(copy_dest_folder_s2):
		os.mkdir(copy_dest_folder_s2)
	if not os.path.isdir(copy_dest_folder_s1m3):
		os.mkdir(copy_dest_folder_s1m3)
	if not os.path.isdir(copy_dest_folder_s1m4):
		os.mkdir(copy_dest_folder_s1m4)
	if not os.path.isdir(copy_dest_folder_s1m5):
		os.mkdir(copy_dest_folder_s1m5)
	if not os.path.isdir(copy_dest_folder_s2m3):
		os.mkdir(copy_dest_folder_s2m3)
	if not os.path.isdir(copy_dest_folder_s2m4):
		os.mkdir(copy_dest_folder_s2m4)
	if not os.path.isdir(copy_dest_folder_s2m5):
		os.mkdir(copy_dest_folder_s2m5)

	diagnostic_folder_names = ['US-SOFT-TISSUE-HEAD-AND-NECK']
	biopsy_folder_names = ['US-THYROID-NON-CYST-FINE-NEEDLE-ASPIRATION', 'US-THYROID-BIOPSY', 'US-FNA-THYROID-IMAGES']
	pathology_folder_names = ['US-THYROID-IMAGES']
	# copy selected pairs of images and rename it
	yield_df1 = pd.read_csv(os.path.join(results_folder, 'info_matching1_b.csv'))
	yield_df2 = pd.read_csv(os.path.join(results_folder, 'info_matching2_b.csv'))
	yield2_nodids = yield_df2[yield_df2.Decisions == 'yield2'].MRN_ENC_NODID_LAT.tolist()
	counter = 0
	s1m3_counter, s1m4_counter = 0, 0
	s2m3_counter, s2m4_counter, s2m5_counter = 0, 0, 0
	img_res_df = pd.DataFrame()
	for index, row in yield_df1.iterrows():
		mrnenc, study = row.MRN_Enc, row.Study
		nodid = row.MRN_ENC_NODID_LAT
		mrn = mrnenc.split('-')[0]
		decision, fin_decision = row.Decisions, row.DecisionFin
		if 'FAIL' in fin_decision:
			continue
		elif 'yield2' in fin_decision:
			yield2_res = yield_df2[yield_df2.MRN_ENC_NODID_LAT == nodid]
			for idx2, row2 in yield2_res.iterrows():
				study = row2.Study
				images = row2.Image
				ocr_res1, ocr_res2 = row2.ocr_res_img1, row2.ocr_res_img2
				yield2_decision = row2.Decisions

			if 'FAIL' not in yield2_decision:
				lsa_res = row.LSA_res
				
				image1, image2 = images.split(',')[0], images.split(',')[1]
				image1_name, image2_name = image1.split('_')[0] + '.png', image2.split('_')[0] + '.png'
				
				view1, view2 = ocr_res1.split('-')[0], ocr_res2.split('-')[0]
				if view1 == 'none':
					view1 = view1 + '1'
				if view2 == 'none':
					view2 = view2 + '2'
				study_name = study.split('_')[0]
				dest1_name, dest2_name = '{}_{}.png'.format(nodid, view1), '{}_{}.png'.format(nodid, view2)
				stage, module = 'S2', 'M5'
				dest_folder = copy_dest_folder_s2m5
				s2m5_counter += 1
			else:
				continue

		else:
			## get images
			images = row.Image
			image1, image2 = images.split(',')[0], images.split(',')[1]
			image1_name, image2_name = image1.split('_')[0] + '.png', image2.split('_')[0] + '.png'
			ocr_res1, ocr_res2 = row.ocr_res_img1, row.ocr_res_img2
			lsa_res = row.LSA_res
			view1, view2 = ocr_res1.split('-')[0], ocr_res2.split('-')[0]
			if view1 == 'none':
				view1 = view1 + '1'
			if view2 == 'none':
				view2 = view2 + '2'
			img1_file = os.path.join(png_folder, mrn, study, image1_name)
			img2_file = os.path.join(png_folder, mrn, study, image2_name)
			dest1_name, dest2_name = '{}_{}.png'.format(nodid, view1), '{}_{}.png'.format(nodid, view2)
			study_name = study.split('_')[0]
			stage = 'none'
			if study_name in biopsy_folder_names + pathology_folder_names:
				stage = 'S1'
				if decision == 'cd':
					module = 'M3'
					dest_folder = copy_dest_folder_s1m3
					s1m3_counter += 1
				elif ('lat' in decision) or ('loc' in decision) or ('num' in decision):
					module = 'M4'
					dest_folder = copy_dest_folder_s1m4
					s1m4_counter += 1
				else:
					continue
			elif study_name in diagnostic_folder_names:
				stage = 'S2'
				if decision == 'cd':
					module = 'M3'
					dest_folder = copy_dest_folder_s2m3
					s2m3_counter += 1
				elif ('lat' in decision) or ('loc' in decision) or ('num' in decision):
					module = 'M4'
					dest_folder = copy_dest_folder_s2m4
					s2m4_counter += 1
				else:
					continue
			else:
				continue
		image1_size = Image.open(img1_file).size
		image2_size = Image.open(img2_file).size
		f.write('====================== \n')
		f.write('Image: {}, size: {}\n'.format(dest1_name, image1_size))
		f.write('Image: {}, size: {}\n'.format(dest2_name, image2_size))
		# copy images
		dest1_file, dest2_file = os.path.join(dest_folder, dest1_name), os.path.join(dest_folder, dest2_name)
		shutil.copyfile(img1_file, dest1_file)
		shutil.copyfile(img2_file, dest2_file)
		counter += 1
		# collect image info
		meta_cols = ['MRN_Enc', 'MRN_ENC_NODID_LAT', 'Study', 'Original_IMG', 'Dest_IMG', 'Image_Dim', 'Stage', 'Module', 'OCR_res', 'LSA_res']
		data_dict = {}
		data_dict['MRN_Enc'] = [mrnenc, mrnenc] 
		data_dict['MRN_ENC_NODID_LAT'] = [nodid, nodid]
		data_dict['Study'] = [study, study]
		data_dict['Original_IMG'] = [image1_name, image2_name]
		data_dict['Dest_IMG'] = [dest1_name, dest2_name]
		data_dict['Image_Dim'] = [image1_size, image2_size]
		data_dict['Stage'] = [stage, stage]
		data_dict['Module'] = [module, module]
		data_dict['OCR_res'] = [ocr_res1, ocr_res2]
		data_dict['LSA_res'] = [lsa_res, lsa_res]
		cur_df = pd.DataFrame(data_dict)
		img_res_df = pd.concat([img_res_df, cur_df])

	f.write('number of total selected nodules: %d \n' % counter)
	f.write('number of total selected nodules in S1M3: %d \n' % s1m3_counter)
	f.write('number of total selected nodules in S1M4: %d \n' % s1m4_counter)
	f.write('number of total selected nodules in S2M3: %d \n' % s2m3_counter)
	f.write('number of total selected nodules in S2M4: %d \n' % s2m4_counter)
	f.write('number of total selected nodules in S2M5: %d \n' % s2m5_counter)
	img_res_df.to_csv(os.path.join(results_folder, 'image_summary_b.csv'), index=False)
	f.close()
	return 0

def info_matching_p2(data_folder, results_folder, log_folder):
	# specifications
	homedir = os.path.expanduser('~')
	log_file = os.path.join(log_folder, 'info_matching2.txt')
	f = open(log_file, 'w')
	m4_data_folder = os.path.join(data_folder, 'M4')
	m5_data_folder = os.path.join(data_folder, 'M5')
	ocr_text_df_b = pd.read_csv(os.path.join(m4_data_folder, 'ocr_outputs_postprocessed_b.csv'))
	measurements_radreport_b = pd.read_csv(os.path.join(m5_data_folder, 'radreport_measure_b.csv'))
	measure_b_df = pd.read_csv(os.path.join(m5_data_folder, 'ocr_outputs_measurements_postprocessed_b.csv'))
	# match ocr and report measurements
	measure_res_df_b = match_image_by_measurements(measurements_radreport_b, measure_b_df, ocr_text_df_b, f)
	measure_res_df_b.to_csv(os.path.join(results_folder, 'info_matching2_b.csv'), index=False)
	f.write('# of succesful yields: {}'.format(sum(measure_res_df_b.Decisions != 'FAIL')))
	f.close()
	return 0 

def process_M5(png_folder, data_folder, results_folder, accn_file, rad_report_file, log_folder):
	# specifications
	homedir = os.path.expanduser('~')
	log_ocr_file = os.path.join(log_folder, 'M5_ocr.txt')
	log_rep_file = os.path.join(log_folder, 'M5_report.txt')
	f0 = open(log_ocr_file, 'w')
	f1 = open(log_rep_file, 'w')
	# m1_data_folder = os.path.join(data_folder, 'M1')
	m2_data_folder = os.path.join(data_folder, 'M2')
	# m3_data_folder = os.path.join(data_folder, 'M3')
	m4_data_folder = os.path.join(data_folder, 'M4')
	m5_data_folder = os.path.join(data_folder, 'M5')
	if not os.path.isdir(m5_data_folder):
		os.mkdir(m5_data_folder)
	data_file = accn_file
	qido_file = os.path.join(homedir, data_file)
	lat_res_df_b = pd.read_csv(os.path.join(m2_data_folder, 'process_lsa_res_b.csv'))
	measure_ocr_res_df_b = pd.read_csv(os.path.join(m4_data_folder, 'ocr_outputs_measurements_b.csv'))
	match_yield1_b = pd.read_csv(os.path.join(results_folder, 'info_matching1_b.csv'))
	report_df_b = rad_report_file
	# get measurements from ocr
	measure_b_df = get_measure_df(match_yield1_b, measure_ocr_res_df_b, lat_res_df_b, f0)
	measure_b_df.to_csv(os.path.join(m5_data_folder, 'ocr_outputs_measurements_postprocessed_b.csv'), index=False)
	# collate radiology reports
	sel_rad_report_df_b = collate_rad_report(match_yield1_b, qido_file, report_df_b, png_folder)
	sel_rad_report_df_b.to_csv(os.path.join(m5_data_folder, 'selected_rad_report_b.csv'), index=False)
	# get measurements from rad reports
	measurements_radreport_b = get_measurements_from_radreport(sel_rad_report_df_b, log_rep_file)
	measurements_radreport_b.to_csv(os.path.join(m5_data_folder, 'radreport_measure_b.csv'))

	f0.close()
	f1.close()
	return 0

def info_matching_p1(png_folder, data_folder, results_folder, log_folder):
	# specifications
	homedir = os.path.expanduser('~')
	log_file = os.path.join(log_folder, 'info_matching1.txt')
	f = open(log_file, 'w')
	m1_data_folder = os.path.join(data_folder, 'M1')
	m2_data_folder = os.path.join(data_folder, 'M2')
	m3_data_folder = os.path.join(data_folder, 'M3')
	m4_data_folder = os.path.join(data_folder, 'M4')
	selected_study_df_b = pd.read_csv(os.path.join(m1_data_folder, 'selected_directories_benign.csv'))
	lat_res_df_b = pd.read_csv(os.path.join(m2_data_folder, 'process_lsa_res_b.csv'))
	ocr_res_df_b = pd.read_csv(os.path.join(m4_data_folder, 'ocr_outputs_postprocessed_b.csv'))
	biopsy_study_b = get_biopsy_studies(lat_res_df_b, selected_study_df_b)
	diagnostic_study_b = get_diagnostic_studies(lat_res_df_b, selected_study_df_b)
	endo_study_b = get_endo_studies(lat_res_df_b, selected_study_df_b)
	lat_res_df_b['BiopsyStudy'] = biopsy_study_b
	lat_res_df_b['DiagnosticStudy'] = diagnostic_study_b
	lat_res_df_b['EndoStudy'] = endo_study_b
	# match info
	matched_patient_b, matched_imgs_b = get_matched_patient(ocr_res_df_b, lat_res_df_b, f)
	matched_patient_b.to_csv(os.path.join(results_folder, 'full_matched_yield1_b.csv'), index=False)
	# sanity checks
	matched_patient_b_checked = sanity_check(matched_patient_b)
	matched_patient_b_checked.to_csv(os.path.join(results_folder, 'info_matching1_b.csv'), index=False)
	f.close()
	return 0

def process_M4(data_folder, log_folder):
	# specifications
	homedir = os.path.expanduser('~')
	log_file = os.path.join(log_folder, 'M4_ocr_res.txt')
	m3_data_folder = os.path.join(data_folder, 'M3')
	m4_data_folder = os.path.join(data_folder, 'M4')
	ocr_folder = os.path.join(m4_data_folder, 'ocr_images')
	ocr_folder_b = os.path.join(ocr_folder, 'benign')
	ocr_measure_folder = os.path.join(m4_data_folder, 'ocr_images_measurement')
	ocr_measure_folder_b = os.path.join(ocr_measure_folder, 'benign')
	## previously saved cd images
	cd_image_folder_b = os.path.join(m3_data_folder, 'caliper_detection_images/benign')
	f = open(log_file, 'w')
	if not os.path.isdir(m4_data_folder):
		os.mkdir(m4_data_folder)
	if not os.path.isdir(ocr_folder):
		os.mkdir(ocr_folder)
	if not os.path.isdir(ocr_folder_b):
		os.mkdir(ocr_folder_b)
	if not os.path.isdir(ocr_measure_folder):
		os.mkdir(ocr_measure_folder)
	if not os.path.isdir(ocr_measure_folder_b):
		os.mkdir(ocr_measure_folder_b)

	# crop images
	print('crop for texts...')
	crop_measure = False
	cropped_bottom_images(cd_image_folder_b, ocr_folder_b, crop_measure)
	print('crop for measurements...')
	crop_measure = True
	cropped_bottom_images(cd_image_folder_b, ocr_measure_folder_b, crop_measure)
	# ocr
	print('Processing texts...')
	benign_ocr_df = ocr_texts(ocr_folder_b)
	benign_ocr_df.to_csv(os.path.join(m4_data_folder, 'ocr_outputs_preprocessed_b.csv'), index=False)
	print('Processing measurements...')
	measure_ocr_df_b = ocr_texts(ocr_measure_folder_b)
	measure_ocr_df_b.to_csv(os.path.join(m4_data_folder, 'ocr_outputs_measurements_b.csv'), index=False)
	# post-process ocr texts
	benign_ocr_df = pd.read_csv(os.path.join(m4_data_folder, 'ocr_outputs_preprocessed_b.csv'))
	laterality_word_set = ['right', 'left', 'Right', 'Left', 'Rt', 'Lt', 'rt', 'lt', 'RT', 'LT']
	location_word_set = ['inf', 'sup', 'mid', 'medial', 'med', 'ant', 'inferior', 'superior', 'posterior', 'anterior']
	view_word_set = ['trans', 'long', 'sag']
	ocr_outputs_b_df = get_ocr_texts(benign_ocr_df, location_word_set, laterality_word_set, view_word_set, f)
	ocr_outputs_b_df.to_csv(os.path.join(m4_data_folder, 'ocr_outputs_postprocessed_b.csv'), index=False)
	
	f.close()
	return 0

def process_M3(png_folder, data_folder, log_folder, threshold):
	# specifications
	homedir = os.path.expanduser('~')
	log_file = os.path.join(log_folder, 'M3_cd.txt')
	m1_data_folder = os.path.join(data_folder, 'M1')
	m2_data_folder = os.path.join(data_folder, 'M2')
	m3_data_folder = os.path.join(data_folder, 'M3')
	f = open(log_file, 'w')
	if not os.path.isdir(m3_data_folder):
		os.mkdir(m3_data_folder)
	score_folder = os.path.join(m3_data_folder, 'caliper_detection_scores')
	if not os.path.isdir(score_folder):
		os.mkdir(score_folder)
	score_folder_b = os.path.join(score_folder, 'benign')
	if not os.path.isdir(score_folder_b):
		os.mkdir(score_folder_b)
	cd_img_folder = os.path.join(m3_data_folder, 'caliper_detection_images')
	if not os.path.isdir(cd_img_folder):
		os.mkdir(cd_img_folder)
	cd_img_folder_b = os.path.join(cd_img_folder, 'benign')
	if not os.path.isdir(cd_img_folder_b):
		os.mkdir(cd_img_folder_b)
	# caliper detection
	dir_file_b = pd.read_csv(os.path.join(m1_data_folder, 'selected_directories_benign.csv'))
	# ## FNA studies
	print("Process FNA studies...")
	mrns_b_fna, dir_file_b_fna = list(dir_file_b['MRN']), list(dir_file_b['BiopsyStudyDir'])
	f.write('number of benign studies from FNA: {}\n'.format( len(dir_file_b_fna)))
	print('get model...')
	get_cd_model(dir_file_b_fna, score_folder_b, f)
	print('copy selected images...')
	copy_selected_imgs(dir_file_b_fna, score_folder_b, cd_img_folder_b, threshold)
	## Diagnostic studies
	print("Process Diagnostic studies...")
	mrns_b_diag, dir_file_b_diag = list(dir_file_b['MRN']), list(dir_file_b['DiagnosticStudyDir'])
	f.write('number of benign studies from Diagnostic: {}\n'.format( len(dir_file_b_diag)))
	print('get model...')
	get_cd_model(dir_file_b_diag, score_folder_b, f)
	print('copy selected images...')
	copy_selected_imgs(dir_file_b_diag, score_folder_b, cd_img_folder_b, threshold)
	## Endo studies
	print("Process Endo studies...")
	mrns_b_path, dir_file_b_path = list(dir_file_b['MRN']), list(dir_file_b['PathologyStudyDir'])
	f.write('number of benign studies from Endocrine: {}\n'.format( len(dir_file_b_path)))
	print('get model...')
	get_cd_model(dir_file_b_path, score_folder_b, f)
	print('copy selected images...')
	copy_selected_imgs(dir_file_b_path, score_folder_b, cd_img_folder_b, threshold)
	f.close()
	return 0

def process_M2(data_folder, path_report_file, log_folder):
	# configs
	homedir = os.path.expanduser('~')
	benign_path_df = path_report_file
	log_file = os.path.join(log_folder, 'M2_summary.txt')
	log_file_b = os.path.join(log_folder, 'M2_lsa_b.txt')
	m1_data_folder = os.path.join(data_folder, 'M1')
	m2_data_folder = os.path.join(data_folder, 'M2')
	if not os.path.isdir(m2_data_folder):
		os.mkdir(m2_data_folder)
	f = open(log_file, 'w')
	# key terms
	laterality_word_set = ['right', 'left', 'Right', 'Left', 'Rt', 'Lt', 'rt', 'lt', 'RT', 'LT', 'isthmus', 'ISTHMUS']
	location_word_set = ['inf', 'sup', 'mid', 'medial', 'med', 'ant', 'inferior', 'superior', 'posterior', 'anterior']
	malignant_diagnosis_word_set = ['papillary', 'carcinoma', 'malignant', 'malignancy']
	benign_diagnosis_word_set = ['benign', 'negative']
	irrelevant_diagnosis_word_set = ['follicular', 'hurthle cell neoplasm', 'nondiagnositic', 'descriptive diagnosis']
	non_malignant_word_set = benign_diagnosis_word_set + irrelevant_diagnosis_word_set
	non_benign_word_set = malignant_diagnosis_word_set 
	# get nodule info
	print('Get nodule info...')
	with open(os.path.join(m1_data_folder, 'selected_benign_mrn.txt'), 'r') as f0:
		content = f0.readlines()
	benign_selected_mrns = [x.strip() for x in content]
	df_b, freq_table_b = return_laterality_numnod(benign_path_df, benign_selected_mrns, laterality_word_set, \
		location_word_set, benign_diagnosis_word_set, non_benign_word_set, log_file_b)
	df_b['ReportSite'].value_counts().to_csv(os.path.join(m2_data_folder, 'sites_benign.csv'))
	df_b.to_csv(os.path.join(m2_data_folder, 'raw_lsa_res_b.csv'), index=False)
	# post-process
	print('Post-processing nodule info...')
	benign_lat_file = os.path.join(m2_data_folder, 'raw_lsa_res_b.csv')
	benign_lat_df = pd.read_csv(benign_lat_file)
	lat_final_df_b = benign_lat_df[~benign_lat_df.ReportSite.str.contains('child')]
	children_b = benign_lat_df[benign_lat_df.ReportSite.str.contains('child')]
	f.write('number of all patients: {}, number of patients that are pediatric: {}'.format(len(np.unique(benign_lat_df.MRN)), len(np.unique(children_b.MRN))))
	children_b.to_csv(os.path.join(m2_data_folder, 'pediatric_patients_b.csv'), index=False)
	benign_lat_res = get_lat_res(lat_final_df_b)
	benign_lat_res.to_csv(os.path.join(m2_data_folder, 'process_lsa_res_b.csv'), index=False)
	f.close()
	return 0

def process_M1(png_folder, data_folder, path_report_file, log_folder):
	# configs
	homedir = os.path.expanduser('~')
	log_file = os.path.join(log_folder, 'M1_summary.txt')
	m1_data_folder = os.path.join(data_folder, 'M1')
	if not os.path.isdir(m1_data_folder):
		os.mkdir(m1_data_folder)
	benign_path_df = path_report_file
	f = open(log_file, 'w')
	# folder names
	diagnostic_folder_names = ['US-SOFT-TISSUE-HEAD-AND-NECK']
	biopsy_folder_names = ['US-THYROID-NON-CYST-FINE-NEEDLE-ASPIRATION', 'US-THYROID-BIOPSY', 'US-FNA-THYROID-IMAGES']
	pathology_folder_names = ['US-THYROID-IMAGES']
	study_folder_names = diagnostic_folder_names + biopsy_folder_names + pathology_folder_names
	# get and match mrn <-> study
	benign_mrns = return_folder_name(png_folder)
	df_study_b = get_number_studies(png_folder, benign_mrns, diagnostic_folder_names, biopsy_folder_names, pathology_folder_names)
	df_study_b.to_csv(os.path.join(m1_data_folder, 'number_of_studies_benign.csv'), index=False)
	benign_no_study_list = list(df_study_b.MRN[df_study_b.NumberTotStudy == 0])
	benign_selected_mrns = [x for x in benign_mrns if x not in benign_no_study_list]
	with open(os.path.join(m1_data_folder, 'selected_benign_mrn.txt'), 'w') as f0:
		for item in benign_selected_mrns:
			f0.write('%s\n' % item)
	f.write('number of original mrns: {}, selected mrns: {}\n'.format(len(np.unique(benign_mrns)), len(np.unique(benign_selected_mrns))))
	benign_selected_df = benign_path_df[['MRN', 'Encounter Identifier', 'Report Date']][benign_path_df['MRN'].astype(str).isin(benign_selected_mrns)]
	selected_benign_paths_diag, selected_benign_datetime_diag = get_dir_path(benign_selected_df, png_folder, diagnostic_folder_names)
	selected_benign_paths_biopsy, selected_benign_datetime_biopsy = get_dir_path(benign_selected_df, png_folder, biopsy_folder_names)
	selected_benign_paths_path, selected_benign_datetime_path = get_dir_path(benign_selected_df, png_folder, pathology_folder_names)
	selected_benign_mrnenc = get_mrnenc(benign_selected_df, benign_selected_mrns)
	# additional info
	benign_selected_df['MRN_Enc'] = selected_benign_mrnenc
	benign_selected_df['BiopsyStudyDate'] = selected_benign_datetime_biopsy
	benign_selected_df['BiopsyStudyDir'] = selected_benign_paths_biopsy
	benign_selected_df['DiagnosticStudyDate'] = selected_benign_datetime_diag
	benign_selected_df['DiagnosticStudyDir'] = selected_benign_paths_diag
	benign_selected_df['PathologyStudyDate'] = selected_benign_datetime_path
	benign_selected_df['PathologyStudyDir'] = selected_benign_paths_path
	# post-process and save 
	cols = list(benign_selected_df.columns)
	a, b = cols.index('Report Date'), cols.index('MRN_Enc')
	cols[b], cols[a] = cols[a], cols[b]
	benign_selected_df = benign_selected_df[cols]
	benign_selected_df.to_csv(os.path.join(m1_data_folder, 'selected_directories_benign.csv'), index=False)

	f.close()
	return 0

if __name__ == "__main__":
	# set up directories
	root_src = '..' 
	# raw images folder
	png_folder = os.path.join(root_src, 'benign')
	# MADLaP generated data while processing
	data_folder = os.path.join(root_src, 'MADLaP_results/data')
	# MADLaP yields folder
	results_folder = os.path.join(root_src, 'MADLaP_results/sel_pngs')
	# MADLaP log files folder
	log_folder = os.path.join(root_src, 'MADLaP_results/logs_b')
	# spreadsheet that saves pathology reports
	path_report_file = pd.read_csv('...')
	# spreadsheet that saves radiology reports
	rad_report_file = pd.reas_csv('...')
	# file that contains accession numbers for all downloaded radiology studies
	accn_file = '...'
	# thresholds for caliper detections scores (0.945) and cropping out image to prevent rulers (0.87)
	cd_thres = [0.945, 0.87]
	main(png_folder, data_folder, results_folder, log_folder, path_report_file, rad_report_file, accn_file, cd_thres)
