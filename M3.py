################################################################
# Functions for M3 - caliper detection
# Code was partially adapted and modified based on Mateusz's paper
################################################################

import csv
import os
import sys
from glob import glob

import numpy as np
import png
import tensorflow as tf
from PIL import Image
from medpy.filter.binary import largest_connected_component
from scipy.ndimage.morphology import binary_dilation

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import shutil
from tqdm import tqdm


# This is needed since the notebook is stored in the object_detection folder.
# download object_detection from online sources
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Get some helper functions
from utils import *

# Configurations
PATH_TO_LABELS = "./model/label_map.pbtxt"
PATH_TO_CKPT = "./model/frozen_inference_graph.pb"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if len(sys.argv) == 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

NUM_CLASSES = 1
min_score_thresh = 0.5
overlay_bboxes = True

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v2.io.gfile.GFile(PATH_TO_CKPT, "rb") as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name="")

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True
)
category_index = label_map_util.create_category_index(categories)

############################################################################
## load_image_into_numpy_array
## numpy2png
## fname2pid
## run_inference_for_single_image
##    were adapted to the following github repository
## get_model was also adapted, but with modifications
############################################################################

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def numpy2png(image, path):
    with open(path, "wb") as pngfile:
        pngWriter = png.Writer(
            image.shape[1], image.shape[0], greyscale=False, alpha=False, bitdepth=8
        )
        pngWriter.write(pngfile, np.reshape(image, (-1, np.prod(image.shape[1:]))))


def fname2pid(fname):
    return fname.split("/")[-1].split(".")[0].lstrip("0")


def run_inference_for_single_image(image, sess):
    # Get handles to input and output tensors
    ops = tf.compat.v1.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in [
        "num_detections",
        "detection_boxes",
        "detection_scores",
        "detection_classes",
        "detection_masks",
    ]:
        tensor_name = key + ":0"
        if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.compat.v1.get_default_graph().get_tensor_by_name(tensor_name)
    if "detection_masks" in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict["detection_boxes"], [0])
        detection_masks = tf.squeeze(tensor_dict["detection_masks"], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict["num_detections"][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(
            detection_masks, [0, 0, 0], [real_num_detection, -1, -1]
        )
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1]
        )
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.0), tf.uint8
        )
        # Follow the convention by adding back the batch dimension
        tensor_dict["detection_masks"] = tf.expand_dims(detection_masks_reframed, 0)
    image_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name("image_tensor:0")

    # Run inference
    output_dict = sess.run(
        tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)}
    )

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict["num_detections"] = int(output_dict["num_detections"][0])
    output_dict["detection_classes"] = output_dict["detection_classes"][0].astype(
        np.uint8
    )
    output_dict["detection_boxes"] = output_dict["detection_boxes"][0]
    output_dict["detection_scores"] = output_dict["detection_scores"][0]
    if "detection_masks" in output_dict:
        output_dict["detection_masks"] = output_dict["detection_masks"][0]
    return output_dict


def get_cd_model(dir_paths, results_folder, f):
    """
        Segment calipers from the images. Return visualization of caliper detection and caliper scores

        Parameters:
            dir_paths: directories that saved images
            results_folder: destination folder
            detection_graph: tensorflow pre-trained model

    """
    for dir_path in tqdm(dir_paths):

        if (pd.isnull(dir_path) == True) or (len(dir_path) < 10):
            continue
        MRN = dir_path.split('/')[-2]
        PATH_TO_MRN = os.path.join(results_folder, MRN)
        if not os.path.isdir(PATH_TO_MRN):
            os.mkdir(PATH_TO_MRN)
        
        study_name = dir_path.split('/')[-1]
        PATH_TO_SAVE_CSV = os.path.join(results_folder, MRN, study_name + '-Calipers-test')
        if not os.path.isdir(PATH_TO_SAVE_CSV):
            os.mkdir(PATH_TO_SAVE_CSV)
        PATH_TO_SAVE_IMG = os.path.join(results_folder, MRN, study_name + '-Nodules-test-bboxes')
        if not os.path.isdir(PATH_TO_SAVE_IMG):
            os.mkdir(PATH_TO_SAVE_IMG)

        REGEX_FOR_PREPROCESSED_INPUT_IMAGES = os.path.join(dir_path, '*.png')
        TEST_IMAGE_PATHS = glob(REGEX_FOR_PREPROCESSED_INPUT_IMAGES)
        if len(TEST_IMAGE_PATHS) == 0:
            print("No images in {}".format(REGEX_FOR_PREPROCESSED_INPUT_IMAGES))
            exit(1)

        with detection_graph.as_default():
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            with tf.compat.v1.Session(config=config) as sess:
                for image_path in TEST_IMAGE_PATHS:
                    image = Image.open(image_path)
                    size_kb = os.path.getsize(image_path) / 1024 
                    if size_kb < 10.0:
                        continue
                    # the array based representation of the image will be used later in order to prepare the
                    # result image with boxes and labels on it.
                    image_np = load_image_into_numpy_array(image)
                    x_pad_size = 0
                    if image_np.shape[0] - image_np.shape[1] > 200:
                        # print("padding {}".format(image_path))
                        x_pad_size = int(0.1 * image_np.shape[1])
                        image_np = np.pad(
                            image_np, ((0, 0), (x_pad_size, x_pad_size), (0, 0)), "constant"
                        )
                    # Actual detection.
                    output_dict = run_inference_for_single_image(image_np, sess)

                    # get bounding box for the ROI
                    image_gs = np.mean(image_np, axis=-1)
                    image_bw = np.greater(image_gs, 3.0)
                    image_bw = binary_dilation(
                        image_bw, np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
                    )
                    image_th = largest_connected_component(image_bw)
                    cc_indices = np.argwhere(image_th)
                    y_min_th = float(np.min(cc_indices[:, 0])) / image_np.shape[0]
                    y_max_th = float(np.max(cc_indices[:, 0])) / image_np.shape[0]
                    x_min_th = float(np.min(cc_indices[:, 1])) / image_np.shape[1]
                    x_max_th = float(np.max(cc_indices[:, 1])) / image_np.shape[1]

                    # filter the result
                    detection_boxes = []
                    detection_scores = []
                    detection_classes = []
                    for i in range(output_dict["num_detections"]):
                        score = output_dict["detection_scores"][i]
                        # filter for score
                        if score > min_score_thresh:
                            # and filter for points outside of the ROI
                            bbox = output_dict["detection_boxes"][i]
                            y = (bbox[0] + bbox[2]) / 2
                            x = (bbox[1] + bbox[3]) / 2
                            if y < y_min_th or y > y_max_th or x < x_min_th or x > x_max_th:
                                continue
                            # and filter for overlapping points
                            if any(
                                [
                                    (db[0] <= y <= db[2] and db[1] <= x <= db[3])
                                    for db in detection_boxes
                                ]
                            ):
                                continue
                            # and filter for aspect ratio
                            bbox_h = int((bbox[2] - bbox[0]) * image_np.shape[0])
                            bbox_w = int((bbox[3] - bbox[1]) * image_np.shape[1])
                            if bbox_h > 2 * bbox_w:
                                # print("aspect ratio {}".format(image_path))
                                continue
                            # and filter for bbox size
                            min_dim = np.min([bbox_h, bbox_w])
                            if int(min_dim) > 24:
                                # print("size {}".format(image_path))
                                continue
                            detection_boxes.append(output_dict["detection_boxes"][i])
                            detection_scores.append(output_dict["detection_scores"][i])
                            detection_classes.append(output_dict["detection_classes"][i])
                    f.write('MRN: {}, Study Name: {}, Image Name: {} \n'.format(MRN, study_name, image_path.split('/')[-1]))
                    f.write('Image size: {} Detection scores: {} \n'.format(image.size, detection_scores))
                    if len(detection_boxes) < 2:
                        f.write("below 0.5 threshold {} \n".format(image_path))
                    f.write('================================================= \n')
                    
                    det_index = -1
                    while len(detection_boxes) < 2:
                        det_index += 1
                        if det_index >= len(output_dict["detection_boxes"]):
                            break
                        # again filter for points outside of the ROI
                        bbox = output_dict["detection_boxes"][det_index]
                        y = (bbox[0] + bbox[2]) / 2
                        x = (bbox[1] + bbox[3]) / 2
                        if y < y_min_th or y > y_max_th or x < x_min_th or x > x_max_th:
                            continue
                        # and again filter for overlapping points
                        if any(
                            [
                                (db[0] <= y <= db[2] and db[1] <= x <= db[3])
                                for db in detection_boxes
                            ]
                        ):
                            continue
                        # and filter for aspect ratio
                        bbox_h = int((bbox[2] - bbox[0]) * image_np.shape[0])
                        bbox_w = int((bbox[3] - bbox[1]) * image_np.shape[1])
                        if bbox_h > 2 * bbox_w:
                            continue
                        # and again filter for bbox size
                        min_dim = np.min([bbox_h, bbox_w])
                        if int(min_dim) > 24:
                            continue
                        detection_boxes.append(output_dict["detection_boxes"][det_index])
                        detection_scores.append(output_dict["detection_scores"][det_index])
                        detection_classes.append(output_dict["detection_classes"][det_index])

                    # Visualization of the results of a detection.
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        np.array(detection_boxes),
                        np.array(detection_classes),
                        np.array(detection_scores),
                        category_index,
                        instance_masks=output_dict.get("detection_masks"),
                        use_normalized_coordinates=True,
                        line_thickness=1,
                        min_score_thresh=0.0,
                    )

                    img_result_path = os.path.join(
                        PATH_TO_SAVE_IMG, os.path.split(image_path)[-1]
                    )
                    csv_result_path = os.path.join(
                        PATH_TO_SAVE_CSV, os.path.split(image_path)[-1]
                    )
                    csv_result_path = csv_result_path.replace("png", "csv")

                    numpy2png(image_np[:, x_pad_size : -x_pad_size - 1, :], img_result_path)
                    yaxis, xaxis, scores = [], [], []
                    for i in range(len(detection_boxes)):
                        bbox = detection_boxes[i]
                        ymin = int(bbox[0] * image_np.shape[0])
                        xmin = int(bbox[1] * image_np.shape[1]) - x_pad_size
                        ymax = int(bbox[2] * image_np.shape[0])
                        xmax = int(bbox[3] * image_np.shape[1]) - x_pad_size
                        yaxis.append((ymin + ymax) / 2)
                        xaxis.append((xmin + xmax) / 2)
                        scores.append(detection_scores[i])
                    cur_data_df = pd.DataFrame({
                        'y-axis': yaxis,
                        'x-axis': xaxis,
                        'detection_scores': scores
                        })
                    cur_data_df.to_csv(csv_result_path, index=False)

def get_imgs_detected_calipers(csv_paths, image_path, threshold):
    """
        Get images (sliced on the left) with number of calipers equal to 2 or 4

        Parameters:
            csv_paths: directories that saved scores from "get_model"
            image_path: original image path
            threshold: caliper score

        Returns:
            target_img_names: qualified images
    """
    target_img_names = []
    for csv_file in glob(csv_paths):
        img_name = csv_file.split('/')[-1].split('.')[-2]
        filename = img_name + '.png'
        img_path = os.path.join(image_path, filename)
        xdim, ydim = Image.open(img_path).size
        caliper_score, crop_percent = threshold[0], threshold[1]
        cur_csv = pd.read_csv(csv_file)
        crop_x = xdim * crop_percent
        scores = cur_csv[cur_csv['x-axis'] < crop_x].detection_scores
        num_calipers = sum(scores > caliper_score)
        if (num_calipers == 2) or (num_calipers == 4):
            target_img_names.append(img_name)
    return target_img_names

def copy_selected_imgs(dir_paths, results_folder, save_folder, threshold):
    """
        Copy selected images to the destination folder

        Parameters:
            dir_paths, results_folder: directories associated with saved scores
            save_folder: destination folder for copied images
            threshold: caliper score

    """

    for dir_path in tqdm(dir_paths):
        if type(dir_path) is float:
            continue
        
        if len(dir_path) <10:
            continue
        MRN = dir_path.split('/')[-2]
        PATH_TO_MRN = os.path.join(save_folder, MRN)
        if not os.path.isdir(PATH_TO_MRN):
            os.mkdir(PATH_TO_MRN)
        study_name = dir_path.split('/')[-1]
        PATH_TO_SAVE_RESULTS = os.path.join(results_folder, MRN, study_name+ '-Calipers-test')
        PATH_TO_CSV_FILES = os.path.join(PATH_TO_SAVE_RESULTS, '*.csv')
        selected_imgs = get_imgs_detected_calipers(PATH_TO_CSV_FILES, dir_path, threshold)
        if len(selected_imgs) > 0:
            selected_img_file = [x + '.png' for x in selected_imgs]
            selected_img_fileloc = [os.path.join(dir_path, x) for x in selected_img_file]
            PATH_TO_SAVE_SELECTED_IMGS = os.path.join(save_folder, MRN, study_name+ '-Calipers-selected')
            if not os.path.isdir(PATH_TO_SAVE_SELECTED_IMGS):
                os.mkdir(PATH_TO_SAVE_SELECTED_IMGS)
            selected_dst_fileloc = [os.path.join(PATH_TO_SAVE_SELECTED_IMGS, x) for x in selected_img_file]
        else:
            continue
        assert len(selected_dst_fileloc) == len(selected_img_fileloc)
        for img_fileloc, dst_fileloc in zip(selected_img_fileloc, selected_dst_fileloc):
            shutil.copyfile(img_fileloc, dst_fileloc)