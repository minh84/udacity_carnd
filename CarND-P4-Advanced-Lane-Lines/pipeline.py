import cv2
import time
import numpy as np

from moviepy.editor import VideoFileClip

from utils import binary_threshold_v1, binary_threshold_v2, warper, measure_curvature, measure_offset
from lane_lines import find_lane_line_hist, find_lane_line_conv, draw_lane_area, draw_curvrad_carpos, LaneLine

def pipeline_single_image(raw_rgb, mtx, dist, bin_fn = binary_threshold_v1):
    '''
    A pipeline from raw rgb to road-image with 
        * lane-line detected
        * curvature-radius measured
        * car-position measured
    :param raw_rgb: 
    :param mtx: 
    :param dist: 
    :param bin_fn: 
    :return: 
    '''
    undist_img = cv2.undistort(raw_rgb, mtx, dist, None, mtx)

    undist_binary = bin_fn(undist_img)
    binary_warped = warper(undist_binary)

    # find lane-line and draw it on road image
    hist_lr_fits = find_lane_line_hist(binary_warped, 9)
    lane_line_img = draw_lane_area(undist_img, binary_warped, hist_lr_fits)

    img_H, img_W = lane_line_img.shape[:2]

    # measure curvature-radius
    l_radius, r_radius = measure_curvature(img_H, hist_lr_fits)
    radius = (int)((l_radius + r_radius)/2)

    # measure car position with respect to center of the lane
    clane_base_pos = measure_offset(img_H, img_W, hist_lr_fits)

    # draw info on road image
    draw_curvrad_carpos(lane_line_img, radius, clane_base_pos)

    return lane_line_img

def pipeline_video_naive(video_file, mtx, dist, out_file, bin_fn = binary_threshold_v1):
    clip = VideoFileClip(video_file)
    process_image = lambda raw_rgb : pipeline_single_image(raw_rgb, mtx, dist, bin_fn)
    clip_lane_line = clip.fl_image(process_image)

    clip_lane_line.write_videofile(out_file, audio=False)
    print('File is saved to {}'.format(out_file))

def pipeline_img_with_memory(raw_rgb, mtx, dist, left_lane, right_lane, bin_fn = binary_threshold_v2):
    undist_img = cv2.undistort(raw_rgb, mtx, dist, None, mtx)

    undist_binary = bin_fn(undist_img)
    binary_warped = warper(undist_binary)

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])  # y is heigh dimension => first dimension
    nonzerox = np.array(nonzero[1])  # x is width dimension => second dimension

    l_fit = left_lane.search(binary_warped, nonzerox, nonzeroy)
    r_fit = right_lane.search(binary_warped, nonzerox, nonzeroy)

    # draw lane-line
    lane_line_img = draw_lane_area(undist_img, binary_warped, (l_fit, r_fit))

    radius = (int) ((left_lane._radius_of_curvature + right_lane._radius_of_curvature)/2)
    clane_base_pos = (left_lane._line_base_pos - right_lane._line_base_pos)/2

    # draw info on road image
    draw_curvrad_carpos(lane_line_img, radius, clane_base_pos)
    return lane_line_img

def pipeline_video_look_ahead(video_file, mtx, dist, out_file, bin_fn = binary_threshold_v2):
    clip = VideoFileClip(video_file)
    left_lane = LaneLine('left')
    right_lane = LaneLine('right')

    process_image = lambda raw_rgb: pipeline_img_with_memory(raw_rgb, mtx, dist, left_lane, right_lane, bin_fn)
    clip_lane_line = clip.fl_image(process_image)

    clip_lane_line.write_videofile(out_file, audio=False)
    print('File is saved to {}'.format(out_file))