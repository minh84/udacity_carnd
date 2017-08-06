import numpy as np
import cv2
import time

from collections import deque
from moviepy.editor import VideoFileClip

from utils import binary_threshold, warper, measure_curvature, measure_offset
from lane_lines import find_lane_line_hist, find_lane_line_conv, draw_lane_area

def pipeline_single_image(raw_rgb, mtx, dist):
    undist_img = cv2.undistort(raw_rgb, mtx, dist, None, mtx)

    undist_binary = binary_threshold(undist_img)
    binary_wraped = warper(undist_binary)

    hist_lr_fits = find_lane_line_hist(binary_wraped, 9)
    lane_line_img = draw_lane_area(undist_img, binary_wraped, hist_lr_fits)
    return lane_line_img

def pipeline_video_naive(video_file, mtx, dist, out_file):
    clip = VideoFileClip(video_file)
    process_image = lambda raw_rgb : pipeline_single_image(raw_rgb, mtx, dist)
    ts = time.time()
    clip_lane_line = clip.fl_image(process_image)

    clip_lane_line.write_videofile(out_file, audio=False)
    print('File is saved to {}'.format(out_file))


class LaneLine(object):
    '''
    We implement LaneLine object so that we can 
        1)  keep track of things like last several:    
                lane-line detection
                curvature was
        2) apply smoothing on lane-line detection by moving-average
        3) do sanity check: 
                lane-lines are roughly parallel (separated by approximately the right distance horizontally)
                check if curvature is similar
    '''
    def __init__(self, n_last = 5):

        # x = A*y^2 + B*y + C, we keep track of n last fitted coefficient
        self._fitA = deque(n_last)
        self._fitB = deque(n_last)
        self._fitC = deque(n_last)

        # average coefficients
        self._avgA = None
        self._avgB = None
        self._avgC = None

        # difference in fit coefficients between last and new fits
        self._diffs = np.array([0., 0., 0.], dtype=np.float64)

        # radius of curvature
        self._radius_of_curvature = None

