import numpy as np
from collections import deque
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label

from lesson_utils import search_windows, add_heat, heat_map_bbox, draw_boxes

def pipeline_single_img(img, windows, clf, **kwargs):
    car_windows = search_windows(img, windows, clf, **kwargs)
    car_hm_bboxes = heat_map_bbox(img, car_windows)
    return draw_boxes(img, car_hm_bboxes)

def pipeline_video(video_file, out_file, windows,  clf, **kwargs):
    clip = VideoFileClip(video_file)

    process_image = lambda raw_rgb: pipeline_single_img(raw_rgb, windows, clf, **kwargs)
    clip_lane_line = clip.fl_image(process_image)

    clip_lane_line.write_videofile(out_file, audio=False)
    print('File is saved to {}'.format(out_file))

class BoundingBoxes(object):
    def __init__(self, img_shape, max_frames = 6, threshold=3):
        self._img_shape     = img_shape
        self._frame_bboxes  = deque([], maxlen=max_frames)
        self._threshold     = threshold

    def append(self, bboxes):
        self._frame_bboxes.append(bboxes)

    def get_heatmap(self):
        heatmap = np.zeros(self._img_shape, dtype=np.float32)
        for frame_bboxes in self._frame_bboxes:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            for bbox in frame_bboxes:
                heatmap[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]] += 1

        # apply threshold
        heatmap[heatmap <= self._threshold] = 0

        # Visualize the heatmap when displaying
        heatmap = np.clip(heatmap, 0, 255)
        return heatmap

    def get_bboxes(self):
        heatmap = self.get_heatmap()

        # Find final boxes from heatmap using label function
        labels = label(heatmap)

        hm_bboxes = []

        # iterate through all detected cars
        for car_number in range(1, labels[1] + 1):
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            hm_bboxes.append(bbox)

        return hm_bboxes

def pipeline_single_img_memory(img, bboxes, windows, clf, **kwargs):
    car_windows = search_windows(img, windows, clf, **kwargs)
    # update new bounding boxes
    bboxes.append(car_windows)
    # integrate over frame
    car_hm_bboxes = bboxes.get_bboxes()

    return draw_boxes(img, car_hm_bboxes)

def pipeline_imgs(imgs, windows, clf, bboxes_setting, **kwargs):
    img_shape, max_frames, threshold = bboxes_setting
    bboxes = BoundingBoxes(img_shape, max_frames=max_frames, threshold=threshold)

    out_imgs = []
    cmaps    = []
    out_labels = []
    for i,img in enumerate(imgs):
        car_windows = search_windows(img, windows, clf, **kwargs)
        hm = np.zeros(img_shape, dtype=np.float32)
        add_heat(hm, car_windows)
        hm = np.clip(hm, 0, 255)
        out_imgs += [draw_boxes(img, car_windows, color=(0.0,0.0,1.0), thick=3), hm]
        cmaps += [None, 'hot']
        out_labels += ['frame {}'.format(i), 'heat-map frame {}'.format(i)]

        # update new bounding boxes
        bboxes.append(car_windows)

    out_imgs += [draw_boxes(imgs[-1], bboxes.get_bboxes(), thick=3), bboxes.get_heatmap()]
    cmaps += [None, 'hot']
    out_labels += ['last frame', 'heat-map last frame']
    return out_imgs, out_labels, cmaps


def pipeline_video_memory(video, out_file, windows, clf, bboxes_setting, **kwargs):
    if isinstance(video, str):
        clip = VideoFileClip(video)
    elif isinstance(video, VideoFileClip):
        clip = video


    img_shape, max_frames, threshold  = bboxes_setting
    bboxes = BoundingBoxes(img_shape, max_frames=max_frames, threshold=threshold)

    process_image = lambda raw_rgb: pipeline_single_img_memory(raw_rgb, bboxes, windows, clf, **kwargs)
    clip_lane_line = clip.fl_image(process_image)

    clip_lane_line.write_videofile(out_file, audio=False)
    print('File is saved to {}'.format(out_file))

