import matplotlib.image as mpimg
import numpy as np
import cv2
from skimage.feature import hog
from scipy.ndimage.measurements import label
import time
import copy

# Define a function to return HOG features and visualization
def get_hog(img, **kwargs):
    '''
    Get HOG feature of an image 
    :param img: a single channel image i.e H x W
    :param kwargs: parameter settings
    :return: 
        if vis = True: return features, hog_image
        else: return features
    '''
    # hog parameters
    orient = kwargs['orient']
    pix_per_cell = kwargs['pix_per_cell']
    cell_per_block = kwargs['cell_per_block']
    hog_channel = kwargs['hog_channel']

    # optional parameter
    vis = kwargs.get('vis', False)
    feature_vec = kwargs.get('feature_vec', True)

    # Call with two outputs if vis==True
    if hog_channel == 'ALL':
        hog_feats = []
        hog_imgs  = []
        for hog_c in range(3):
            output = hog(img[:,:, hog_c],
                        block_norm='L1-sqrt',
                        orientations=orient,
                        pixels_per_cell=(pix_per_cell, pix_per_cell),
                        cells_per_block=(cell_per_block, cell_per_block),
                        transform_sqrt=True,
                        visualise=vis,
                        feature_vector=feature_vec)
            if vis:
                hog_feats.extend(output[0])
                hog_imgs.extend(output[1])
            else:
                hog_feats.extend(output)

        if vis:
            return hog_feats, hog_imgs
        else:
            return hog_feats
    else:
        return  hog(img[:,:, hog_channel],
                block_norm='L1-sqrt',
                orientations=orient,
                pixels_per_cell=(pix_per_cell, pix_per_cell),
                cells_per_block=(cell_per_block, cell_per_block),
                transform_sqrt=True,
                visualise=vis, feature_vector=feature_vec)

name2cspace = {'HSV': cv2.COLOR_RGB2HSV,
               'LUV': cv2.COLOR_RGB2LUV,
               'HLS': cv2.COLOR_RGB2HLS,
               'YUV': cv2.COLOR_RGB2YUV,
               'YCrCb': cv2.COLOR_RGB2YCrCb}

# default hog-channel is 0 exception for HSV/HLS
cspace2chog = {'HSV': 2,
               'HLS': 1}

# Define a function to compute binned color features
def get_bin_spatial(img, **kwargs):
    spatial_size = kwargs.get('spatial_size', (32,32))
    # Use cv2.resize().ravel() to create the feature vector
    # Return the feature vector
    return cv2.resize(img, spatial_size).ravel()


# Define a function to compute color histogram features
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def get_color_hist(img, **kwargs):
    nbins = kwargs.get('nbins', 32)
    bins_range = kwargs.get('bins_range', (0.0, 1.0))
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0])).astype(np.float32)
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

name2feats = {'bin_spatial': get_bin_spatial,
              'color_hist': get_color_hist,
              'hog': get_hog}

def read_img(imgfn, color_space='RGB'):
    img = mpimg.imread(imgfn)
    if color_space != 'RGB':
        img = cv2.cvtColor(img, name2cspace[color_space])
    return img

def scaled_img(img, color_space='RGB'):
    if color_space == 'HSV' or color_space == 'HLS' or color_space == 'LUV':
        img = np.copy(img)
        if color_space == 'LUV':
            img[:, : ,0] /= 100
            img[:, :, 1] = (img[:, :, 1] + 134) / 354
            img[:, :, 2] = (img[:, :, 2] + 140) / 362
        else:
            img[:, :, 0] /= 360

    return img

def convert_img(img, color_space='RGB', scaled=True):
    if color_space != 'RGB':
        img = cv2.cvtColor(img, name2cspace[color_space])

    if scaled:
        return scaled_img(img, color_space)
    else:
        return img

def single_img_feature(img,
                       **kwargs):
    color_space = kwargs['color_space']
    feats = kwargs['feats']
    scaled = kwargs['scaled']

    img_features = []
    # apply color conversion if other than 'RGB'
    img = convert_img(img, color_space, scaled)

    for feat in feats:
        img_features.append(name2feats[feat](img, **kwargs[feat]))
    return  np.concatenate(img_features)

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs,
                     flags,
                     combine_setting,
                     color_space='RGB',
                     feats = ['bin_spatial', 'color_hist', 'hog'],
                     scaled = True):
    kwargs = copy.deepcopy(combine_setting)
    kwargs['color_space'] = color_space
    kwargs['feats']   = feats
    kwargs['scaled'] = scaled

    # set default hog-channel if not exist
    if 'hog' in kwargs:
        if kwargs['hog'].get('hog_channel') is None:
            kwargs['hog']['hog_channel'] = cspace2chog.get(color_space, 0)

    # Create a list to append feature vectors to
    features = []
    targets = []
    ts = time.time()
    # Iterate through the list of images
    for file,flag in zip(imgs, flags):
        # read in each one by one
        image = mpimg.imread(file)
        features.append(single_img_feature(image, **kwargs))
        flip_image = cv2.flip(image, 1)
        features.append(single_img_feature(flip_image, **kwargs))
        targets.extend([flag, flag])

    print('Build feature {} cost {:.2f} second(s)'.format(feats, time.time() - ts))
    # Return list of feature vectors
    return np.array(features), np.array(targets), kwargs


# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
def slide_window(img_shape, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img_shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img_shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0] * (xy_overlap[0]))
    ny_buffer = np.int(xy_window[1] * (xy_overlap[1]))
    nx_windows = np.int((xspan - nx_buffer) / nx_pix_per_step)
    ny_windows = np.int((yspan - ny_buffer) / ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]

            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=2, color_lastbox=False):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for i, bbox in enumerate(bboxes):
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, tuple(bbox[0]), tuple(bbox[1]), color, thick)

    if color_lastbox:
        cv2.rectangle(imcopy, tuple(bbox[0]), tuple(bbox[1]), (0,0,0), thickness=6)
    # Return the image copy with boxes drawn
    return imcopy

def search_windows(img, windows, clf, **kwargs):
    if (img.dtype == np.uint8):
        img = img.astype(np.float32)/255

    on_windows = []
    for window in windows:
        sub_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        features = single_img_feature(sub_img, **kwargs)
        pred = clf.predict(features)
        if pred == 1:
            on_windows.append(window)
    return on_windows

def add_heat(heatmap, bboxes):
    for bbox in bboxes:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]] += 1

def heat_map_bbox(img, bboxes, threshold=3, debug=False):
    heatmap = np.zeros_like(img, dtype=np.float32)

    # build heatmap
    add_heat(heatmap, bboxes)

    # apply threshold
    heatmap[heatmap <= threshold] = 0

    # Visualize the heatmap when displaying
    heatmap = np.clip(heatmap, 0, 255)

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

    if debug:
        return hm_bboxes, heatmap
    else:
        return hm_bboxes


