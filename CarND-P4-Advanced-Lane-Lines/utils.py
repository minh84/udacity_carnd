import numpy as np
import cv2
import matplotlib.pyplot as plt

def show_img(img, label, ax=None, cmap=None,figsize=(10,5)):
    '''
    this function show image & label
    '''
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    # draw image
    if label is not None:
        ax.text(5, -30, label, color='k', backgroundcolor='c', fontsize=10)

    ax.imshow(img, cmap=cmap)
    ax.axis('off')


def view_imgs(imgs, labels, figsize, multi_col=True, sharex=True, sharey=True):
    '''
    this function show images side by side
    :param imgs: 
    :param labels: 
    :param figsize: 
    :param multi_col: 
    :param sharex: 
    :param sharey: 
    :return: 
    '''
    nrows = 1 if multi_col else len(labels)
    ncols = len(labels) if multi_col else 1
    fig, axes = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols, sharey=sharex, sharex=sharey)
    for i, ax in enumerate(axes.flatten()):
        show_img(imgs[i], labels[i], ax)

def calibrate_camera(img_files, nx, ny):
    '''
    This function with calibrate camera using chessboard images
    :param img_files: image files
    :param nx: number of inner-corners along x-axis
    :param ny: number of inner-corners along y-axis
    :return:
        img_WH: width and height of image
        mtx:    camera matrix
        dist:   distortion coefficient
        rvecs:  rotation vectors
        tvecs:  translation vectors
    '''
    img0 = cv2.imread(img_files[0])
    H, W, _ = img0.shape
    img_WH  = (W, H)

    # create fixed object points
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    # load images and find corners
    obj_points = []
    img_points = []
    for img_file in img_files:
        bgr = cv2.imread(img_file)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        # print (gray.shape, gray.shape[::-1])
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if ret:
            img_points.append(corners)
            obj_points.append(objp)

    # now we can calibrate camera matrix/distortion coefficient
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points,
                                                       img_points,
                                                       img_WH,
                                                       None, None)
    if not ret:
        raise Exception('Calibrate camera FAILED, please check calib image files')

    return img_WH, mtx, dist, rvecs, tvecs


def abs_sobel_thresh(img_channel, sobel_kernel=3, orient='x', thresh=(0, 255)):
    '''
    absolute Sobel threshold in a specific direction
    args: 
        img_channel: is a channel of the image (e.g grayscale channel or S channel e.t.c)
    '''
    if orient == 'x':
        sobel = cv2.Sobel(img_channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(img_channel, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    sobel = np.abs(sobel)
    scaled_sobel = np.uint8(255. * sobel / np.max(sobel))
    binary_output = np.zeros_like(scaled_sobel)

    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return binary_output


def mag_thresh(img_channel, sobel_kernel=3, mag_thresh=(0, 255)):
    '''
    magniture threshold sqrt(sobolx^2 + soboly^2)
    args:
        img_channel: is a channel of the image
    '''
    sobelx = cv2.Sobel(img_channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img_channel, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    magn = np.sqrt(np.square(sobelx) + np.square(sobely))
    scaled_magn = np.uint8(255 * magn / np.max(magn))

    binary_output = np.zeros_like(magn)
    binary_output[(scaled_magn >= mag_thresh[0]) & (scaled_magn <= mag_thresh[1])] = 1

    return binary_output


def dir_thresh(img_channel, sobel_kernel=3, thresh=(0, np.pi / 2)):
    '''
    apply direction threshold
    '''
    sobelx = cv2.Sobel(img_channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img_channel, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    absgraddir = np.arctan2(np.abs(sobely), np.abs(sobelx))

    binary_output = np.zeros_like(absgraddir)

    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    return binary_output


def color_thresh(img_channel, thresh=(0, 255)):
    '''
    apply color threshold
    '''
    binary_output = np.zeros_like(img_channel)
    binary_output[(img_channel > thresh[0]) & (img_channel <= thresh[1])] = 1
    return binary_output

def binary_threshold(rgb_in, ksize=3):
    '''
    apply a combination of above threshold-functions to create a binary image from rgb-image
    :param rgb_in: 
    :param ksize: 
    :return: 
    '''
    gray = cv2.cvtColor(rgb_in, cv2.COLOR_RGB2GRAY)
    hls = cv2.cvtColor(rgb_in, cv2.COLOR_RGB2HLS)

    gradx = abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    grady = abs_sobel_thresh(gray, orient='y', sobel_kernel=ksize, thresh=(20, 100))
    mag_binary = mag_thresh(gray, sobel_kernel=ksize, mag_thresh=(30, 100))
    dir_binary = dir_thresh(gray, sobel_kernel=ksize, thresh=(0.7, 1.3))
    color_bin = color_thresh(hls[:, :, 2], thresh=(150, 255))

    combined = np.zeros_like(gray)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (color_bin == 1)] = 1
    return combined

src_points = np.float32([(205, 720), (595,450), (685,450), (1110,720)])
dst_points = np.float32([(300, 720), (300, 0),  (950, 0),  (950, 720)])
P_M = cv2.getPerspectiveTransform(src_points, dst_points)
P_Minv = cv2.getPerspectiveTransform(dst_points, src_points)

def warper(img_in):
    '''
    convert to bird-eye view
    :param img_in: 
    :return: 
    '''
    H, W = img_in.shape[:2]
    return cv2.warpPerspective(img_in, P_M, (W, H), flags=cv2.INTER_LINEAR)

def warper_inv(img_in):
    '''
    convert to bird-eye view
    :param img_in: 
    :return: 
    '''
    H, W = img_in.shape[:2]
    return cv2.warpPerspective(img_in, P_Minv, (W, H), flags=cv2.INTER_LINEAR)

def curvature_radius(yval, A, B, quad_eps=1e-5):
    return ((1.0 + (2.0*A*yval + B)**2)**1.5) / (2.0 * max(quad_eps, np.abs(A)))

def measure_curvature(img_H, lr_fits, ypix2m = 30./720, xpix2m=3.7/700):
    '''
    measure the curvature-radius from image-height, left-right fits
    :param img_H: 
    :param lr_fits: 
    :param ypix2m: 
    :param xpix2m: 
    :return: 
    '''
    # we need to convert pixel => m
    yval = img_H * ypix2m

    # x = ay^2 + by + c
    # => alpha * x = alpha * (a/(beta^2) * (y*beta)^2 + (b/beta)*(y*beta)+c)
    # => new_A = alpha * old_A / (beta^2), new_B = alpha * old_B / beta
    left_fit, right_fit = lr_fits
    l_A = xpix2m * left_fit[0] / (ypix2m**2)
    l_B = xpix2m * left_fit[1] / ypix2m
    r_A = xpix2m * right_fit[0] / (ypix2m ** 2)
    r_B = xpix2m * right_fit[1] / ypix2m

    l_radius = curvature_radius(yval, l_A, l_B)
    r_radius = curvature_radius(yval, r_A, r_B)
    return  l_radius, r_radius

def measure_offset(img_H, img_W, lr_fits, xpix2m = 3.7/700):
    '''
    
    :param img_H: 
    :param img_W: 
    :param lr_fits: 
    :param xpix2m: 
    :return: 
    '''
    yval     = img_H
    midpoint = img_W / 2
    left_fit, right_fit = lr_fits

    x_left  = left_fit[0] * yval**2 + left_fit[1] * yval + left_fit[2]
    x_right = right_fit[0] * yval**2 + right_fit[1] * yval + right_fit[2]
    xcenter = (x_left + x_right) / 2

    pts = np.array([[[xcenter, yval]]], dtype = np.float64)
    orig_pts = cv2.perspectiveTransform(pts, P_Minv)

    offset_pixel = orig_pts[0][0][0] - midpoint
    return offset_pixel * xpix2m