import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import deque

from utils import warper_inv, measure_curvature, measure_offset, curvature_radius

def find_lane_line_hist(binary_warped, nwindows, margin=100, minpix=50, debug=False):
    '''
    Find lane line using histogram
    :param binary_warped: 
    :param nwindows: 
    :param margin: 
    :param minpix: 
    :param debug: 
    :return: 
    '''
    H, W = binary_warped.shape
    half_H = H // 2
    half_W = W // 2

    # histogram of the half-bottom
    half_bottom_hist = np.sum(binary_warped[half_H:, :], axis=0)

    # taking the peak as starting points
    leftx_base = np.argmax(half_bottom_hist[:half_W])
    rightx_base = np.argmax(half_bottom_hist[half_W:]) + half_W

    # height of sliding window
    window_H = H // nwindows

    # create empty lists to receive left and right lane pixel indices and sliding windows
    l_lane_idx = []
    r_lane_idx = []

    if debug:
        l_windows = []
        r_windows = []

    # set lane-line postion, then update for each window
    leftx_pos = leftx_base
    rightx_pos = rightx_base

    # identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])  # y is heigh dimension => first dimension
    nonzerox = np.array(nonzero[1])  # x is width dimension => second dimension

    # slide through windows
    for iw in range(nwindows):
        # identify window boundaries
        win_y_low = H - (iw + 1) * window_H
        win_y_high = H - iw * window_H

        win_xleft_low = leftx_pos - margin
        win_xleft_high = leftx_pos + margin

        win_xright_low = rightx_pos - margin
        win_xright_high = rightx_pos + margin

        # store window-points
        if debug:
            l_windows.append([(win_xleft_low, win_y_low), (win_xleft_high, win_y_high)])
            r_windows.append([(win_xright_low, win_y_low), (win_xright_high, win_y_high)])

        # identify the nonzero pixels in x and y within the window (stored as index)
        nonzero_win_left = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & \
                            (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]

        nonzero_win_right = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & \
                             (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # append these indices to the lists
        l_lane_idx.append(nonzero_win_left)
        r_lane_idx.append(nonzero_win_right)

        # re-adjust the position when you found enough pixels in a window
        if len(nonzero_win_left) > minpix:
            leftx_pos = np.int(np.mean(nonzerox[nonzero_win_left]))
        if len(nonzero_win_right) > minpix:
            rightx_pos = np.int(np.mean(nonzerox[nonzero_win_right]))

    # concatenate of indices
    l_lane_idx = np.concatenate(l_lane_idx)
    r_lane_idx = np.concatenate(r_lane_idx)

    # extract left and right lane pixel position
    leftx = nonzerox[l_lane_idx]
    lefty = nonzeroy[l_lane_idx]
    rightx = nonzerox[r_lane_idx]
    righty = nonzeroy[r_lane_idx]

    # finally fit a second order polynomial to each: note that we fit x = quadratic(y)
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # return fitted curves, left-points, right-points and windows
    if debug:
        return (left_fit, right_fit), ((lefty, leftx), (righty, rightx)), (l_windows, r_windows)
    else:
        return (left_fit, right_fit)


def find_ceintroids(binary_warped, nwindows, window_width=50, margin=100):
    H, W = binary_warped.shape
    window_height = H // nwindows

    window_centroids = []
    window = np.ones(window_width)

    # get starting points: using 1/4 bottom part
    bottom_H = 3 * H // 4
    half_ww = window_width // 2
    midpoint = H // 2

    l_sum = np.sum(binary_warped[bottom_H:, :midpoint], axis=0)
    l_center = np.argmax(np.convolve(window, l_sum)) - half_ww
    r_sum = np.sum(binary_warped[bottom_H:, midpoint:], axis=0)
    r_center = np.argmax(np.convolve(window, r_sum)) - half_ww + midpoint

    window_centroids.append((l_center, r_center))

    for iw in range(1, nwindows):
        # get image layer
        win_y_low = H - (iw + 1) * window_height
        win_y_high = H - iw * window_height
        image_layer = np.sum(binary_warped[win_y_low:win_y_high, :], axis=0)

        # compute convolution
        conv_signal = np.convolve(window, image_layer)

        # find the best left/right centroid by using past left/right as a reference
        # we assume new centroid has difference to the past one less than margin
        l_min_idx = max(l_center + half_ww - margin, 0)
        l_max_idx = min(l_center + half_ww + margin, W)

        r_min_idx = max(r_center + half_ww - margin, 0)
        r_max_idx = min(r_center + half_ww + margin, W)

        # update centroid
        l_center = np.argmax(conv_signal[l_min_idx:l_max_idx]) + l_min_idx - half_ww
        r_center = np.argmax(conv_signal[r_min_idx:r_max_idx]) + r_min_idx - half_ww

        # append new centroid
        window_centroids.append((l_center, r_center))
    return window_centroids


def find_lane_line_conv(binary_warped, nwindows, window_width=50, margin=100, debug=False):
    '''
    find lane line using convolution + sliding window
    :param binary_warped: 
    :param nwindows: 
    :param window_width: 
    :param margin: 
    :param debug: 
    :return: 
    '''
    # get centroids
    H, W = binary_warped.shape

    window_centroids = find_ceintroids(binary_warped, nwindows, window_width=window_width, margin=margin)

    window_height = H // nwindows
    half_ww = window_width // 2

    # create empty lists to receive left and right lane pixel indices and sliding windows
    l_lane_idx = []
    r_lane_idx = []

    if debug:
        l_windows = []
        r_windows = []

    # identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])  # y is heigh dimension => first dimension
    nonzerox = np.array(nonzero[1])  # x is width dimension => second dimension

    for iw in range(nwindows):
        l_center, r_center = window_centroids[iw]

        win_y_low = H - (iw + 1) * window_height
        win_y_high = H - iw * window_height

        win_xleft_low = l_center - half_ww
        win_xleft_high = l_center + half_ww

        win_xright_low = r_center - half_ww
        win_xright_high = r_center + half_ww

        # store window-points
        if debug:
            l_windows.append([(win_xleft_low, win_y_low), (win_xleft_high, win_y_high)])
            r_windows.append([(win_xright_low, win_y_low), (win_xright_high, win_y_high)])

        # identify the nonzero pixels in x and y within the window (stored as index)
        nonzero_win_left = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & \
                            (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]

        nonzero_win_right = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & \
                             (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # append these indices to the lists
        l_lane_idx.append(nonzero_win_left)
        r_lane_idx.append(nonzero_win_right)

    # concatenate of indices
    l_lane_idx = np.concatenate(l_lane_idx)
    r_lane_idx = np.concatenate(r_lane_idx)

    # extract left and right lane pixel position
    leftx = nonzerox[l_lane_idx]
    lefty = nonzeroy[l_lane_idx]
    rightx = nonzerox[r_lane_idx]
    righty = nonzeroy[r_lane_idx]

    # finally fit a second order polynomial to each: note that we fit x = quadratic(y)
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # return fitted curves, left-points, right-points and windows
    if debug:
        return (left_fit, right_fit), ((lefty, leftx), (righty, rightx)), (l_windows, r_windows)
    else:
        return (left_fit, right_fit)


def visualize_lane_line(binary_warped,
                        left_right_fits,
                        left_right_points,
                        left_right_windows,
                        title=None):
    '''
    this helper function 
    :param binary_warped: 
    :param left_right_fits: 
    :param left_right_points: 
    :param left_right_windows: 
    :return: 
    '''
    # create color image for the visualization
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

    # mark left point by red-color and right points by blue-color
    lefty, leftx = left_right_points[0]
    righty, rightx = left_right_points[1]
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # draw windows
    left_windows, right_windows = left_right_windows
    nwindows = len(left_windows)
    for iw in range(nwindows):
        cv2.rectangle(out_img, left_windows[iw][0], left_windows[iw][1], (0, 255, 0), 2)
        cv2.rectangle(out_img, right_windows[iw][0], right_windows[iw][1], (0, 255, 0), 2)

    # draw the fitted curve
    left_fit, right_fit = left_right_fits
    ypoints = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_xpoints = left_fit[0] * ypoints ** 2 + left_fit[1] * ypoints + left_fit[2]
    right_xpoints = right_fit[0] * ypoints ** 2 + right_fit[1] * ypoints + right_fit[2]

    # plot it
    f, ax = plt.subplots(1, 1, figsize=(14, 7), sharex=True)
    ax.imshow(out_img)
    ax.plot(left_xpoints, ypoints, color='yellow')
    ax.plot(right_xpoints, ypoints, color='yellow')
    ax.set_xlim(0, binary_warped.shape[1])
    ax.set_ylim(binary_warped.shape[0], 0)
    if title is not None:
        ax.set_title(title, fontsize=20)

def draw_lane_area(in_img, binary_warped, left_right_fits):
    '''
    This function draw lane-line on road image
    :param in_img: 
    :param binary_warped: 
    :param left_right_fits: 
    :return: 
    '''
    img_H, img_W = binary_warped.shape

    # left/right lane-lines
    left_fit, right_fit = left_right_fits
    ypoints = np.linspace(0, img_H - 1, img_H)
    left_xpoints = left_fit[0] * ypoints ** 2 + left_fit[1] * ypoints + left_fit[2]
    right_xpoints = right_fit[0] * ypoints ** 2 + right_fit[1] * ypoints + right_fit[2]

    # create an image to draw the lane-line area on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_xpoints, ypoints]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_xpoints, ypoints])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = warper_inv(color_warp)

    # Combine the result with the original image
    return cv2.addWeighted(in_img, 1, newwarp, 0.3, 0)

def draw_curvrad_carpos(img, radius, clane_base_pos):
    '''
    This function draws measuring information on road image
    :param img: 
    :param radius: 
    :param clane_base_pos: 
    :return: 
    '''

    if (clane_base_pos > 1e-2):
        left_or_right = ' right '
    elif (clane_base_pos < -1e-2):
        left_or_right = ' left '
    else:
        left_or_right = ' '

    cv2.putText(img,
                'Radius of Curvature = {:5d}(m)'.format(radius),
                (30, 30), cv2.FONT_HERSHEY_COMPLEX,
                1.0, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(img,
                'Vehical is of {:.2f}m{}of center'.format(np.abs(clane_base_pos), left_or_right),
                (30, 80), cv2.FONT_HERSHEY_COMPLEX,
                1.0, (255, 255, 255), 2, cv2.LINE_AA)

    return img


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

    def __init__(self,
                 left_or_right='left',
                 n_last=5,
                 margin=100,
                 minpix=50,
                 nwindows=9,
                 img_WH=(1280,720),
                 xpix2m=3.7/700,
                 ypix2m=30.0/720):

        # x = A*y^2 + B*y + C, we keep track of n last fitted coefficient
        self._fitA = deque(maxlen=n_last)
        self._fitB = deque(maxlen=n_last)
        self._fitC = deque(maxlen=n_last)

        # store search parameter
        self._margin   = margin
        self._minpix   = minpix
        self._nwindows = nwindows
        self._img_WH   = img_WH

        # parameter to convert pixel to meter
        self._xpix2m = xpix2m
        self._ypix2m = ypix2m

        # is left lane or right lane (need this for blind search)
        self._left_or_right = left_or_right

        # average coefficients
        self._avgFit = None

        # radius of curvature
        self._radius_of_curvature = None

        # distance in meters of vehicle center from the line
        self._line_base_pos = None

    def window_search(self,
                     binary_warped,
                     nonzerox,
                     nonzeroy):

        H, W = binary_warped.shape
        half_H = H // 2
        half_W = W // 2

        # histogram of the half-bottom
        if self._left_or_right == 'left':
            half_bottom_hist = np.sum(binary_warped[half_H:, :half_W], axis=0)
            x_pos = np.argmax(half_bottom_hist)
        else:
            half_bottom_hist = np.sum(binary_warped[half_H:, half_W:], axis=0)
            x_pos = np.argmax(half_bottom_hist) + half_W

        # taking the peak as starting points

        # height of sliding window
        window_H = H // self._nwindows

        # sliding window histogram search
        lane_idx = []

        for iw in range(self._nwindows):
            # identify window boundaries
            win_y_low = H - (iw + 1) * window_H
            win_y_high = H - iw * window_H

            win_x_low  = x_pos - self._margin
            win_x_high = x_pos + self._margin


            # identify the nonzero pixels in x and y within the window (stored as index)
            nonzero_win = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & \
                           (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]

            # append these indices to the lists
            lane_idx.append(nonzero_win)

            # re-adjust the position when you found enough pixels in a window
            if len(nonzero_win) > self._minpix:
                x_pos = np.int(np.mean(nonzerox[nonzero_win]))

        # concatenate found points
        lane_idx = np.concatenate(lane_idx)

        # extract points of interest
        x = nonzerox[lane_idx]
        y = nonzeroy[lane_idx]

        # finally fit a second order polynomial to each: note that we fit x = quadratic(y)
        fit = np.polyfit(y, x, 2)

        # return fitted polynomial
        return fit

    def average_search(self,
                       nonzerox,
                       nonzeroy):
        if self._avgFit is None:
            raise Exception('Expect average fit coefficient, found None')

        # define region of interest using previous average fits
        nonzero_curve = self._avgFit[0] * (nonzeroy**2) + self._avgFit[1] * nonzeroy + self._avgFit[2]
        lane_idx = ((nonzerox > nonzero_curve - self._margin) & (nonzerox < nonzero_curve + self._margin))

        if len(lane_idx) > 30:
            # extract points of interest
            x = nonzerox[lane_idx]
            y = nonzeroy[lane_idx]

            # finally fit a second order polynomial to each: note that we fit x = quadratic(y)
            fit = np.polyfit(y, x, 2)

            # return fitted polynomial
            return True, fit
        else:
            return False, None

    def update_fit(self, new_fit):
        # update parameters
        self._fitA.append(new_fit[0])
        self._fitB.append(new_fit[1])
        self._fitC.append(new_fit[2])
        self._avgFit[0] = np.mean(self._fitA)
        self._avgFit[1] = np.mean(self._fitB)
        self._avgFit[2] = np.mean(self._fitC)

    def get_measure(self, fit):
        # now we know the fit parameter we will compute
        #       lane-curvature & position of the car
        A_in_m = self._xpix2m * fit[0] / (self._ypix2m ** 2)
        B_in_m = self._xpix2m * fit[1] / self._ypix2m
        yval = self._img_WH[1]
        radius_of_curvature = curvature_radius(yval * self._ypix2m, A_in_m, B_in_m)

        # compute position of the car
        xval = fit[0] * (yval ** 2) + fit[1] * yval + fit[2]
        pts = np.array([[[xval, yval]]], dtype=np.float64)
        orig_pts = warper_inv(pts)

        offset_pixel = orig_pts[0][0][0] - self._img_WH[0] / 2
        line_base_pos = offset_pixel * self._xpix2m

        return  radius_of_curvature, line_base_pos

    def search(self, binary_warped, nonzerox, nonzeroy):
        if self._avgFit is None:
            fit = self.window_search(binary_warped, nonzerox, nonzeroy)

            self._avgFit = fit
            self._fitA.append(fit[0])
            self._fitB.append(fit[1])
            self._fitC.append(fit[2])
        else:
            ret, fit = self.average_search(nonzerox, nonzeroy)
            # if not succesfull, switch back to blind-search
            if not ret:
                fit = self.window_search(binary_warped, nonzerox, nonzeroy)

            self.update_fit(fit)

        self._radius_of_curvature, self._line_base_pos = self.get_measure(fit)

        return fit