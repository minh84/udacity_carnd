import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import warper_inv, measure_curvature, measure_offset

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

    # Compute the curvature-radius & car position
    l_radius, r_radius = measure_curvature(img_H, left_right_fits)
    pos_offset = measure_offset(img_H, img_W, left_right_fits)
    radius = (int)((l_radius+r_radius)/2)

    if (pos_offset > 1e-2):
        left_or_right = ' right '
    elif (pos_offset < -1e-2):
        left_or_right = ' left '
    else:
        left_or_right = ' '

    # Combine the result with the original image
    img = cv2.addWeighted(in_img, 1, newwarp, 0.3, 0)
    cv2.putText(img,
                'Radius of Curvature = {:5d}(m)'.format(radius),
                (30, 30), cv2.FONT_HERSHEY_COMPLEX,
                1.0, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(img,
                'Vehical is of {:.2f}m{}of center'.format(np.abs(pos_offset), left_or_right),
                (30, 80), cv2.FONT_HERSHEY_COMPLEX,
                1.0, (255, 255, 255), 2, cv2.LINE_AA)

    return img