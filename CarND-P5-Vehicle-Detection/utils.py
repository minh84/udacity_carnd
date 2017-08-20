import glob
import os
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from moviepy.editor import VideoFileClip

def show_img(img, label=None, ax=None, cmap=None):
    '''
    this function show image & label
    '''
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    # draw image
    if label is not None:
        ax.text(0, 0, label, color='k', backgroundcolor='c', fontsize=10)

    ax.imshow(img, cmap=cmap)
    ax.axis('off')

def grid_view(imgs, labels, figsize, nrows, ncols, sharex=True, sharey=True, cmaps=None):
    fig, axes = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols, sharey=sharex, sharex=sharey)
    for i, ax in enumerate(axes.flatten()):
        cmap = None
        if cmaps is not None:
            cmap = cmaps[i]
        show_img(imgs[i], labels[i], ax, cmap=cmap)


def view_imgs(imgs, labels, figsize, multi_col=True, sharex=True, sharey=True):
    nrows = 1 if multi_col else len(labels)
    ncols = len(labels) if multi_col else 1
    grid_view(imgs, labels, figsize, nrows=nrows, ncols=ncols, sharex=sharex, sharey=sharey)


def video2images(video_file, output_dir):
    '''
    This function converts video into frame images (jpg format)
    It uses cv2.VideoCapture
    :param video_file: a mp4 video
    :param output_dir: where to save frame images
    :return: 
    '''
    if not os.path.isfile(video_file):
        raise Exception('Video file {} does NOT exist'.format(video_file))

    clip = VideoFileClip(video_file)
    frame_idx = 0
    ts = time.time()
    for frame in clip.iter_frames():
        frame_idx += 1
        mpimg.imsave('{}/frame_{:05d}.png'.format(output_dir, frame_idx), frame)
    te = time.time()
    print('{} frames are saved to {}, took {:.2f} seconds'.format(frame_idx, output_dir, te-ts))
    return frame_idx

def list_imgs(imgdirs, ext = 'png'):
    imgfiles = []
    for imgdir in imgdirs:
        fns = glob.glob('{}/*.{}'.format(imgdir, ext))
        imgfiles += list(np.sort(fns))
    return imgfiles
