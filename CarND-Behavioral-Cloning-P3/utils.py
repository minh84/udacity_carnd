import cv2

def crop(img, top=60, bottom=20):
    return img[top:-bottom, :, :]


def resize(img, width=200, height=66):
    return cv2.resize(img, (width, height), cv2.INTER_AREA)


def preprocess(img,
               crop_top=60,
               crop_bottom=20,
               resize_width=200,
               resize_height=66):
    # crop img
    img = crop(img, crop_top, crop_bottom)

    # resize img
    img = resize(img, resize_width, resize_height)

    return img