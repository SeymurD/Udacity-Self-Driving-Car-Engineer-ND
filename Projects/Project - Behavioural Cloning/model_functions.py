import numpy as np
import cv2

from sklearn.utils import shuffle


def displayCV2(img):
    '''
    Utility method to display a CV2 Image
    '''
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_img_for_visualization(image, angle, pred_angle, frame):
    '''
    Used by visualize_dataset method to format image prior to displaying. Converts colorspace back to original BGR, applies text to display steering angle and frame number (within batch to be visualized), and applies lines representing steering angle and model-predicted steering angle (if available) to image.
    '''
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)
    img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    h, w = img.shape[0:2]
    # apply text for frame number and steering angle
    cv2.putText(img, 'frame: ' + str(frame), org=(2, 18), fontFace=font, fontScale=.5, color=(200, 100, 100),
                thickness=1)
    cv2.putText(img, 'angle: ' + str(angle), org=(2, 33), fontFace=font, fontScale=.5, color=(200, 100, 100),
                thickness=1)
    # apply a line representing the steering angle
    cv2.line(img, (int(w / 2), int(h)), (int(w / 2 + angle * w / 4), int(h / 2)), (0, 255, 0), thickness=4)
    if pred_angle is not None:
        cv2.line(img, (int(w / 2), int(h)), (int(w / 2 + pred_angle * w / 4), int(h / 2)), (0, 0, 255), thickness=4)
    return img


def visualize_dataset(X, y, y_pred=None):
    '''
    format the data from the dataset (image, steering angle) and display
    '''
    for i in range(len(X)):
        if y_pred is not None:
            img = process_img_for_visualization(X[i], y[i], y_pred[i], i)
        else:
            img = process_img_for_visualization(X[i], y[i], None, i)
        name_dir = './report_images/batch_vis_' + i + '.jpg'
        cv2.imwrite(name_dir, img)


def preprocess_image(img):
    '''
    Method for preprocessing images: this method is the same used in drive.py, except this version uses
    BGR to YUV and drive.py uses RGB to YUV (due to using cv2 to read the image here, where drive.py images are
    received in RGB)
    '''
    # original shape: 160x320x3, input shape for neural net: 66x200x3
    # crop to 105x320x3
    # new_img = img[35:140,:,:]
    # crop to 40x320x3
    new_img = img[50:140, :, :]
    # apply subtle blur
    new_img = cv2.GaussianBlur(new_img, (3, 3), 0)
    # scale to 66x200x3 (same as nVidia)
    new_img = cv2.resize(new_img, (200, 66), interpolation=cv2.INTER_AREA)
    # scale to ?x?x3
    # new_img = cv2.resize(new_img,(80, 10), interpolation = cv2.INTER_AREA)
    # convert to YUV color space (as nVidia paper suggests)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2YUV)
    return new_img


def process_img_for_video(image, angle, pred_angle, frame):
    '''
    Used by visualize_dataset method to format image prior to adding to video
    '''
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)
    img = cv2.resize(img,None,fx=3, fy=3, interpolation = cv2.INTER_CUBIC)
    h,w = img.shape[0:2]
    # apply text for frame number and steering angle
    cv2.putText(img, 'frame: ' + str(frame), org=(2,18), fontFace=font, fontScale=.5, color=(255,255,255), thickness=1)
    cv2.putText(img, 'angle: ' + str(angle), org=(2,33), fontFace=font, fontScale=.5, color=(255,255,255), thickness=1)
    # apply a line representing the steering angle
    cv2.line(img,(int(w/2),int(h)),(int(w/2+angle*w/4),int(h/2)),(0,255,0),thickness=4)
    if pred_angle is not None:
        cv2.line(img,(int(w/2),int(h)),(int(w/2+pred_angle*w/4),int(h/2)),(0,0,255),thickness=4)
    return img


def random_distort(img, angle):
    '''
    method for adding random distortion to dataset images, including random brightness adjust, and a random
    vertical shift of the horizon position
    '''
    new_img = img.astype(float)
    # random brightness - the mask bit keeps values from going beyond (0,255)
    value = np.random.randint(-28, 28)
    if value > 0:
        mask = (new_img[:, :, 0] + value) > 255
    if value <= 0:
        mask = (new_img[:, :, 0] + value) < 0
    new_img[:, :, 0] += np.where(mask, 0, value)
    # random shadow - full height, random left/right side, random darkening
    h, w = new_img.shape[0:2]
    mid = np.random.randint(0, w)
    factor = np.random.uniform(0.6, 0.8)
    if np.random.rand() > .5:
        new_img[:, 0:mid, 0] *= factor
    else:
        new_img[:, mid:w, 0] *= factor
    # randomly shift horizon
    h, w, _ = new_img.shape
    horizon = 2 * h / 5
    v_shift = np.random.randint(-h / 8, h / 8)
    pts1 = np.float32([[0, horizon], [w, horizon], [0, h], [w, h]])
    pts2 = np.float32([[0, horizon + v_shift], [w, horizon + v_shift], [0, h], [w, h]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    new_img = cv2.warpPerspective(new_img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return (new_img.astype(np.uint8), angle)


def generate_training_data(image_paths, angles, batch_size=128, validation_flag=False):
    '''
    method for the model training data generator to load, process, and distort images, then yield them to the
    model. if 'validation_flag' is true the image is not distorted. also flips images with turning angle magnitudes of greater than 0.33, as to give more weight to them and mitigate bias toward low and zero turning angles
    '''
    image_paths, angles = shuffle(image_paths, angles)
    X, y = ([], [])
    while True:
        for i in range(len(angles)):
            img = cv2.imread(image_paths[i])
            angle = angles[i]
            img = preprocess_image(img)
            if not validation_flag:
                img, angle = random_distort(img, angle)
            X.append(img)
            y.append(angle)
            if len(X) == batch_size:
                yield (np.array(X), np.array(y))
                X, y = ([], [])
                image_paths, angles = shuffle(image_paths, angles)
            # flip horizontally and invert steer angle, if magnitude is > 0.33
            if abs(angle) > 0.33:
                img = cv2.flip(img, 1)
                angle *= -1
                X.append(img)
                y.append(angle)
                if len(X) == batch_size:
                    yield (np.array(X), np.array(y))
                    X, y = ([], [])
                    image_paths, angles = shuffle(image_paths, angles)


def generate_training_data_for_visualization(image_paths, angles, batch_size=20, validation_flag=False):
    '''
    method for loading, processing, and distorting images
    if 'validation_flag' is true the image is not distorted
    '''
    X = []
    y = []
    image_paths, angles = shuffle(image_paths, angles)
    for i in range(batch_size):
        img = cv2.imread(image_paths[i])
        angle = angles[i]
        img = preprocess_image(img)
        if not validation_flag:
            img, angle = random_distort(img, angle)
        X.append(img)
        y.append(angle)
    return (np.array(X), np.array(y))
