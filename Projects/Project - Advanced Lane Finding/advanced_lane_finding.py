import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import config

from moviepy.editor import VideoFileClip
from scipy import stats
from lane_functions import *

# Initialize global variables
config.falseCount = 0           # Checks for false positive histogram transform calls
config.curveDist = 0            # Distance between fitted left and right curves
config.videoType = 'project'    # Options: project, challenge, harder_challenge

# Set image directories for calibration of camera
imageDir = 'camera_cal/calibration*.jpg'
testCalibImage = 'test_images/straight_lines1.jpg'
# Set image directory for test images
testLaneImages = 'test_images/test*.jpg'

# Calibrate Camera
undist, mtx, dist = calibrate_camera(imageDir, testCalibImage)

# Setup clip initialization
clip = VideoFileClip("test_videos/project_video.mp4")

# Run initial frame configuration
firstFrame = clip.get_frame(0)

# Apply undistortion, warp and gradient operations
gradImageFirst, vertices = grad_image(firstFrame)
warpImageFirst, warp_mtx = warp_perspective(gradImageFirst, mtx, dist, vertices, 'forward')
# Search for lane pixels and fit first polyline
resultFirst = search_around_poly(warpImageFirst, None, None, True)


# Define process function in main py file to avoid passing above
# calibration steps to lane_functions.py
def process_image(image):
    # Apply undistortion, warp and gradient operations
    gradImage, vertices = grad_image(image)
    warpImage, warp_mtx = warp_perspective(gradImage, mtx, dist, vertices, 'forward')

    # Use previous line to find lane pixels and fit polyline
    laneCurvesWarp = search_around_poly(warpImage, config.left_fit_global, config.right_fit_global, False)

    # Reverse warped image
    laneCurves, _ = warp_perspective(laneCurvesWarp, mtx, dist, vertices, 'reverse')

    # Overlay onto original image
    overlayImage = cv2.addWeighted(image, 1, laneCurves, 0.3, 0)

    concatImage = cv2.hconcat([overlayImage, laneCurvesWarp])

    # Measure Curvature
    left_R_curve, right_R_curve, offset = measure_curvature_real()
    if (left_R_curve > 5000) and (right_R_curve > 5000):
        textMeasure = 'Vehicle is moving along a straight line'
    else:
        textMeasure = 'Left Lane Radius: ' + str(round(left_R_curve, 2)) + \
                    '(m), Right Lane Radius: ' + str(round(right_R_curve, 2)) + '(m)'
    addCurveVal = cv2.putText(concatImage, textMeasure, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), thickness=2)
    result = cv2.putText(addCurveVal, offset, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), thickness=2)

    return result


# Run moviepy on remaining frames where first frame is redundant
white_output = 'test_videos_output/project_final.mp4'
white_clip = clip.fl_image(process_image)  # NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)
