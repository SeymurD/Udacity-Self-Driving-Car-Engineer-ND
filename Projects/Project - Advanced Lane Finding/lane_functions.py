import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import config
from scipy import stats


def calibrate_camera(image_dir, test_img):
    '''
    The following function takes in the list of calibration images and performs
    a calibration using the provided images.
    :param image_dir: directory of calibration images
    :param test_img: directory of test image
    :return: undistorted image, distortion matrix and distance coefficients
    '''
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(image_dir)

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            # cv2.drawChessboardCorners(img, (9, 6), corners, ret)
            # write_name = 'corners_found'+str(idx)+'.jpg'
            # cv2.imwrite(write_name, img)
            # cv2.imshow('img', img)
            # cv2.waitKey(0)
    # Load test image
    img = cv2.imread(test_img)
    img_size = (img.shape[1], img.shape[0])

    # Perform calibration and undistort the test_image
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    # Save undistorted test image for comparison
    cv2.imwrite('camera_cal/test_undist_project.jpg', undist)

    return undist, mtx, dist


def color_space_transform(image, h_threshold=(0, 179), l_threshold=(0, 255), s_threshold=(0, 255), r_threshold=(0, 255),
                          sx_thresh=(10, 100), sy_thresh=(10, 100), sxy_thresh=(10, 100)):
    '''
    Applies color space transformation of the original image from BGR to HLS using thresholded parameters
    from each individual HLS channel. Combination of Lightness and Saturation are used and then masked with
    a Sobel X operator. Sobel Y is commuted along with the magnitude of Sobel X and Y together for user flexibility.
    :param image:
    :param h_threshold: Hue threshold
    :param l_threshold: Lightness threshold
    :param s_threshold: Saturation threshold
    :param r_threshold: Red channel threshold
    :param sx_thresh: Sobel X threshold
    :param sy_thresh: Sobel Y threshold
    :param sxy_thresh: Magnitude of Sobel X & Y threshold
    :return: binary_h, binary_l, binary_s, sxbinary, sybinary, sxy, l_s, ls_sx
    '''
    # Transform to HLS space
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    h_channel = hls[:, :, 0]
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    binary_h = np.zeros_like(h_channel)
    binary_l = np.zeros_like(l_channel)
    binary_s = np.zeros_like(s_channel)

    binary_h[(h_channel > h_threshold[0]) & (h_channel <= h_threshold[1])] = 1
    binary_l[(l_channel > l_threshold[0]) & (l_channel <= l_threshold[1])] = 1
    binary_s[(s_channel > s_threshold[0]) & (s_channel <= s_threshold[1])] = 1

    l_s = cv2.bitwise_or(binary_l, binary_s)  # Combine hue and lightness

    # Isolate single red-channel
    r_channel = image[:, :, 2]
    binary_r = np.zeros_like(r_channel)
    binary_r[(r_channel > r_threshold[0]) & (r_channel <= r_threshold[1])] = 1

    # -------------------------------Sobel X-----------------------------------
    sobelx = cv2.Sobel(r_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobelx = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Apply low-pass filter before binary thresholding
    scaled_sobelx = cv2.boxFilter(scaled_sobelx, -1, (7, 7), normalize=True)
    # scaled_sobelx = cv2.bilateralFilter(scaled_sobelx, 9, 255, 150)
    # scaled_sobelx = cv2.dilate(scaled_sobelx, (3, 3))
    # scaled_sobelx = cv2.boxFilter(scaled_sobelx, -1, (3, 3), normalize=True)

    # Threshold x-gradient
    sxbinary = np.zeros_like(scaled_sobelx)
    sxbinary[(scaled_sobelx >= sx_thresh[0]) & (scaled_sobelx <= sx_thresh[1])] = 1

    # -------------------------------Sobel Y-----------------------------------
    sobely = cv2.Sobel(r_channel, cv2.CV_64F, 0, 1)  # Take the derivative in y
    abs_sobely = np.absolute(sobely)  # Absolute y derivative to accentuate lines away from horizontal
    scaled_sobely = np.uint8(255 * abs_sobely / np.max(abs_sobely))

    # Threshold y-gradient
    sybinary = np.zeros_like(scaled_sobely)
    sybinary[(scaled_sobely >= sy_thresh[0]) & (scaled_sobely <= sy_thresh[1])] = 1

    # ------------------------------Sobel Mag----------------------------------
    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)
    sxy = np.zeros_like(gradmag)
    sxy[(gradmag >= sxy_thresh[0]) & (gradmag <= sxy_thresh[1])] = 1

    # --------------------------Bitwise Operations-----------------------------
    ls_sx = cv2.bitwise_and(l_s, l_s, mask=sxbinary)  # Combine hue, lightness, Sobel X

    return binary_h, binary_l, binary_s, sxbinary, sybinary, sxy, l_s, ls_sx


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def grad_image(image):
    '''
    Applies corresponding gradient operations and thresholds based on the video project inputted.
    :param image: Image after camera calibration
    :return: processImg, vertices
    '''
    imshape = image.shape

    # image = region_of_interest(image, vertices) # Optional: ROI crop

    # Filter space transforms based on the challenge videos
    if config.videoType == 'project':
        vertices = np.array([[(200, imshape[0]),  # bottom-left
                              (imshape[1] // 2 - 75, 470),  # top-left
                              (imshape[1] // 2 + 80, 470),  # top-right
                              (imshape[1] - 150, imshape[0])]], dtype=np.int32)  # bottom-right
        _, _, s_img, sx, _, _, _, _ = color_space_transform(image, s_threshold=(90, 255), sx_thresh=(15, 255))
        processImg = np.dstack((np.zeros_like(sx), s_img, sx)) * 255
        processImg = cv2.cvtColor(processImg, cv2.COLOR_RGB2GRAY)
        # processImg = np.dstack((processImg, processImg, processImg)) # Optional: Stack channel to draw
    elif config.videoType == 'challenge':
        vertices = np.array([[(200, imshape[0]),  # bottom-left
                              (imshape[1] // 2 - 75, 470),  # top-left
                              (imshape[1] // 2 + 80, 470),  # top-right
                              (imshape[1] - 125, imshape[0])]], dtype=np.int32)  # bottom-right
        _, _, s_img, sx, _, _, _, _ = color_space_transform(image, s_threshold=(90, 255), sx_thresh=(15, 255))
        processImg = np.dstack((np.zeros_like(sx), s_img, sx)) * 255
        processImg = cv2.cvtColor(processImg, cv2.COLOR_RGB2GRAY)
    elif config.videoType == 'harder_challenge':
        vertices = np.array([[(100, imshape[0]),  # bottom-left
                              (imshape[1] // 2 - 250, 520),  # top-left
                              (imshape[1] // 2 + 250, 520),  # top-right
                              (imshape[1] - 100, imshape[0])]], dtype=np.int32)  # bottom-right

        _, _, _, _, _, _, _, lsx = color_space_transform(image, l_threshold=(110, 255), s_threshold=(50, 255),
                                                         r_threshold=(90, 255), sx_thresh=(10, 100))
        processImg = np.dstack((lsx, lsx, lsx)) * 255
        processImg = cv2.cvtColor(processImg, cv2.COLOR_RGB2GRAY)

    return processImg, vertices


def warp_perspective(img, mtx, dist, vertices, warpDirection):
    '''
    Creates a 'birds-eye-view' perspective in order to perform curve fitting of the frame and undistorts image.
    :param img: Binary image
    :param mtx: Camera matrix
    :param dist: Distortion Coefficients
    :param vertices: Vertices of polyFill that capture just the lane lines (manually set)
    :param warpDirection: Used for warping and unwarping the view
    :return: warped, M
    '''
    # Undistort image
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    img_size = (img.shape[1], img.shape[0])
    # Set source and destination corners for transform
    # Source is same as ROI vertices
    src = np.float32([vertices[0][2], vertices[0][3], vertices[0][0], vertices[0][1]])
    # Destination corners
    dst = np.float32([[img.shape[1] - 300, 0], [img.shape[1] - 300, img.shape[0]], [300, img.shape[0]], [300, 0]])

    # Get perspective matrix
    M = cv2.getPerspectiveTransform(src, dst)

    # Warp the image
    if warpDirection == 'forward':
        warped = cv2.warpPerspective(undist, M, img_size, flags=cv2.INTER_LINEAR)
    elif warpDirection == 'reverse':
        warped = cv2.warpPerspective(undist, M, img_size, flags=cv2.WARP_INVERSE_MAP)
    # cv2.rectangle(warped, (300, 0), (img.shape[1]-300, img.shape[0]), (255, 0, 0), thickness=5) Optional: Visualize

    return warped, M


def find_lane_init(binary_warped):
    # Take histogram of binary image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    if config.videoType == 'harder_challenge':
        histogram[histogram < 10000] = 0
        histogram[:200] = 0
        histogram[1000:] = 0
    # Stack channels to draw on image
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Check to see if spacing is too far
    if ((rightx_base - leftx_base) > 530) and (config.videoType == 'harder_challenge'):
        rightx_base = leftx_base + 530

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 50
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        bOptimize = False
        while not bOptimize:
            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                          (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low),
                          (win_xright_high, win_y_high), (0, 255, 0), 2)

            # Identify the nonzero pixels in x and y within the window #
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            testMargin = 5
            test_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_high) & (nonzerox < (win_xleft_high + testMargin))).nonzero()[0]
            test_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= (win_xright_low - testMargin)) & (nonzerox < win_xright_low)).nonzero()[0]

            if len(test_right_inds) > 40:
                win_xright_low -= testMargin
                win_xright_high -= testMargin
                good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                                   (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            elif len(test_left_inds) > 40:
                win_xleft_low += testMargin
                win_xleft_high += testMargin
                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                                  (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            else:
                bOptimize = True

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_poly_init(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_init(binary_warped)

    try:
        # Fit a second order polynomial to each using `np.polyfit`
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
    except:
        # In areas where the vehicle takes extremely tight turns the lanes disappear
        # and if lighting conditions blow out the lane for a few frames we can keep
        # prior estimate to as the tight corners tend to maintain same curvaturee
        print('Polyfit did not work!')
        left_fit = config.left_fit_global
        right_fit = config.right_fit_global

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    try:
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1 * ploty ** 2 + 1 * ploty
        right_fitx = 1 * ploty ** 2 + 1 * ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')

    config.left_fit_global = left_fit
    config.right_fit_global = right_fit

    return out_img


def fit_poly(img_shape, leftx, lefty, rightx, righty):
    try:
        # Fit a second order polynomial to each using `np.polyfit`
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
    except:
        # In areas where the vehicle takes extremely tight turns the lanes disappear
        # and if lighting conditions blow out the lane for a few frames we can keep
        # prior estimate to as the tight corners tend to maintain same curvaturee
        print('Polyfit did not work!')
        left_fit = config.left_fit_global
        right_fit = config.right_fit_global

    # Check if A parameters are not significantly different
    if (abs(left_fit[0] - right_fit[0]) > 3e-04):
        print('Diff too big')
        left_fit = config.left_fit_global
        right_fit = config.right_fit_global

    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])
    # Calcualte both polynomials using ploty, left_fit and right_fit ###
    try:
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1 * ploty ** 2 + 1 * ploty
        right_fitx = 1 * ploty ** 2 + 1 * ploty

    # Evaluate delta of the curves
    d = right_fitx - left_fitx
    config.curveDist = np.mean(d)
    if config.videoType == 'harder_challenge':
        laneSizeLimit = 400
    else:
        laneSizeLimit = 500
    if np.mean(d) < laneSizeLimit:
        # Use previous curve if large noise forces curve to change rapidly
        left_fit = config.left_fit_global
        right_fit = config.right_fit_global
        try:
            left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
            right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            left_fitx = 1 * ploty ** 2 + 1 * ploty
            right_fitx = 1 * ploty ** 2 + 1 * ploty

    # Update global params
    config.left_fit_global = left_fit
    config.right_fit_global = right_fit

    return left_fitx, right_fitx, ploty


def search_around_poly(binary_warped, left_fit, right_fit, isFirst):
    # First image setup
    if isFirst:
        resultFirst = fit_poly_init(binary_warped)
        return resultFirst

    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    margin = 25

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Set the area of search based on activated x-values
    # within the +/- margin of our polynomial function
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) &
                      (nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) &
                       (nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

    # Monitor if noise is contributing to lane deviation
    bOptimize = False
    deltaMarginL = 0
    deltaMarginR = 0
    while not bOptimize:
        testMargin = 5
        test_left_lane_inds = (
                (nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[
                    2] + margin + deltaMarginL)) &
                (nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + (
                        margin + deltaMarginL + testMargin)))).nonzero()[0]
        test_right_lane_inds = ((nonzerox > (
                right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - (
                margin + deltaMarginR + testMargin))) &
                                (nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - (
                                        margin + deltaMarginR)))).nonzero()[0]

        if len(test_left_lane_inds) > 1000:
            deltaMarginL += testMargin
            left_lane_inds = (
                    (nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[
                        2] - margin + deltaMarginR)) &
                    (nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[
                        2] + margin + deltaMarginR)))
        if len(test_right_lane_inds) > 1000:
            deltaMarginR += testMargin
            right_lane_inds = (
                    (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[
                        2] - margin - deltaMarginR)) &
                    (nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[
                        2] + margin - deltaMarginR)))
        else:
            bOptimize = True

    # Check if incoming portion along curve has detected pixel values
    # Experimentally shown to occur when image is blown out
    bDetected = True
    if not bDetected:
        closest_left_count = (nonzeroy[left_lane_inds] < (0.2 * binary_warped.shape[0]))
        closest_right_count = (nonzeroy[left_lane_inds] < (0.2 * binary_warped.shape[0]))
        if (len(closest_left_count) < 10000) or (len(closest_right_count) < 10000):
            bDetected = False
            histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
            histogram[histogram < 15000] = 0
            histogram[:200] = 0
            histogram[1000:] = 0
            midpoint = np.int(histogram.shape[0] // 2)
            leftPoint = np.argmax(histogram[:midpoint])
            rightPoint = np.argmax(histogram[midpoint:]) + midpoint
            if (leftPoint > 200) and (rightPoint < 1000) \
                    and (stats.ks_1samp(histogram, stats.norm.cdf, alternative='greater')[0] > 0.49):
                config.falseCount += 1
                if config.falseCount > 2:
                    print('Entered hist')
                    config.falseCount = 0
                    fit_poly_init(binary_warped)

    if bDetected:
        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit new polynomials
        left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
    else:
        # Use old lines params
        # Use previous curve if large noise forces curve to change rapidly
        left_fit = config.left_fit_global
        right_fit = config.right_fit_global
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        try:
            left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
            right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            left_fitx = 1 * ploty ** 2 + 1 * ploty
            right_fitx = 1 * ploty ** 2 + 1 * ploty

    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    # left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    # left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
    # left_line_pts = np.hstack((left_line_window1, left_line_window2))
    #
    # right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    # right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
    # right_line_pts = np.hstack((right_line_window1, right_line_window2))

    left_line_window_wide = np.array([np.flipud(np.transpose(np.vstack([left_fitx - (margin // 8), ploty])))])
    right_line_window_wide = np.array([np.transpose(np.vstack([right_fitx + (margin // 8), ploty]))])
    left_right_combo_pts = np.hstack((right_line_window_wide, left_line_window_wide))

    # Draw the lane onto the warped blank image
    # cv2.fillPoly(window_img, np.int_([left_line_pts]), (255, 255, 0))
    # cv2.fillPoly(window_img, np.int_([right_line_pts]), (255, 255, 0))
    cv2.fillPoly(window_img, np.int_([left_right_combo_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.6, 0)

    # Plot the polynomial lines onto the image
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    ## End visualization steps ##

    return result


def measure_curvature_real():
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / config.curveDist  # meters per pixel in x dimension

    # Scale existing fitted curves in pixel space, instead of refitting:
    # x = mx(my**2)*a*(y**2) + (mx/my)*b*y + c
    # Therefore we just scale the value of A and B in the quadratic formula
    left_fit_cr = [(xm_per_pix / (ym_per_pix ** 2)) * config.left_fit_global[0],
                   (xm_per_pix / ym_per_pix) * config.left_fit_global[1], config.left_fit_global[2]]
    right_fit_cr = [(xm_per_pix / (ym_per_pix ** 2)) * config.right_fit_global[0],
                    (xm_per_pix / ym_per_pix) * config.right_fit_global[1], config.right_fit_global[2]]
    ploty = np.linspace(0, 719, num=720)

    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    # Calculation of offset from center
    egoCenter = 1280 // 2  # Midpoint along x-axis
    xEgoLeft = config.left_fit_global[0] * (y_eval ** 2) + config.left_fit_global[1] * y_eval + \
               config.left_fit_global[2]
    xEgoRight = config.right_fit_global[0] * (y_eval ** 2) + config.right_fit_global[1] * y_eval + \
                config.right_fit_global[2]
    offsetVal = (egoCenter - (((xEgoRight - xEgoLeft) // 2) + xEgoLeft)) * xm_per_pix
    if offsetVal < 0:
        offset = 'Vehicle is ' + str(abs(round(offsetVal, 2))) + '(m) right of center'
    elif offsetVal >= 0:
        offset = 'Vehicle is ' + str(abs(round(offsetVal, 2))) + '(m) left of center'

    return left_curverad, right_curverad, offset
