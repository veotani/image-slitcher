import cv2
import numpy as np
import pyram_blend
from time import time
from tqdm import tqdm

# Use the key points to stitch the images
def get_stitched_image(img1, img2, M):
    # Get width and height of input images
    w1, h1 = img1.shape[:2]
    w2, h2 = img2.shape[:2]

    # Get the canvases' dimensions
    img1_dims = np.float32([[0, 0], [0, w1], [h1, w1], [h1, 0]]).reshape(-1, 1, 2)
    img2_dims_temp = np.float32([[0, 0], [0, w2], [h2, w2], [h2, 0]]).reshape(-1, 1, 2)

    # Get relative perspective of second image
    img2_dims = cv2.perspectiveTransform(img2_dims_temp, M)

    # Resulting dimensions
    result_dims = np.concatenate((img1_dims, img2_dims), axis=0)

    # Getting images together
    # Calculate dimensions of match points
    [x_min, y_min] = np.int32(result_dims.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(result_dims.max(axis=0).ravel() + 0.5)

    # Create output array after affine transformation
    transform_dist = [-x_min, -y_min]
    transform_array = np.array([[1, 0, transform_dist[0]],
                                [0, 1, transform_dist[1]],
                                [0, 0, 1]])

    # Warp images to get the resulting image
    result_img = cv2.warpPerspective(img2, transform_array.dot(M),
                                     (x_max - x_min, y_max - y_min))

    only_img_2 = result_img.copy()
    only_img_1 = result_img.copy()
    only_img_1[:,:] = (0,0,0)
    only_img_1[transform_dist[1]:w1 + transform_dist[1],
    transform_dist[0]:h1 + transform_dist[0]] = img1

    # Return the result
    return only_img_1, only_img_2

# Find SIFT and return homography matrix
def get_surf_homography(img1, img2):
    # Initialize SURF
    surf = cv2.xfeatures2d.SURF_create()

    # Extract key points and descriptors from black&white images
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    k1, d1 = surf.detectAndCompute(gray1, None)
    k2, d2 = surf.detectAndCompute(gray2, None)

    # Bruteforce matcher on the descriptors
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(d1, d2, k=2)

    # Make sure that the matches are good
    verify_ratio = 0.8
    verified_matches = []
    for m1, m2 in matches:
        # Add to array only if it's a good match
        if m1.distance < verify_ratio * m2.distance:
            verified_matches.append(m1)

    # Minimum number of matches
    min_matches = 8
    if len(verified_matches) > min_matches:

        # Array to store matching points
        img1_pts = []
        img2_pts = []

        # Add matching points to array
        for match in verified_matches:
            img1_pts.append(k1[match.queryIdx].pt)
            img2_pts.append(k2[match.trainIdx].pt)
        img1_pts = np.float32(img1_pts).reshape(-1, 1, 2)
        img2_pts = np.float32(img2_pts).reshape(-1, 1, 2)

        # Compute homography matrix
        M, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC, 2.5)
        return M
    else:
        print('Error: Not enough matches')
        exit()

# Equalize Histogram of Color Images
def equalize_histogram_color(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img

# Main function definition
def main():
    start_time = time()
    # image1 = cv2.imread('images/scottsdale_right_01.png')
    # image2 = cv2.imread('images/scottsdale_left_01.png')
    image1 = cv2.imread('images/anton2_l.jpg')
    image2 = cv2.imread('images/anton2_r.jpg')

    # Склейка, встроенная в OpenCV; для сравнения
    stitcher = cv2.createStitcher(False)
    cv2stitched = stitcher.stitch((image1, image2))[1]
    cv2.imwrite('results/blended_OpenCV.png',cv2stitched)

    print('Time elapsed for OpenCV stitching: {0:.2f} sec'.format(round(time() - start_time,2)))
    image_warping_time = time()

    # Equalize histogram
    image1 = equalize_histogram_color(image1)
    image2 = equalize_histogram_color(image2)
    image1[image1 == 0] = 1
    image2[image2 == 0] = 1
    # Use SURF to find keypoints and return homography matrix
    M = get_surf_homography(image1, image2)

    # Stitch the images together using homography matrix
    only_img_1, only_img_2 = get_stitched_image(image2, image1, M)

    cv2.imwrite('intermediate/only_img1.png', only_img_1)
    cv2.imwrite('intermediate/only_img2.png', only_img_2)

    print('Time elapsed for image warping: {0:.2f} sec'.format(round(time() - image_warping_time,2)))

    pyram_blend.main(only_img_1, only_img_2)

    print('Done, exiting...')

# Call main function
if __name__ == '__main__':
    main()
