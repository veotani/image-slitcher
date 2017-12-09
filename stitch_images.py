import cv2
import numpy as np
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
    #img1_as_first = result_img.copy()


    only_img_2 = result_img.copy()
    only_img_1 = result_img.copy()
    only_img_1[:,:] = (0,0,0)
    only_img_1[transform_dist[1]:w1 + transform_dist[1],
    transform_dist[0]:h1 + transform_dist[0]] = img1

    w_res, h_res = result_img.shape[:2]

    #cv2.imshow("img1", only_img_1)
    #cv2.imshow("img2", only_img_2)

    mask_img = result_img.copy()
    mask_img[:,:] = (0,0,0)

    for w in tqdm(range(w_res)):
        for h in range(h_res):
            if not all(only_img_1[w,h] ==[0,0,0]) and not all(only_img_2[w,h] == [0,0,0]):
                mask_img[w,h] = (127,127,127)
            elif not all(only_img_1[w,h] ==[0,0,0]) and all(only_img_2[w,h] == [0,0,0]):
                mask_img[w,h] = (50,50,50)
            elif all(only_img_1[w,h] ==[0,0,0]) and not all(only_img_2[w,h] == [0,0,0]):
                mask_img[w,h] = (255,255,255)
    #cv2.imshow('Mask', mask_img)

    # img1_as_first[transform_dist[1]:w1 + transform_dist[1],
    # transform_dist[0]:h1 + transform_dist[0]] = img1

    #img2_as_first = img1_as_first.copy()

    # cv2.imshow("result_img", result_img)
    # for w in range(w_res):
    #     for h in range(h_res):
    #         #if len(np.where(result_img[w, h] != 0)[0]) > 0:
    #         #полоски на ярких изображениях можно убрать, поставив порог на ~600
    #         if np.sum(result_img[w,h]) > 0:
    #             img2_as_first[w,h] = result_img[w,h]

    #cv2.imshow("first_as_first", img1_as_first)
    #cv2.imshow("sec_as_first", img2_as_first)
    #res = cv2.addWeighted(img1_as_first, 0.5, img2_as_first, 0.5, 0.0)
    #kernel = np.ones((5, 5), np.float32) / 25
    #dst = cv2.filter2D(res, -1, kernel)
    #cv2.imshow("oh my god please be good", res)
    #pyram_blend(img1_as_first, img2_as_first)


    #ДАЛЬШЕ ТВОРЯТСЯ УЖАСЫ!!!!!!!!! БЕРИТЕ ДЕТЕЙ И БЕГИТЕ, ТАКОЙ СТРАШНЫЙ КОД ДАЖЕ Я НИ РАЗУ РАНЬШЕ НЕ ВИДЕЛ!
    #БЕГИТЕ
    #УЖАСНО!!!!!
    #но я постараюсь задокументировать его ;)
    #но у меня вряд ли получится

    #инициализируем то, что потом станет тупо изображением два на результатном холсте
    #img2_on_res = result_img.copy()

    # #идём по ширине картинки
    # for pixel in range(len(result_img)):
    #     #идём по длине картинки
    #     for a in range(len(result_img[pixel])):
    #         #вот тут внимание...
    #         #ЕСЛИ ПИКСЕЛЬ НЕЧЁРНЫЙ
    #         if len(np.where(result_img[pixel][a] != 0)[0]) > 0:
    #             #и при этом он залез на потенциальную зону первого изображения
    #             if pixel < h1 and a > w1:
    #                 #МОЧИМ ГАДА
    #                 result_img[pixel][a][(np.where(result_img[pixel][a] != 0))] = 0
    # #я писал это 20 минут назад но не знаю, зачем эта строка. пусть остаётся...
    # img2_common = result_img.copy()


    # result_img[transform_dist[1]:w1 + transform_dist[1],
    # transform_dist[0]:h1 + transform_dist[0]] = img1
    # for pixel in range(len(result_img)):
    #     for a in range(len(result_img[pixel])):
    #         if len(np.where(result_img[pixel][a] != 0)[0]) == 0 or len(np.where(img2_common[pixel][a] != 0)[0]) == 0:
    #             result_img[pixel][a][(np.where(result_img[pixel][a] != 0))] = 0
    # img1_common = result_img.copy()
    #
    # for pixel in range(len(img1)):
    #     for a in range(len(img1[pixel])):
    #         if len(np.where(img2_common[pixel][a] != 0)[0]) > 0:
    #             img1[pixel][a][(np.where(img1[pixel][a] != 0))] = 0
    # img2_on_res[transform_dist[1]:w1 + transform_dist[1],
    # transform_dist[0]:h1 + transform_dist[0]] = img1


    # common = cv2.addWeighted(img1_common, 0.5, img2_common, 0.5, 0.0)
    # cv2.imshow("part1", common)
    # res = cv2.addWeighted(common, 0.5, img2_on_res, 0.5, 0.0)
    #cv2.imshow("part2", img2_on_res)
    #cv2.imshow("RESULTTT", res)


    #hm = cv2.addWeighted(reso1, 0.4, result_img, 0.6, 0.0)
    #cv2.imshow("wow", hm)
    #left = 1000
    #for a in coordinates:
    #    if a[1]<left:
    #        left = a[1]
    #nu_i_chto = result_img[left:w1, :]
    #cv2.imshow("abab", nu_i_chto)

   #overl_img2 = result_img[:w1]:


    # Return the result
    return only_img_1, only_img_2, mask_img


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

def pyram_blend(A, B):
    # generate Gaussian pyramid for A
    G = A.copy()
    gpA = [G]
    for i in range(6):
        G = cv2.pyrDown(G)
        gpA.append(G)

    # generate Gaussian pyramid for B
    G = B.copy()
    gpB = [G]
    for i in range(6):
        G = cv2.pyrDown(G)
        gpB.append(G)

    # generate Laplacian Pyramid for A
    lpA = [gpA[5]]
    for i in range(5,0,-1):
        GE = cv2.pyrUp(gpA[i])
        L = cv2.subtract(gpA[i-1],GE)
        lpA.append(L)

    # generate Laplacian Pyramid for B
    lpB = [gpB[5]]
    for i in range(5,0,-1):
        GE = cv2.pyrUp(gpB[i])
        L = cv2.subtract(gpB[i-1],GE)
        lpB.append(L)

    # Now add left and right halves of images in each level
    LS = []
    for la,lb in zip(lpA,lpB):
        rows,cols,dpt = la.shape
        ls = np.hstack((la[:,0:cols/2], lb[:,cols/2:]))
        LS.append(ls)

    # now reconstruct
    ls_ = LS[0]
    for i in range(1,6):
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.add(ls_, LS[i])

    # image with direct connecting each half
    real = np.hstack((A[:,:cols/2],B[:,cols/2:]))

    cv2.imwrite('Pyramid_blending2.jpg',ls_)
    cv2.imwrite('Direct_blending.jpg',real)



# Main function definition
def main():
    img1 = cv2.imread('D:\\DownloadsBrowser\\panorama-stitching\\images\\scottsdale_right_01.png')
    img2 = cv2.imread('D:\\DownloadsBrowser\\panorama-stitching\\images\\scottsdale_left_01.png')


    #Склейка, встроенная в OpenCV; для сравнения
    #stitcher = cv2.createStitcher(False)
    #super = stitcher.stitch((img1, img2))[1]
    #cv2.imshow('super', super)

    # Equalize histogram
    img1 = equalize_histogram_color(img1)
    img2 = equalize_histogram_color(img2)

    #Show input images
    input_images = np.hstack((img1, img2))
    #cv2.imshow ('Input Images', input_images)

    # Use SURF to find keypoints and return homography matrix
    M = get_surf_homography(img1, img2)

    # Stitch the images together using homography matrix
    only_img_1, only_img_2, mask_image = get_stitched_image(img2, img1, M)
    cv2.imwrite('only_img1.png', only_img_1)
    cv2.imwrite('only_img2.png', only_img_2)
    cv2.imwrite('mask_image.png', mask_image)

    print('збс')
    # Show the resulting image
    #cv2.imshow('Result', result_image)
    #cv2.waitKey()




# Call main function
if __name__ == '__main__':
    main()
