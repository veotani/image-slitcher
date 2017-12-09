import sys
import os
import numpy as np
import cv2
import scipy
from scipy import ndimage
from scipy.stats import norm
from scipy.signal import convolve2d
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy as sp
from skimage.morphology import closing

'''split rgb image to its channels'''


def split_rgb(image):
    red = None
    green = None
    blue = None
    (blue, green, red) = cv2.split(image)
    return red, green, blue


'''generate a 5x5 kernel'''


def generating_kernel(a):
    w_1d = np.array([0.25 - a / 2.0, 0.25, a, 0.25, 0.25 - a / 2.0])
    return np.outer(w_1d, w_1d)


'''reduce image by 1/2'''


def ireduce(image):
    out = None
    kernel = generating_kernel(0.4)
    outimage = scipy.signal.convolve2d(image, kernel, 'same')
    out = outimage[::2, ::2]
    return out


'''expand image by factor of 2'''


def iexpand(image):
    out = None
    kernel = generating_kernel(0.4)
    outimage = np.zeros((image.shape[0] * 2, image.shape[1] * 2), dtype=np.float64)
    outimage[::2, ::2] = image[:, :]
    out = 4 * scipy.signal.convolve2d(outimage, kernel, 'same')
    return out


'''create a gaussain pyramid of a given image'''


def gauss_pyramid(image, levels):
    output = []
    output.append(image)
    tmp = image
    for i in range(0, levels):
        tmp = ireduce(tmp)
        output.append(tmp)
    return output


'''build a laplacian pyramid'''


def lapl_pyramid(gauss_pyr):
    output = []
    k = len(gauss_pyr)
    for i in range(0, k - 1):
        gu = gauss_pyr[i]
        egu = iexpand(gauss_pyr[i + 1])
        if egu.shape[0] > gu.shape[0]:
            egu = np.delete(egu, (-1), axis=0)
        if egu.shape[1] > gu.shape[1]:
            egu = np.delete(egu, (-1), axis=1)
        output.append(gu - egu)
    output.append(gauss_pyr.pop())
    return output


'''Blend the two laplacian pyramids by weighting them according to the mask.'''


def blend(lapl_pyr_white, lapl_pyr_black, gauss_pyr_mask):
    blended_pyr = []
    k = len(gauss_pyr_mask)
    for i in range(0, k):
        p1 = gauss_pyr_mask[i] * lapl_pyr_white[i]
        p2 = (1 - gauss_pyr_mask[i]) * lapl_pyr_black[i]
        blended_pyr.append(p1 + p2)
    return blended_pyr


'''Reconstruct the image based on its laplacian pyramid.'''


def collapse(lapl_pyr):
    output = None
    output = np.zeros((lapl_pyr[0].shape[0], lapl_pyr[0].shape[1]), dtype=np.float64)
    for i in range(len(lapl_pyr) - 1, 0, -1):
        lap = iexpand(lapl_pyr[i])
        lapb = lapl_pyr[i - 1]
        if lap.shape[0] > lapb.shape[0]:
            lap = np.delete(lap, (-1), axis=0)
        if lap.shape[1] > lapb.shape[1]:
            lap = np.delete(lap, (-1), axis=1)
        tmp = lap + lapb
        lapl_pyr.pop()
        lapl_pyr.pop()
        lapl_pyr.append(tmp)
        output = tmp
    return output

def closest_point(node, nodes):
    nodes = np.asarray(nodes)
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2)

def linear_function(x, xmin, xmax):
    res = (255/(xmax-xmin))*x - (255*xmin/(xmax-xmin))
    if res<0:
        res = 0
    elif res > 255:
        res = 255
    return res

# мб стоит удалить я хз
def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2.0 + (Y-center[1])**2.0)
    # mask = dist_from_center <= radius
    mask = np.exp(-np.square(dist_from_center)/2.0)*255.0/(np.sqrt(2.0*np.pi))
    plt.imshow(dist_from_center, interpolation='nearest')
    plt.show()
    return mask

#Мб стоит удалить
def get_bounding_rect(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x, y, h, w = cv2.boundingRect(cnt)
    return x, y, h, w

def main():
    image1 = cv2.imread('D:\\DownloadsBrowser\\panorama-stitching\\only_img1.png')
    image2 = cv2.imread('D:\\DownloadsBrowser\\panorama-stitching\\only_img2.png')
    mask = cv2.imread('D:\\DownloadsBrowser\\panorama-stitching\\mask_image.png')
    r1 = None
    g1 = None
    b1 = None
    r2 = None
    g2 = None
    b2 = None
    rm = None
    gm = None
    bm = None

    # EBEMSYA S MASKOI
    # Задача: в области наложения сделать плавный переход от (0,0,0) к (255,255,255) с помощью некой функции (и только в ней)

    #шаг 1: обозначаем зоны А, Б и combined
    zoneA = []
    zoneB = []
    zoneC = []
    w_res, h_res = mask.shape[:2]

    thresh1 = 50
    thresh2 = 127
    thresh3 = 255

    # Load image
    im = sp.misc.imread('D:\\DownloadsBrowser\\panorama-stitching\\mask_image.png')

    # # RAZMECHAEM ZONY
    # for w in tqdm(range(w_res)):
    #     for h in range(h_res):
    #         if all(mask[w,h] == [50,50,50]):
    #             zoneA.append((w,h))
    #         elif all(mask[w,h] == [127,127,127]):
    #             zoneC.append((w,h))
    #         elif all(mask[w,h] == [255, 255, 255]):
    #             zoneB.append((w,h))
    # print(len(zoneA))
    # print(len(zoneB))
    # print(len(zoneC))
    # #zoneAB = zoneA + zoneB
    # min_x, min_y = np.min(zoneC, axis=0)
    # max_x, max_y = np.max(zoneC, axis=0)

    # #СОЗДАЁМ МАСКУ ДЛЯ БЛЕНДИНГА: ЛИНЕЙНЫЙ ГРАДИЕНТ ПО ЗАДАННОМУ НАПРАВЛЕНИЮ
    #
    # # Направление блендинга: по вектору {direction_x, direction_y} (не забываем, что y идет вниз)
    # direction_x = 1.0
    # direction_y = -0.25
    # # Коэффициент расширения маски (тип того)
    # stretch_coef = 0.20
    #
    mask_redone = mask.copy()
    # mask_redone2 = mask.copy()
    # for point in tqdm(zoneA):
    #      mask_redone[point] = (0,0,0)
    # for point in tqdm(zoneC):
    #     p = (point[1]*direction_x + point[0]*direction_y)/(direction_x+direction_y)
    #     min_p = (min_x*direction_x + min_y*direction_y)/(direction_x+direction_y)
    #     max_p = (max_x * direction_x + max_y * direction_y) / (direction_x + direction_y)
    #     # intensity = linear_function(point[0], min_x+(max_x-min_x)*0.20, max_x-(max_x-min_x)*0.20)
    #     intensity = linear_function(p, min_p + (max_p - min_p) * stretch_coef, max_p - (max_p - min_p) * stretch_coef)
    #     mask_redone[point] = (intensity, intensity, intensity)
    # for point in tqdm(zoneA):
    #     mask_redone[point] = (0,0,0)
    # # Гауссиановское размытие результата(пока хз зачем)
    # # Оно показало себя плохо, отключил.
    # # mask_redone = ndimage.gaussian_filter(mask_redone, sigma=3)
    # # /EBEMSYA S MASKOI

    # СОЗДАЁМ МАСКУ ДЛЯ БЛЕНДИНГА КАК ЦИРКУЛЯРКУ ИЗ 1го ИЗОБРАЖЕНИЯ

    mask_redone[:,:] = [0,0,0]
    x, y, h, w = get_bounding_rect(image1)

    mask_circular = create_circular_mask(h, w)

    resultC = np.zeros(mask_circular.shape, dtype=mask_circular.dtype)
    tmpC = []
    tmpC.append(mask_circular)
    tmpC.append(mask_circular)
    tmpC.append(mask_circular)
    resultC = cv2.merge(tmpC, resultC)
    mask_redone[x:x + h, y:y + w] = resultC
    # /ЦИРКУЛЯРКА


    cv2.imwrite('mask_redone.png', mask_redone)
    mask = mask_redone


    (r1, g1, b1) = split_rgb(image1)
    (r2, g2, b2) = split_rgb(image2)
    (rm, gm, bm) = split_rgb(mask)

    r1 = r1.astype(float)
    g1 = g1.astype(float)
    b1 = b1.astype(float)

    r2 = r2.astype(float)
    g2 = g2.astype(float)
    b2 = b2.astype(float)

    rm = rm.astype(float) / 255
    gm = gm.astype(float) / 255
    bm = bm.astype(float) / 255

    rr = rm.__mul__(r2) + (1.0 - rm).__mul__(r1)
    gr = gm.__mul__(g2) + (1.0 - gm).__mul__(g1)
    br = bm.__mul__(b2) + (1.0 - bm).__mul__(b1)

    resultN = np.zeros(image1.shape, dtype=image1.dtype)
    tmpN = []
    tmpN.append(br)
    tmpN.append(gr)
    tmpN.append(rr)
    resultN = cv2.merge(tmpN, resultN)


    cv2.imwrite('blended_nopyramids.jpg', resultN)

    #Далее идёт пирамидальный блендинг, он навскидку показывает себя хуже линейного
    return

    # Automatically figure out the size
    min_size = min(r1.shape)
    depth = int(math.floor(math.log(min_size, 2))) - 4  # at least 16x16 at the highest level.

    gauss_pyr_maskr = gauss_pyramid(rm, depth)
    gauss_pyr_maskg = gauss_pyramid(gm, depth)
    gauss_pyr_maskb = gauss_pyramid(bm, depth)

    gauss_pyr_image1r = gauss_pyramid(r1, depth)
    gauss_pyr_image1g = gauss_pyramid(g1, depth)
    gauss_pyr_image1b = gauss_pyramid(b1, depth)

    gauss_pyr_image2r = gauss_pyramid(r2, depth)
    gauss_pyr_image2g = gauss_pyramid(g2, depth)
    gauss_pyr_image2b = gauss_pyramid(b2, depth)

    lapl_pyr_image1r = lapl_pyramid(gauss_pyr_image1r)
    lapl_pyr_image1g = lapl_pyramid(gauss_pyr_image1g)
    lapl_pyr_image1b = lapl_pyramid(gauss_pyr_image1b)

    lapl_pyr_image2r = lapl_pyramid(gauss_pyr_image2r)
    lapl_pyr_image2g = lapl_pyramid(gauss_pyr_image2g)
    lapl_pyr_image2b = lapl_pyramid(gauss_pyr_image2b)

    outpyrr = blend(lapl_pyr_image2r, lapl_pyr_image1r, gauss_pyr_maskr)
    outpyrg = blend(lapl_pyr_image2g, lapl_pyr_image1g, gauss_pyr_maskg)
    outpyrb = blend(lapl_pyr_image2b, lapl_pyr_image1b, gauss_pyr_maskb)

    outimgr = collapse(blend(lapl_pyr_image2r, lapl_pyr_image1r, gauss_pyr_maskr))
    outimgg = collapse(blend(lapl_pyr_image2g, lapl_pyr_image1g, gauss_pyr_maskg))
    outimgb = collapse(blend(lapl_pyr_image2b, lapl_pyr_image1b, gauss_pyr_maskb))
    # blending sometimes results in slightly out of bound numbers.
    outimgr[outimgr < 0] = 0
    outimgr[outimgr > 255] = 255
    outimgr = outimgr.astype(np.uint8)

    outimgg[outimgg < 0] = 0
    outimgg[outimgg > 255] = 255
    outimgg = outimgg.astype(np.uint8)

    outimgb[outimgb < 0] = 0
    outimgb[outimgb > 255] = 255
    outimgb = outimgb.astype(np.uint8)

    result = np.zeros(image1.shape, dtype=image1.dtype)
    tmp = []
    tmp.append(outimgb)
    tmp.append(outimgg)
    tmp.append(outimgr)
    result = cv2.merge(tmp, result)
    cv2.imwrite('blended.jpg', result)

    print('збс')


if __name__ == '__main__':
    main()