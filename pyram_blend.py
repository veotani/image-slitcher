import sys
import os
import numpy as np
import cv2
import scipy
from scipy.signal import convolve2d
import math
from tqdm import tqdm
from time import time

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

'''Линейная функция для создания маски'''
def linear_function(x, xmin, xmax):
    res = (255/(xmax-xmin))*x - (255*xmin/(xmax-xmin))
    if res<0:
        res = 0
    elif res > 255:
        res = 255
    return res

'''Создаём маску, где своим цветом показаны зоны изображений и область наложения'''
def create_mask(image1, image2):
    mask = np.zeros(image1.shape, dtype=image1.dtype)

    thresh_a = 50
    thresh_c = 127
    thresh_b = 255

    red1, green1, blue1 = image1[:, :, 0], image1[:, :, 1], image1[:, :, 2]
    mask1 = (red1 > 0) | (green1 > 0) | (blue1 > 0)
    mask[mask1] = [thresh_a, thresh_a, thresh_a]
    red2, green2, blue2 = image2[:, :, 0], image2[:, :, 1], image2[:, :, 2]
    mask2 = (red2 > 0) | (green2 > 0) | (blue2 > 0)
    mask[mask2] = [thresh_b, thresh_b, thresh_b]
    mask3 = mask1 & mask2
    mask[mask3] = [thresh_c, thresh_c, thresh_c]

    return mask, thresh_a, thresh_b, thresh_c

def main(image1, image2):
    start_time = time()

    mask, thresh_a, thresh_b, thresh_c = create_mask(image1, image2)

    cv2.imwrite("intermediate/mask.png", mask)

    # Текущая задача: получить контуры С с различием A-C и B-C

    # Значения пикселей в маске, размечающие области изображений

    # Создаём маску для линейного блендинга по градиентной маске по заданному направлению
    # Обозначаем зоны А, Б и C:=A^B
    zone_a_log = ((mask==thresh_a).all(axis=2))
    zone_a = np.argwhere(zone_a_log == True)
    zone_b_log = ((mask==thresh_b).all(axis=2))
    zone_b = np.argwhere(zone_b_log == True)
    zone_c_log = ((mask==thresh_c).all(axis=2))
    zone_c = np.argwhere(zone_c_log == True)

    # Левая нижняя и правая верхняя границы зоны наложения, в которой будет создаваться градиентная маска
    min_x, min_y = np.min(zone_c, axis=0)
    max_x, max_y = np.max(zone_c, axis=0)

    # Направление блендинга: по вектору {direction_x, direction_y} (не забываем, что y идет вниз)
    direction_x = 1.0
    direction_y = -0.15

    # Коэффициент расширения маски (для лучшего перехода от 0 к 1)
    # Чем больше - тем дальше от изображения лежат границы 0 и 1
    stretch_coef = 0.25

    mask_redone = mask.copy()
    # Создаём градиентную маску
    for x,y in zone_a:
         mask_redone[x,y] = (0,0,0)
    for x, y in zone_b:
        mask_redone[x, y] = (255, 255, 255)
    for x,y in zone_c:
        p = (y*direction_x + x*direction_y)/(direction_x+direction_y)
        min_p = (min_x*direction_x + min_y*direction_y)/(direction_x+direction_y)
        max_p = (max_x * direction_x + max_y * direction_y) / (direction_x + direction_y)
        # intensity = linear_function(point[0], min_x+(max_x-min_x)*0.20, max_x-(max_x-min_x)*0.20)
        intensity = linear_function(p, min_p + (max_p - min_p) * stretch_coef, max_p - (max_p - min_p) * stretch_coef)
        mask_redone[x,y] = (intensity, intensity, intensity)

    cv2.imwrite('results/mask.png', mask_redone)
    mask = mask_redone

    # Применяем линейный блендинг по альфа значениям из маски
    image1f = image1.astype(float)
    image2f = image2.astype(float)
    maskf = mask.astype(float) / 255.0
    result = maskf*image2f + (1.0 - maskf)*image1f
    cv2.imwrite('results/blended_linear.jpg', result)

    print('Time elapsed for linear blending: {0:.2f} sec'.format(round(time() - start_time,2)))
    time_pyramid = time()
    #Далее идёт пирамидальный блендинг, он навскидку показывает себя хуже линейного
    #return

    (r1, g1, b1) = split_rgb(image1)
    (r2, g2, b2) = split_rgb(image2)
    (rm, gm, bm) = split_rgb(mask)

    r1 = r1.astype(float)
    g1 = g1.astype(float)
    b1 = b1.astype(float)

    r2 = r2.astype(float)
    g2 = g2.astype(float)
    b2 = b2.astype(float)

    rm = rm.astype(float) / 255.0
    gm = gm.astype(float) / 255.0
    bm = bm.astype(float) / 255.0

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
    cv2.imwrite('results/blended_pyramid.jpg', result)
    print('Time elapsed for pyramidal blending: {0:.2f} sec'.format(round(time() - time_pyramid,2)))


if __name__ == '__main__':
    main()