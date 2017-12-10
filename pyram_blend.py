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

'''create a gaussian pyramid of a given image'''
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

'''Нахождение границы области наложения'''
def get_borders(mask):
    w1, h1 = mask.shape[:2]
    bigger_img = np.concatenate((np.ones((w1, 1)) * False, mask, np.ones((w1, 1)) * False), 1)
    bigger_img = np.concatenate((np.ones((1, h1+2)) * False, bigger_img, np.ones((1, h1+2)) * False), 0)
    bigger_img = bigger_img.astype(bool)
    a = bigger_img[1:w1+1, 2:h1+2]
    b = bigger_img[1:w1+1, 0:h1]
    c = bigger_img[0:w1, 1:h1+1]
    d = bigger_img[2:w1+2, 1:h1+1]
    borders = a | b | c | d
    borders = borders & ~bigger_img[1:w1+1, 1:h1+1]
    return borders

'''Ищем ближайшую точку'''
def closest_node(node, nodes):
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2)

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

    img_height = image1.shape[0]
    img_width = image1.shape[1]
    # Если изображение слишком большое, ресайзим его
    max_height = 1000
    max_width = 1000
    scale = 0.25
    if img_height > max_height and img_width > max_width:
        image1_small = scipy.misc.imresize(image1, (int(img_height * scale), int(img_width * scale)))
        image2_small = scipy.misc.imresize(image2, (int(img_height * scale), int(img_width * scale)))
        mask, thresh_a, thresh_b, thresh_c = create_mask(image1_small, image2_small)
        cv2.imwrite("intermediate/mask_downsized.png", mask)
        print('Image resized! The new one is {0:.2f} times smaller.'.format(round(1/scale,2)))
    else:
        mask, thresh_a, thresh_b, thresh_c =create_mask(image1, image2)
        cv2.imwrite("intermediate/mask_normal.png", mask)


    # Текущая задача: получить контуры С с различием A-C и B-C

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


    # Если изображение было слишком большое, возвращаем маску назад
    if img_height > 1000 and img_width > 1000:
        mask_redone = scipy.misc.imresize(mask_redone, (img_height, img_width))
        cv2.imwrite('results/mask_linear_gradient.png', mask_redone)

    # Применяем линейный блендинг по альфа значениям из маски
    image1f = image1.astype(float)
    image2f = image2.astype(float)
    maskf = mask_redone.astype(float) / 255.0
    result = maskf*image2f + (1.0 - maskf)*image1f
    cv2.imwrite('results/result_linear_gradient.png', result)

    print('Time elapsed for linear gradient blending: {0:.2f} sec'.format(round(time() - start_time,2)))
    time_cool = time()

    # Создаём более сложную маску, учитывающую расстояния до границ области

    # Ищем границы
    borders = get_borders(zone_c_log)
    disp_borders = np.zeros(borders.shape)
    disp_borders[borders] = 255
    disp_borders[zone_a_log&borders] = 100
    disp_borders[zone_b_log&borders] = 200
    borders_a = np.argwhere(disp_borders == 100)
    borders_b = np.argwhere(disp_borders == 200)
    mask[zone_a_log] = 0

    for point_c in tqdm(zone_c):
        da = np.sum(np.square(borders_a[closest_node(point_c, borders_a)] - point_c)) ** (1 / 2)
        db = np.sum(np.square(borders_b[closest_node(point_c, borders_b)] - point_c)) ** (1 / 2)
        if da >= db:
            mask[point_c[0], point_c[1]] = (1 - (db/(2*da)))*255
        else:
            mask[point_c[0], point_c[1]] = (da/(db*2)) * 255
    cv2.imwrite("results/mask_nonlinear_gradient.png", mask)

    # Применяем линейный блендинг по альфа значениям из маски
    image1f = image1.astype(float)
    image2f = image2.astype(float)
    maskf = mask.astype(float) / 255.0
    result = maskf*image2f + (1.0 - maskf)*image1f
    cv2.imwrite('results/result_nonlinear_gradient.png', result)

    print('Time elapsed for non-linear gradient blending: {0:.2f} sec'.format(round(time() - time_cool,2)))

if __name__ == '__main__':
    main()