import sys
import cv2
import numpy as np

# Main function definition
def main():
    # Get input set of images
    img1 = cv2.imread('D:\\DownloadsBrowser\\panorama-stitching\\images\\anton_r.jpg')
    img2 = cv2.imread('D:\\DownloadsBrowser\\panorama-stitching\\images\\anton_l.jpg')
    #img1 = cv2.imread(sys.argv[1])
    #img2 = cv2.imread(sys.argv[2])

    stitcher = cv2.createStitcher(False)
    result = stitcher.stitch((img1, img2))
    cv2.imshow('Image 143', result[1])
    cv2.imshow('Image 151', result[1])
    cv2.imshow('Result', result[1])
    cv2.waitKey()
main()