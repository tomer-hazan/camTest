import cv2
import numpy as np
from PIL import Image
from vimba import Vimba
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import colorsys


import util

def find_defrances_in_given_image_set(number_of_images):
    frame_list = util.acquire_images_as_ndArrays(number_of_images, "images")
    dif_list = []
    scaller = 50
    if (len(frame_list) > 1):  # subtracting adjacent frames to make a ndarray of the changes
        for i in range(len(frame_list) - 1):
            dif = util.getPosDifFrame(frame_list[i], frame_list[i+1], 20)*scaller
            # dif = util.create_color_map(dif)
            dif_list.append(dif)
            util.create_color_map(dif)
            cv2.imwrite("images/dif "+str(i)+".png", dif)
def process_and_color_for_range(start, end):
    scaller = 50
    frame1 =  np.asarray(Image.open("images/frame " + str(start) + '.jpg'))
    if(end-start<1):return
    for image in range(start,end):
        frame2 = np.asarray(Image.open("images/frame " + str(image+1) + '.jpg'))
        dif = util.getPosDifFrame(frame1, frame2, 12) * scaller
        util.create_color_map(dif)
        cv2.imwrite("images/dif " + str(image) + ".png", dif)
        frame1=frame2

def process():
    frame = util.acquire_images_as_ndArrays(1, "testMetirial")[0]
    sum_lines = [row.sum() for row in frame]
    plt.plot(sum_lines)
    plt.show()

def work_on_already_dif_images():
    dif_list = util.acquire_images_as_ndArrays(29, "testMetirial", "dif")
    if (len(dif_list) > 1):  # subtracting adjacent frames to make a ndarray of the changes
        for i in range(len(dif_list)):
            dif = util.minLimit(dif_list[i],10)
            dif_list[i]=dif*30

    util.create_video_from_images_list(dif_list,"post procceses diffrances",1)
    # cv2.waitKey(0)
def main():
    # find_defrances_in_given_image_set(60)
    util.create_video_from_images_folder("temp","temp video")
if __name__ == '__main__':
    main()