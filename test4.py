import cv2
import numpy as np
from vimba import *
from PIL import Image
import time

import sizeDetactionTest
import util



def find_disturbance_y(image):
    #proccese the image of change between frames and return the y value of it
    return 0


def main():
    with Vimba.get_instance() as vimba:
        number_of_images = 3
        if (number_of_images > 0):
            cams = util.non_trigger_setup(vimba)
            with cams[0] as cam:
                frame_list, time_log = util.get_and_save_multiple_images_and_time_log(cam, number_of_images)

            # _, rect = sizeDetactionTest.detect_image(frame_list[0])
            # print("(x,y,width,height) = (" + str(rect[0]) + ", " + str(rect[1]) + ", " + str(rect[2]) + ", " + str(
            #     rect[3]) + ")")
            # sizeDetactionTest.draw_and_show_rect(frame_list[0], rect[0], rect[1], rect[2], rect[3])
            dif_list = []
            dif_time_list = []
            if (len(frame_list) > 1):  # subtracting adjacent frames to make a ndarray of the changes
                for i in range(len(frame_list) - 1):
                    name = str(i + 1) + "-" + str(i)
                    dif_list.append(util.getDifShowFrame(name, frame_list[i], frame_list[i + 1], 1000, 10))
                    # dif_dist_list = find_disturbance_y(dif_list[-1])
                    # dif_time_list.append(time_log[i] - time_log[i + 1])
                    # print(str(dif_dist_list[-1] / dif_time_list[-1]) + " - time in pixels/ns")

            sum_dif = util.sum_list(dif_list)
            cv2.imshow("sum dif", sum_dif.astype(np.uint8))
            avrage_dif = sum_dif / len(dif_list)
            cv2.imshow("average dif", avrage_dif.astype(np.uint8))
            cv2.waitKey(0)

if __name__ == '__main__':
    main()








