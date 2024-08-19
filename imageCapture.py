import cv2
from vimba import *
import time

import testTrigger
import util


def main():
    with (Vimba.get_instance() as vimba):
        number_of_imsges = 60
        cams = util.trigger_setup(vimba)
        if (len(cams) == 0):
            raise Exception("no camera found")
        with cams[0] as cam:
            # images = util.get_multiple_images(cam,number_of_imsges)
            # util.save_multipal_images(images
            util.capture_and_save_constant_fps(cam,0.5,number_of_imsges)











if __name__ == '__main__':
    main()