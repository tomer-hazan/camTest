import cv2
from vimba import *

import testTrigger
import util


def main():
    with (Vimba.get_instance() as vimba):
        number_of_imsges = 1
        cams = util.trigger_setup(vimba)
        if (len(cams) == 0):
            raise Exception("no camera found")
        with cams[0] as cam:
            images = util.get_multiple_images(cam,number_of_imsges)
            util.save_multipal_images(images)
            # util.create_video_from_images_list(images,"images")
            for image in range(len(images)):
                img = util.minLimit( images[image],120)
                cv2.imshow(str(image),img)
            cv2.waitKey(0)








if __name__ == '__main__':
    main()