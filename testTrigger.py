import cv2
import numpy as np
from vimba import *
from PIL import Image
import time

import sizeDetactionTest
import util
from threading import Thread


def main():
    with (Vimba.get_instance() as vimba):
        number_of_images = 10
        cams = util.trigger_setup(vimba)
        with cams[0] as cam:
            frame_list = util.get_multiple_images(cam, number_of_images)
            util.save_multiple_images(frame_list)
        # frame_list = [image[y:y + height, x:x + width] for image in frame_list]
        dif_list = []
        if len(frame_list) > 1:
            for i in range(len(frame_list) - 1):
                dif_list.append(util.getDifFrame(frame_list[i], frame_list[i + 1], 15))

            util.save_multiple_images(dif_list, "processedImages", "dif")
            for dif in dif_list:
                util.create_color_map(dif)
        util.create_video_from_images_list(frame_list, "images")
        print("made images video")
        util.create_video_from_images_list(dif_list, "differences", 30)
        print("made differences video")
        cv2.waitKey(0)
        # differences_video = Thread(target = util.display_video_repeatedly, args = ("videos/differences", ))
        # photos_video = Thread(target=util.display_video_repeatedly, args=("videos/images",))
        # differences_video.start()
        # photos_video.start()
        # util.display_image("sum dif", sum_dif.astype(np.uint8))
        # Thread(target=util.display_image, args=("sum dif",sum_dif.astype(np.uint8))).start()
        # Thread(target=util.display_image, args=("frame 0", frame_list[0].astype(np.uint8))).start()
        # Thread(target=util.display_image, args=("avrage dif", avrage_dif.astype(np.uint8))).start()
        # photos_video.join()


if __name__ == '__main__':
    main()
