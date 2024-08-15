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
            capture_and_save_constant_fps(cam,0.5,number_of_imsges)

def capture_and_save_constant_fps(cam,time_between_frames,number_of_images):
    frame_capture_time = time.time()
    for img in range(number_of_images):
        frame = cam.get_frame()
        print(time.time()-frame_capture_time)
        frame_capture_time = time.time()
        frame.convert_pixel_format(PixelFormat.Mono8)
        cv2.imwrite('images/frame ' + str(img) + '.jpg', frame.as_opencv_image())
        time.sleep(time_between_frames-(time.time()-frame_capture_time)-0.07)









if __name__ == '__main__':
    main()