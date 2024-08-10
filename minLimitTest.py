import cv2
from vimba import Vimba, PixelFormat
import testTrigger

with (Vimba.get_instance() as vimba):
    cams = testTrigger.setUp(vimba)
    with cams[0] as cam:
        limit=0
        image = testTrigger.get_multiple_images(cam,1)[0]

        while True:
            limited_image = testTrigger.minLimit(image, limit)
            cv2.imshow("image with limit "+ str(limit), limited_image)
            cv2.imshow("original image " + str(limit), image)
            cv2.waitKey(1)
            limit = int(input("what limit should it be? - "))
            cv2.destroyAllWindows()