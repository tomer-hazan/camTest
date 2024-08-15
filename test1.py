import cv2
from vimba import *

import util

with Vimba.get_instance() as vimba:
    util.create_video_from_images_folder("difImages","dif video")

