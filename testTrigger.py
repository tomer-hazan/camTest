import cv2
import numpy as np
from vimba import *
from PIL import Image
import time
import util
from threading import Thread

def main():
    with (Vimba.get_instance () as vimba):
        number_of_images=5
        cams = util.trigger_setup(vimba)
        with cams [0] as cam:
            frame_list = util.get_multiple_images(cam,number_of_images)
            util.save_multipal_images([frame_list[0]])
        dif_list = []
        if(len(frame_list)>1):
            for i in range(len(frame_list)-1):
                # name = str(i+1) +"-"+str(i)
                # dif_list.append( getDifShowFrame(name,frame_list[i],frame_list[i+1],10,0))
                dif_list.append(util.getDifFrame(frame_list[i], frame_list[i + 1], 90))
        # for val in range(len(frame_list)-1):
        #     dif_list.append(minLimit( find_differance(frame_list[val],frame_list[val+1]),10))

        sum_dif = util.sum_list(dif_list)

        avrage_dif = sum_dif/len(dif_list)

        util.create_video_from_images_list(frame_list,"images")
        print("made images video")
        util.create_video_from_images_list(dif_list, "differences",30)
        print("made differences video")
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
