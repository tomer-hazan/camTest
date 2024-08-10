import cv2
import numpy as np
from vimba import *
from PIL import Image
import time

import sizeDetactionTest


def setUp(vimba):
    cams = vimba.get_all_cameras()

    if(len(cams)==0):
        raise Exception("no camera found")
    try:
        with cams[0] as cam:
            #exposure
            cam.ExposureAuto.set("Off")
            cam.ExposureTimeAbs.set(64885)#1033940
            cam.Gamma.set(1)
            cam.TriggerSelector.set("FrameStart")
            cam.TriggerMode.set("Off")
            cam.TriggerSource.set("Freerun")#Line1 faster but no electric isolation
            cam.Gain.set("32")
    except:
        raise Exception("the cammera is used elsewhere")


    print("finished set up")
    return cams

def find_differance(f1,f2):
    ret =  (np.absolute(f2.astype(np.int16) - f1.astype(np.int16))).astype(np.uint8)
    print("finished finding diffrances")
    return ret

def get_and_save_multipal_images(cam,number_of_images):
    frame_number = 0
    frame_cuptcherd_time = time.time_ns()
    frame_list = []
    time_log=[]
    for frame in cam.get_frame_generator(limit=number_of_images):
        time_log.append(time.time_ns() - frame_cuptcherd_time)
        print(time.time_ns() - frame_cuptcherd_time)
        frame.convert_pixel_format(PixelFormat.Mono8)
        image = frame.as_opencv_image()
        image = minLimit(image, 0)
        frame_list.append(image)
        name = 'frame ' + str(frame_number) + '.jpg'
        cv2.imwrite(name, image)
        frame_number += 1
        print("saved " + name)
        frame_cuptcherd_time=time.time_ns()
    print("saved all images")
    return (frame_list,time_log)

def get_and_show_multipal_images(cam, number_of_images):
    frame_number = 0
    frame_cuptcherd_time = time.time_ns()
    for frame in cam.get_frame_generator(limit=number_of_images):
        print(time.time_ns() - frame_cuptcherd_time)
        frame.convert_pixel_format(PixelFormat.Mono8)
        image = frame.as_opencv_image()
        image = minLimit(image, 0)
        name = 'frame ' + str(frame_number) + '.jpg'
        cv2.imshow(name, image)
        frame_number += 1
        print("saved " + name)
        frame_cuptcherd_time=time.time_ns()
    print("showed all images")

def minLimit(image,limit):
    image = cv2.cvtColor(image, cv2.CAP_PVAPI_PIXELFORMAT_MONO8)
    return np.array([[col if col[0] >= limit else [0, 0, 0]  for col in row ] for row in image]).astype(np.uint8)

def accuire_images_as_ndarrays(number_of_images):
    frame_list = []
    for i in range(number_of_images):
        img = Image.open('frame '+str(i)+'.jpg')
        numpy_frame = np.asarray(img)
        frame_list.append( numpy_frame)
    return frame_list

def getDifShowFrame(name,frame1,frame2,scaller=1,min_limit=0):
    dif_frame = find_differance(frame1,frame2)
    dif_frame = cv2.cvtColor(dif_frame, cv2.CAP_PVAPI_PIXELFORMAT_MONO8)
    dif_frame = minLimit(dif_frame,min_limit)
    cv2.imshow(name,dif_frame*scaller)
    return dif_frame

def show_and_remove_low(frame,name,low_limit=0):
    frame = cv2.cvtColor(frame, cv2.CAP_PVAPI_PIXELFORMAT_MONO8)
    frame = minLimit(frame, low_limit)
    cv2.imshow(str(name), frame)

def avrage_list(list):
    sum=sum_list(list)/len(list)
    return sum.astype(np.int32)

def sum_list(list):
    if(len(list)==0):return 0
    sum = list[0].astype(np.int32)
    for item in range(len(list)-1):
        sum+=list[item+1]
    return sum.astype(np.int32)

def find_disturbance_y(image):
    #proccese the image of change between frames and return the y value of it
    return 0







def main():
    with Vimba.get_instance() as vimba:
        number_of_images = 3
        if (number_of_images > 0):
            cams = setUp(vimba)
            with cams[0] as cam:
                frame_list, time_log = get_and_save_multipal_images(cam, number_of_images)

            for image in range(len(frame_list)):  # removes dark pixels
                frame_list[image] = minLimit(frame_list[image], 0)
            _, rect = sizeDetactionTest.detect_image(frame_list[0])
            print("(x,y,width,height) = (" + str(rect[0]) + ", " + str(rect[1]) + ", " + str(rect[2]) + ", " + str(
                rect[3]) + ")")
            sizeDetactionTest.draw_and_show_rect(frame_list[0], rect[0], rect[1], rect[2], rect[3])
            dif_list = []
            dif_dist_list = []
            dif_time_list = []
            if (len(frame_list) > 1):  # subtracting adjacent frames to make a ndarray of the changes
                for i in range(len(frame_list) - 1):
                    name = str(i + 1) + "-" + str(i)
                    dif_list.append(getDifShowFrame(name, frame_list[i], frame_list[i + 1], 100, 10))
                    dif_dist_list = find_disturbance_y(dif_list[-1])
                    dif_time_list.append(time_log[i] - time_log[i + 1])
                    print(str(dif_dist_list[-1] / dif_time_list[-1]) + " - time in pixels/ns")

            sum_dif = sum_list(dif_list)
            cv2.imshow("sum dif", sum_dif.astype(np.uint8))
            avrage_dif = sum_dif / len(dif_list)
            cv2.imshow("average dif", avrage_dif.astype(np.uint8))
            cv2.waitKey(0)

if __name__ == '__main__':
    main()

# with Vimba.get_instance () as vimba:
#     cams = setUp()
#     with cams [0] as cam:
#
#         #Camera is set to BayerGR8
#         f1 = getAndShowFrame("frame 1",cam)
#         f2 = getAndShowFrame("frame 2", cam)
#         FD = np.absolute( f2.astype(np.int16)-f1.astype(np.int16))
#         FD = FD.astype(np.uint8)
#         cv2.imshow("difference",FD)
#         FD2 = minLimit(FD,10)*10
#         cv2.imshow(" above 10", FD2)
#         cv2.waitKey(0)








