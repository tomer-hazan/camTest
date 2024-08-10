import cv2
import numpy as np
from vimba import *
from PIL import Image
import time
import util
from threading import Thread
def setUp(vimba):
    cams = vimba.get_all_cameras()
    if(len(cams)==0):
        raise Exception("no camera found")
    with cams[0] as cam:
        #exposure
        cam.ExposureAuto.set("Off")
        cam.ExposureTimeAbs.set(30000)#65000#400000

        #trigger
        cam.TriggerSelector.set("FrameStart")
        cam.TriggerSource.set("Line1")#Line1 faster but no electric isolation
        # cam.TriggerSource.set("Line2")  # Line2 slower but with electric isolation
        cam.TriggerActivation.set("RisingEdge")
        cam.AcquisitionFrameCount.set(5)#the limit of the number of imaes that can be taken
        cam.AcquisitionMode.set("MultiFrame")#SingleFrame
        cam.Gain.set("0")#16
        cam.Gamma.set("1")
        cam.TriggerMode.set("On")


        print("event triger - " + str(cam.EventFrameTriggerReady))



    print("finished set up")
    return cams

def find_differance(f1,f2):
    ret =  (np.absolute(f2.astype(np.int16) - f1.astype(np.int16))).astype(np.uint8)
    print("finished finding diffrances")
    return ret

def get_multiple_images(cam,number_of_images):
    image_list = []
    try:
        frame_cuptcherd_time=time.time_ns()
        for frame in cam.get_frame_generator(limit=number_of_images):
            print(time.time_ns() - frame_cuptcherd_time)
            image_list.append(frame)
            frame_cuptcherd_time=time.time_ns()
    except:
        raise Exception("no trigger")
    for frame in range(len(image_list)):
        image_list[frame].convert_pixel_format(PixelFormat.Mono8)
        image_list[frame] = image_list[frame].as_opencv_image()
        image_list[frame] = minLimit(image_list[frame] , 60)
        print("transformed frame "+str(frame))
    return image_list

def save_multipal_images(frame_list):
    if frame_list==[]:raise Exception("no image cuptcherd")
    for frame in range(len(frame_list)):
        name = 'frame ' + str(frame) + '.jpg'
        cv2.imwrite(name, frame_list[frame])
        print("saved " + name)

def get_and_save_multipal_images(cam,number_of_images):
    print("event triger - " + str(cam.EventFrameTriggerReady))
    frame_number = 0
    frame_cuptcherd_time = time.time_ns()
    try:
        for frame in cam.get_frame_generator(limit=number_of_images):
            print(time.time_ns() - frame_cuptcherd_time)
            frame.convert_pixel_format(PixelFormat.Mono8)
            image = frame.as_opencv_image()
            image = minLimit(image, 0)
            name = 'frame ' + str(frame_number) + '.jpg'
            cv2.imwrite(name, image)
            frame_number += 1
            print("saved " + name)
            frame_cuptcherd_time=time.time_ns()
    except:
        raise Exception("no trigger")
    print("saved all images")

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

def getDifFrame(frame1,frame2,min_limit=0):
    dif_frame = find_differance(frame1,frame2)
    dif_frame = cv2.cvtColor(dif_frame, cv2.CAP_PVAPI_PIXELFORMAT_MONO8)
    dif_frame = minLimit(dif_frame,min_limit)
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
def main():
    with (Vimba.get_instance () as vimba):
        number_of_images=300
        cams = setUp(vimba)
        with cams [0] as cam:
            frame_list = get_multiple_images(cam,number_of_images)
            save_multipal_images([frame_list[0]])
        dif_list = []
        if(len(frame_list)>1):
            for i in range(len(frame_list)-1):
                # name = str(i+1) +"-"+str(i)
                # dif_list.append( getDifShowFrame(name,frame_list[i],frame_list[i+1],10,0))
                dif_list.append(getDifFrame(frame_list[i], frame_list[i + 1], 90))
        # for val in range(len(frame_list)-1):
        #     dif_list.append(minLimit( find_differance(frame_list[val],frame_list[val+1]),10))

        sum_dif = sum_list(dif_list)
        cv2.imshow("sum dif", sum_dif.astype(np.uint8))
        cv2.imshow("frame 0", frame_list[0].astype(np.uint8))
        avrage_dif = sum_dif/len(dif_list)
        cv2.imshow("avrage dif",avrage_dif.astype(np.uint8))
        util.create_video_from_images_list(frame_list,"images")
        print("made images video")
        util.create_video_from_images_list(dif_list, "diffrances")
        print("made diffrances video")
        differences_video = Thread(target = util.display_video_repeatedly, args = ("differences", ))
        photos_video = Thread(target=util.display_video_repeatedly, args=("images",))
        differences_video.start()
        photos_video.start()
        cv2.waitKey(0)

if __name__ == '__main__':
    main()
