import colorsys
import time

import cv2
import os

import numpy as np
from PIL import Image
from vimba import PixelFormat


def display_image(name,image,time=0):#time 0 means endless
    cv2.imshow(name,image)
    cv2.waitKey(time)
def display_video(video_name):
    cap = cv2.VideoCapture(video_name+'.mp4')

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    # Read until video is completed
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:

            # Display the resulting frame
            cv2.imshow(video_name, frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()
    cv2.destroyWindow(video_name)


def display_video_repeatedly(video_name):
    while True:
        display_video(video_name)

def create_video_from_images_folder(folder,name):
    video_filename = name+'.mp4'
    valid_images = [i for i in os.listdir(folder) if i.endswith((".jpg", ".jpeg", ".png"))]

    first_image = cv2.imread(os.path.join(folder, valid_images[0]))
    h, w, _ = first_image.shape

    codec = cv2.VideoWriter_fourcc(*'mp4v')
    vid_writer = cv2.VideoWriter(video_filename, codec, 30, (w, h))

    for img in valid_images:
        loaded_img = cv2.imread(os.path.join(folder, img))
        for _ in range(20):
            vid_writer.write(loaded_img)

    vid_writer.release()

def create_video_from_images_list(images_list,name,scaller=1):
    video_filename = "videos/"+name+'.mp4'
    h, w, _ = images_list[0].shape

    codec = cv2.VideoWriter_fourcc(*'mp4v')
    vid_writer = cv2.VideoWriter(video_filename, codec, 30, (w, h))

    for img in images_list:
        for _ in range(20):
            vid_writer.write(img*scaller)
    print("finished creating " +name+" video")
    vid_writer.release()

def trigger_setup(vimba):
    cams = vimba.get_all_cameras()
    if(len(cams)==0):
        raise Exception("no camera found")
    with cams[0] as cam:
        #exposure
        cam.ExposureAuto.set("Off")
        cam.ExposureTimeAbs.set(1000000)#65000#400000

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

def normal_setup(vimba):
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
def find_positive_differance(f1,f2):
    ret =  (f2.astype(np.int16) - f1.astype(np.int16))
    ret[ret < 0] = 0
    ret=ret.astype(np.uint8)
    print("finished finding diffrances")
    return ret

def get_multiple_images(cam,number_of_images):
    image_list = capture_multiple_images(cam,number_of_images)
    image_list = transform_image_list(image_list)
    return image_list
def capture_multiple_images(cam,number_of_images):
    image_list = []
    try:
        frame_cuptcherd_time=time.time_ns()
        for frame in cam.get_frame_generator(limit=number_of_images):
            print(time.time_ns() - frame_cuptcherd_time)
            image_list.append(frame)
            frame_cuptcherd_time=time.time_ns()
    except:
        raise Exception("no trigger")
    return image_list
def transform_image_list(image_list):
    for frame in range(len(image_list)):
        image_list[frame].convert_pixel_format(PixelFormat.Mono8)
        image_list[frame] = image_list[frame].as_opencv_image()
        image_list[frame] = minLimit(image_list[frame] , 60)
        print("transformed frame "+str(frame))
    return image_list
def save_multipal_images(frame_list):
    if frame_list==[]:raise Exception("no image cuptcherd")
    for frame in range(len(frame_list)):
        name = "images/"+'frame ' + str(frame) + '.jpg'
        cv2.imwrite(name, frame_list[frame])
        print("saved " + name)

def minLimit(image,limit):
    image = cv2.cvtColor(image, cv2.CAP_PVAPI_PIXELFORMAT_MONO8)
    return np.array([[col if col[0] >= limit else [0, 0, 0]  for col in row ] for row in image]).astype(np.uint8)

def limits(image,min_limit=0,max_limit=255):
    image = cv2.cvtColor(image, cv2.CAP_PVAPI_PIXELFORMAT_MONO8)
    return np.array([[col if col[0] >= min_limit and col[0] <= max_limit else [0, 0, 0]  for col in row ] for row in image]).astype(np.uint8)


def accuire_images_as_ndarrays(number_of_images,folder=None,name="frame"):
    frame_list = []
    if(folder!=None):
        for i in range(number_of_images):
            img = Image.open(folder+"/"+name+' '+str(i)+'.jpg')
            numpy_frame = np.asarray(img)
            frame_list.append( numpy_frame)
    else:
        for i in range(number_of_images):
            img = Image.open('frame '+str(i)+'.jpg')
            numpy_frame = np.asarray(img)
            frame_list.append( numpy_frame)

    return frame_list

def getDifFrame(frame1,frame2,min_limit=0,max_limit=255):
    # dif_frame = find_differance(frame1,frame2)
    dif_frame = find_differance(frame1,frame2)
    dif_frame = cv2.cvtColor(dif_frame, cv2.CAP_PVAPI_PIXELFORMAT_MONO8)
    dif_frame = limits(dif_frame,min_limit,max_limit)
    return dif_frame
def getPosDifFrame(frame1,frame2,min_limit=0,max_limit=255):
    # dif_frame = find_differance(frame1,frame2)
    dif_frame = find_positive_differance(frame1,frame2)
    dif_frame = cv2.cvtColor(dif_frame, cv2.CAP_PVAPI_PIXELFORMAT_MONO8)
    dif_frame = limits(dif_frame,min_limit,max_limit)
    return dif_frame

def getDifShowFrame(name, image1, image2, scaller, min_limit):
    dif_image = getDifFrame(image1,image2,min_limit)
    cv2.imshow(name,dif_image*scaller)
    return dif_image

def sum_list(list):
    if(len(list)==0):return 0
    sum = list[0].astype(np.int32)
    for item in range(len(list)-1):
        sum+=list[item+1]
    return sum.astype(np.int32)
def get_and_save_multipal_images_and_time_log(cam,number_of_images):
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
        name = "images/"+'frame ' + str(frame_number) + '.jpg'
        cv2.imwrite(name, image)
        frame_number += 1
        print("saved " + name)
        frame_cuptcherd_time=time.time_ns()
    print("saved all images")
    return (frame_list,time_log)

def create_color_map(image):
    for row in range(len(image)):
        for col in range(len(image[row])):
            image[row][col] = np.multiply(colorsys.hsv_to_rgb(image[row][col][0] / 360.0, 1, 1), image[row][col][0])

    return image