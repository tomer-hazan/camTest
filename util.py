import colorsys
import time

import cv2
import os

import numpy as np
from PIL import Image
from vimba import PixelFormat


def display_image(name, image, time=0):  #time 0 means endless
    cv2.imshow(name, image)
    cv2.waitKey(time)


def display_video(video_name):
    """
        a function that displays a video (using open cv 2) once
        @:param video_name: str - name of the mp4 file
    """
    cap = cv2.VideoCapture(video_name + '.mp4')

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
    """
        a function that displays a video (using open cv 2) forever
        @:param video_name: str - name of the mp4 file
        @problems stops all other viewed images and behaves poorly in a thread
    """
    while True:
        display_video(video_name)


def create_video_from_images_folder(folder, name):
    """
        a function that creates a video (using open cv 2) from all the jpg/jpeg/png images inside a given folder
        @:param folder: str - name of the folder the images are stored in
        @:param name: str - the name of the new mp4 file
    """
    video_filename = name + '.mp4'
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


def create_video_from_images_list(images_list, name, scaler=1):
    """
        a function that creates a video (using open cv 2) from a list of jpg/jpeg/png images
        @:param images_list: python list ([]) - list of the images (ndArray's) to be combined to a video
        @:param name: str - the name of the new mp4 file
        @:param scaler: number - a scaler for the brightness of the image

    """
    video_filename = "videos/" + name + '.mp4'
    h, w, _ = images_list[0].shape

    codec = cv2.VideoWriter_fourcc(*'mp4v')
    vid_writer = cv2.VideoWriter(video_filename, codec, 30, (w, h))

    for img in images_list:
        for _ in range(20):
            vid_writer.write(img * scaler)
    print("finished creating " + name + " video")
    vid_writer.release()


def trigger_setup(vimba):
    """
        a function that puts the right config for physical triggering on the first active camera (in the active cameras list)
        @:param vimba: Vimba.get_instance - a variable that used for getting and interacting with the allied vision camera
        @:exception  no camera found - the function couldn't find any active cameras
        @:returns the list of active cameras
    """
    cams = vimba.get_all_cameras()
    if (len(cams) == 0):
        raise Exception("no camera found")
    with cams[0] as cam:
        #exposure
        cam.ExposureAuto.set("Off")
        cam.ExposureTimeAbs.set(12185)  #65000#1000000

        #trigger
        cam.TriggerSelector.set("FrameStart")
        cam.TriggerSource.set("Line1")  #Line1 faster but no electric isolation
        # cam.TriggerSource.set("Line2")  # Line2 slower but with electric isolation
        cam.TriggerActivation.set("RisingEdge")
        cam.AcquisitionMode.set("SingleFrame")  #SingleFrame
        cam.Gain.set("21")  #16
        cam.Gamma.set("1")
        cam.TriggerMode.set("On")

        print("event triger - " + str(cam.EventFrameTriggerReady))

    print("finished set up")
    return cams


def non_trigger_setup(vimba):
    """
        a function that puts the right config for continuance triggering (no trigger) on the first active camera (in the active cammeras list)
        @:param vimba: Vimba.get_instance - a variable that used for getting and interacting with the allied vision camera
        @:exception  no camera found - the function couldn't find any active cameras
        @:returns the list of active cameras
    """
    cams = vimba.get_all_cameras()

    if (len(cams) == 0):
        raise Exception("no camera found")
    try:
        with cams[0] as cam:
            #exposure
            cam.ExposureAuto.set("Off")
            cam.ExposureTimeAbs.set(64885)  #1033940
            cam.Gamma.set(1)
            cam.TriggerSelector.set("FrameStart")
            cam.TriggerMode.set("Off")
            cam.TriggerSource.set("Freerun")  #Line1 faster but no electric isolation
            cam.Gain.set("32")
    except:
        raise Exception("the cammera is used elsewhere")

    print("finished set up")
    return cams


def find_differance(f1, f2):
    """
        a function that returns an ndArray that is the absolute value of the differences of two frames in the form of ndArray's (f2-f1)
        @:param f1: ndArray - represents the first frame
        @:param f2: ndArray - represents the second frame
        @:returns ndArray -  represents an image of the change between two frames
    """
    ret = (np.absolute(f2.astype(np.int16) - f1.astype(np.int16))).astype(np.uint8)
    print("finished finding diffrances")
    return ret


def find_positive_differance(f1, f2):
    """
        a function that returns an ndArray that is the positive values of the differences of two frames in the form of ndArray's (f2-f1)
        @note: for negative values would be zero
        @:param f1: ndArray - represents the first frame
        @:param f12: ndArray - represents the second frame
        @:returns ndArray -  represents an image of the positive change between two frames
    """
    ret = (f2.astype(np.int16) - f1.astype(np.int16))
    ret[ret < 0] = 0
    ret = ret.astype(np.uint8)
    print("finished finding diffrances")
    return ret


def capture_multiple_images(cam, number_of_images):
    """
        a function that captures a given amount of images using the given camera
        @:param cam: Camera - the camera that would be used to take the images with
        @:param number_of_images: int - represents the number of images the functions would capture
        @:returns image_list: [ndArray] -  represents a list of the captured images
    """
    image_list = []
    try:
        frame_cuptcherd_time = time.time_ns()
        for frame in cam.get_frame_generator(limit=number_of_images):
            print(time.time_ns() - frame_cuptcherd_time)
            image_list.append(frame)
            frame_cuptcherd_time = time.time_ns()
    except:
        raise Exception("no trigger")
    return image_list


def get_multiple_images(cam, number_of_images):
    """
        a function that captures a given amount of images using the given camera and transforms it using the transform_image_list()
        @:param cam: Camera - the camera that would be used to take the images with
        @:param number_of_images: int - represents the number of images the functions would capture
        @:returns image_list: [ndArray] -  represents a list of the captured and processed images
    """
    image_list = capture_multiple_images(cam, number_of_images)
    image_list = transform_image_list(image_list)
    return image_list


def transform_image_list(image_list):
    """
        a function that transformes a list of images to list of ndArray's and removes the pixels whose values are lower then a threshold
        @:param image_list: [images] - the image list to be transformed
        @:returns image_list: [ndArray] -  the given list (after the transformation)
    """
    for frame in range(len(image_list)):
        image_list[frame].convert_pixel_format(PixelFormat.Mono8)
        image_list[frame] = image_list[frame].as_opencv_image()
        image_list[frame] = minLimit(image_list[frame], 60)
        print("transformed frame " + str(frame))
    return image_list


def save_multipal_images(frame_list, name="frame"):
    """
        a function that saves to a file (inside the images folder) a list of images
        @:param frame_list: [ndArray] - the image list to be saved
        @:param name: the name of the new files
        @note: the files are going to be in the form of: "name 0", "name 1", "name 2" ...
    """
    if frame_list == []: raise Exception("no image cuptcherd")
    for frame in range(len(frame_list)):
        file_name = "images/" + name + " " + str(frame) + '.jpg'
        cv2.imwrite(file_name, frame_list[frame])
        print("saved " + file_name)


def minLimit(image, limit):
    """
        a function that filters out all the pixels whose value is lower than the given limit
        @:param image: ndArray - represents a given image
        @:param limit: number - a value that set the lower threshold for this function
        @:returns: ndArray -  the filtered image
    """
    image = cv2.cvtColor(image, cv2.CAP_PVAPI_PIXELFORMAT_MONO8)
    return np.array([[col if col[0] >= limit else [0, 0, 0] for col in row] for row in image]).astype(np.uint8)


def limits(image, min_limit=0, max_limit=255):
    """
        a function that filters out all the pixels whose value is out of the given range
        @:param image: ndArray - represents a given image
        @:param min_limit: number - a value that set the lower threshold for this function
        @:param max_limit: number - a value that set the higher threshold for this function
        @:returns: ndArray -  the filtered image
    """
    image = cv2.cvtColor(image, cv2.CAP_PVAPI_PIXELFORMAT_MONO8)
    return np.array(
        [[col if col[0] >= min_limit and col[0] <= max_limit else [0, 0, 0] for col in row] for row in image]).astype(
        np.uint8)


def acquire_images_as_ndArrays(number_of_images, folder=None, name="frame"):
    """
        a function that reads a given number of images from a given folder and returns them as ndArrays
        @:param number_of_images: int - the amount of images this function would read
        @:param folder: string - the name of the directory from wich the images would be read
         (if no value is given then it would take the images from the current folder)
        @:param name: string - the name of the image files
         (the files should be organised as such: "name 0", "name 1", "name 2" ...)
        @:returns frame_list: [ndArray] -  the list of the newly read images
    """
    frame_list = []
    number_of_images_str = str(number_of_images)
    if (folder != None):
        for i in range(number_of_images):
            img = Image.open(folder + "/" + name + ' ' + str(i) + '.jpg')
            numpy_frame = np.asarray(img)
            frame_list.append(numpy_frame)
            print("opened " +str(i)+"/"+number_of_images_str)
    else:
        for i in range(number_of_images):
            img = Image.open('frame ' + str(i) + '.jpg')
            numpy_frame = np.asarray(img)
            frame_list.append(numpy_frame)
            print("opened " + str(i) + "/" + number_of_images_str)

    return frame_list


def getDifFrame(frame1, frame2, min_limit=0, max_limit=255):
    """
        a function that returns an ndArray that is the absolute value of the differences of two frames in the form of ndArray's (f2-f1)
        after filtering the pixel values such that they would be higher than the min_limit and lower than the max_limit
        @:param f1: ndArray - represents the first frame
        @:param f2: ndArray - represents the second frame
        @:param min_limit: number - a value that set the lower threshold for this function
        @:param max_limit: number - a value that set the higher threshold for this function
        @:returns: ndArray -  filtered image that represents the change between two ndArrays image
    """
    dif_frame = find_differance(frame1, frame2)
    dif_frame = cv2.cvtColor(dif_frame, cv2.CAP_PVAPI_PIXELFORMAT_MONO8)
    dif_frame = limits(dif_frame, min_limit, max_limit)
    return dif_frame


def getPosDifFrame(frame1, frame2, min_limit=0, max_limit=255):
    """
        a function that returns an ndArray that is the positive values of the differences of two frames in the form of ndArray's (f2-f1)
        after filtering the pixel values such that they would be higher than the min_limit and lower than the max_limit
        @note: for negative values would be zero
        @:param f1: ndArray - represents the first frame
        @:param f2: ndArray - represents the second frame
        @:param min_limit: number - a value that set the lower threshold for this function
        @:param max_limit: number - a value that set the higher threshold for this function
        @:returns: ndArray -  filtered image that represents the change between two ndArrays image
    """
    dif_frame = find_positive_differance(frame1, frame2)
    dif_frame = cv2.cvtColor(dif_frame, cv2.CAP_PVAPI_PIXELFORMAT_MONO8)
    dif_frame = limits(dif_frame, min_limit, max_limit)
    return dif_frame


def getDifShowFrame(name, image1, image2, scaler, min_limit):
    """
        a function that returns the difference between two ndArrays just like in getDifFrame() and displays the image
        @:param f1: image1 - represents the first frame
        @:param f2: image2 - represents the second frame
        @:param min_limit: number - a value that set the lower threshold for this function
        @:param scaler: number - a value that the image pixels values are going to be scaled by when the image is being displayed
        @:returns: ndArray -  filtered image that represents the change between two ndArrays image
    """
    dif_image = getDifFrame(image1, image2, min_limit)
    cv2.imshow(name, dif_image * scaler)
    return dif_image


def sum_list(list):
    """
        a function that returns the ndArray that represents the sum of all the ndArrays in the  given list
        @:param list: [ndArray] - represents list of frames
        @:returns: ndArray -  image that each of its pixels have the value of the sum of the values in the corresponding pixels in the given list
    """
    if (len(list) == 0): return 0
    sum = list[0].astype(np.int32)
    for item in range(len(list) - 1):
        sum += list[item + 1]
    return sum.astype(np.int32)


def get_and_save_multipal_images_and_time_log(cam, number_of_images):
    """
        a function that captures and saves to a file a given amount of images using the given camera and transforms it using the like in transform_image_list()
        @:param cam: Camera - the camera that would be used to take the images with
        @:param number_of_images: int - represents the number of images the functions would capture
        @:returns image_list: [ndArray] -  represents a list of the captured and processed images
        @:returns time_log: [float] -  list of the time between capturing consecutive frames
    """

    frame_number = 0
    frame_cuptcherd_time = time.time_ns()
    frame_list = []
    time_log = []
    for frame in cam.get_frame_generator(limit=number_of_images):
        time_log.append(time.time_ns() - frame_cuptcherd_time)
        print(time.time_ns() - frame_cuptcherd_time)
        frame.convert_pixel_format(PixelFormat.Mono8)
        image = frame.as_opencv_image()
        image = minLimit(image, 0)
        frame_list.append(image)
        name = "images/" + 'frame ' + str(frame_number) + '.jpg'
        cv2.imwrite(name, image)
        frame_number += 1
        print("saved " + name)
        frame_cuptcherd_time = time.time_ns()
    print("saved all images")
    return (frame_list, time_log)


def create_color_map(image):
    """
        a function that takes a monochrome image and colors its power such that pixel with lower value would be red and with higher value would be blue
        @note: the given image would be changed (make a clone if you want to keep the original image)
        @:param image: ndArray - the image to be colored
        @:returns image: ndArray -  the original (but now colored ) image
    """
    for row in range(len(image)):
        for col in range(len(image[row])):
            image[row][col] = np.multiply(colorsys.hsv_to_rgb(image[row][col][0] / 360.0, 1, 1), image[row][col][0])

    return image


def capture_and_save_constant_fps(cam, time_between_frames, number_of_images):
    """
    a function that captures and saves to files a given amount of images with a constant fps (constant time between frames)
    :param cam: Camera - the camera that would be used to take the images with
    :param time_between_frames: float - the time the camera waits between each frame
    :param number_of_images: int - the amount of images that would be captured
    """
    frame_capture_time = time.time()
    for img in range(number_of_images):
        frame = cam.get_frame()
        print(time.time() - frame_capture_time)
        frame_capture_time = time.time()
        frame.convert_pixel_format(PixelFormat.Mono8)
        cv2.imwrite('images/frame ' + str(img) + '.jpg', frame.as_opencv_image())
        time.sleep(time_between_frames - (time.time() - frame_capture_time) - 0.07)
