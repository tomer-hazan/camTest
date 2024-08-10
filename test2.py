from vimba import *
import cv2
import numpy as np

def setUp():
    cams = vimba.get_all_cameras()
    with cams[0] as cam:
        cam.ExposureAuto.set("Off")
        cam.ExposureTimeAbs.set(65000)
    return cams

def getAndShowFrame(name,cam):
    frame = cam.get_frame()
    frame = frame.as_numpy_ndarray()  # replaces the original vimba.Frame object with a numpy.ndarray
    frame = cv2.cvtColor(frame, cv2.CAP_PVAPI_PIXELFORMAT_MONO8)
    cv2.imshow(name,frame)
    return frame

def minLimit(image,limit):
    return np.array([[col if col[0] > limit else [0, 0, 0]  for col in row ] for row in image]).astype(np.uint8)


with Vimba.get_instance () as vimba:
    cams = setUp()
    with cams [0] as cam:

        #Camera is set to BayerGR8
        f1 = getAndShowFrame("frame 1",cam)
        f2 = getAndShowFrame("frame 2", cam)
        FD = np.absolute( f2.astype(np.int16)-f1.astype(np.int16))
        FD = FD.astype(np.uint8)
        cv2.imshow("difference",FD)
        FD2 = minLimit(FD,10)*10
        cv2.imshow(" above 10", FD2)
        cv2.waitKey(0)


