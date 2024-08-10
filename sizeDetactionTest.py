import cv2
import numpy as np
from vimba import PixelFormat, Vimba

import test4


def get_image(vimba):
    cams = test4.setUp(vimba)
    with cams[0] as cam:
        for frame in cam.get_frame_generator(limit=1):
            frame.convert_pixel_format(PixelFormat.Mono8)
            image = frame.as_opencv_image()
            cv2.imwrite("frame 0.jpg", image)
            image = test4.minLimit(image, 0)

            return image
def detect_new_image(vimba):
    image = get_image(vimba)
    return detect_image(image)

def detect_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.stackBlur(gray, (17, 17), 0)
    cv2.imwrite("blured.jpg", blur)
    # Apply edge detection
    edges = cv2.Canny(blur, 40, 150, apertureSize=3)

    # Apply Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

    # Create a copy of the original image to draw lines on
    line_image = image.copy()

    # Draw the lines
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)


        # Find intersections of lines to detect the rectangle
        # (For simplicity, assuming lines form a rectangle, we can find the bounding box of these lines)

        # Collect all points from the detected lines
        points = []
        for line in lines:
            points.append((line[0][0], line[0][1]))
            points.append((line[0][2], line[0][3]))
        # Find the bounding box of these points
        x, y, width, height = cv2.boundingRect(np.array(points))
        cv2.imshow('Detected Lines', line_image)
        return (image,( x, y, width, height))

    cv2.imshow('Detected Lines', line_image)
    return (image,( 0, 0, 0, 0))
def draw_and_show_rect(image,x,y,width,height):
    print("(x,y,width,height) = (" + str(x) + ", " + str(y) + ", " + str(width) + ", " + str(height) + ")")
    # Draw the bounding box on the original image
    cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)

    # Display the result
    cv2.imshow('Detected Rectangle', image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    with Vimba.get_instance() as vimba:
        image, (x,y,w,h) = detect_new_image(vimba)
        draw_and_show_rect(image, x,y,w,h)


if __name__ == '__main__':
    main()
