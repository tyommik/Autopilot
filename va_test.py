import cv2
from time import gmtime, strftime
import np
import datetime

# image is expected be in RGB color space
def select_rgb_blue_black(image):
    # define range of blue color in HSV
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])

    blue_mask = cv2.inRange(image, lower_blue, upper_blue)
    # define range of black color in HSV
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([170, 100, 50])
    # Threshold the HSV image to get only blue colors
    black_mask = cv2.inRange(image, lower_black, upper_black)

    # Bitwise-AND mask and original image
    mask = cv2.bitwise_or(blue_mask, black_mask)
    res = cv2.bitwise_and(image,image, mask= blue_mask)
    return mask

def convert_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

def convert_hls(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

def convert_gray_scale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def apply_smoothing(image, kernel_size=15):
    """
    kernel_size must be postivie and odd
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def detect_edges(image, low_threshold=0, high_threshold=20):
    return cv2.Canny(image, low_threshold, high_threshold)


def filter_region(image, vertices):
    """
    Create the mask using the vertices and apply it to the input image
    """
    mask = np.zeros_like(image)
    if len(mask.shape) == 2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        cv2.fillPoly(mask, vertices, (255,) * mask.shape[2])  # in case, the input image has a channel dimension
    return cv2.bitwise_and(image, mask)


def select_region(image):
    """
    It keeps the region surrounded by the `vertices` (i.e. polygon).  Other area is set to 0 (black).
    """
    # first, define the polygon by vertices
    rows, cols = image.shape[:2]
    bottom_left = [cols * 0.1, rows * 0.95]
    top_left = [cols * 0.4, rows * 0.6]
    bottom_right = [cols * 0.9, rows * 0.95]
    top_right = [cols * 0.6, rows * 0.6]
    # the vertices are an array of polygons (i.e array of arrays) and the data type must be integer
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    return filter_region(image, vertices)


def hough_lines(image):
    """
    `image` should be the output of a Canny transform.

    Returns hough lines (not the image with lines)
    """
    return cv2.HoughLinesP(image, rho=1, theta=np.pi / 180, threshold=0, minLineLength=20, maxLineGap=30)

# Экстраполяция линий
def average_slope_intercept(lines):
    left_lines = []  # (slope, intercept)
    left_weights = []  # (length,)
    right_lines = []  # (slope, intercept)
    right_weights = []  # (length,)

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 == x1:
                continue  # ignore a vertical line
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            if slope < 0:  # y is reversed in image
                left_lines.append((slope, intercept))
                left_weights.append((length))
            else:
                right_lines.append((slope, intercept))
                right_weights.append((length))

    # add more weight to longer lines
    left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None

    return left_lane, right_lane  # (slope, intercept), (slope, intercept)


def make_line_points(y1, y2, line):
    """
    Convert a line represented in slope and intercept into pixel points
    """
    if line is None:
        return None

    slope, intercept = line

    # make sure everything is integer as cv2.line requires it
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    y1 = int(y1)
    y2 = int(y2)

    return ((x1, y1), (x2, y2))


def lane_lines(image, lines):
    left_lane, right_lane = average_slope_intercept(lines)

    y1 = image.shape[0]  # bottom of the image
    y2 = y1 * 0.6  # slightly lower than the middle

    left_line = make_line_points(y1, y2, left_lane)
    right_line = make_line_points(y1, y2, right_lane)

    return left_line, right_line


def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=20):
    # make a separate image to draw lines and combine with the orignal later
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            l = line.tolist()[0]
            x1 = l[0]
            y1 = l[1]
            x2 = l[2]
            y2 = l[3]
            cv2.line(line_image, (x1, y1), (x2, y2), color, thickness)
    # image1 * α + image2 * β + λ
    # image1 and image2 must be the same shape.
    return cv2.addWeighted(image, 1.0, line_image, 0.95, 0.0)

def isCenterPosition(lines):
    pass

if __name__ == "__main__":
    cap = cv2.VideoCapture('http://192.168.204.105:8081/?action=stream')

    if cap.isOpened():
        # get vcap property
        width = cap.get(3)  # float
        height = cap.get(4) # float

    while True:
      ret, frame = cap.read()
      if ret:
          # cv2.putText(img=frame,
          #             text=str(strftime("%H:%M:%S", gmtime())),
          #             org=(int(width / 1.5), int(height - 10)),
          #             fontFace=cv2.FONT_HERSHEY_DUPLEX,
          #             fontScale=3,
          #             color=(255, 255, 255),
          #             thickness=2)
          blue_black_filter = select_rgb_blue_black(frame)
          smooth_img = apply_smoothing(blue_black_filter, kernel_size=17)
          canny_img = detect_edges(smooth_img)
          roi_img = select_region(canny_img)
          found_lines = hough_lines(roi_img)
          print(found_lines)
          extra_lines = draw_lane_lines(frame, found_lines)

          cv2.imshow('Video', extra_lines)
      if cv2.waitKey(1) == 27:
        exit(0)

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()