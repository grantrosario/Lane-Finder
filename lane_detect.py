import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from numpy.polynomial import Polynomial as P
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# Keep track of previous left lane x values and right lane x values
PREV_LEFT_X1 = None
PREV_LEFT_X2 = None
PREV_RIGHT_X1 = None
PREV_RIGHT_X2 = None

def grayscale(img):
    """ Convers image to grayscale"""
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Darken the grayscale image
    contrast = 1
    brightness = -50

    return cv2.addWeighted(img, contrast, np.zeros(img.shape, img.dtype), 0, brightness)

def toHSV(img):
	img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
	return img

def iso_yellow(img):

	# hue_threshold = 90
	# sat_threshold = 240
	# val_threshold = 240

	upper_yellow = np.array([30, 255, 255])
	lower_yellow = np.array([10, 50, 210])

	mask = cv2.inRange(img, lower_yellow, upper_yellow)
	return mask

def iso_white(img):

	# hue_threshold = 90
	# sat_threshold = 240
	# val_threshold = 240

	upper_yellow = np.array([30, 255, 255])
	lower_yellow = np.array([10, 50, 210])

	mask = cv2.inRange(img, lower_yellow, upper_yellow)
	return mask

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):

    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def find_slope(line):
    """Calculate the slope of a line with format [x1 y1 x2 y2]"""
    return (float(line[3]) - line[1]) / (float(line[2]) - line[0])


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """Calculate the lines from points detected by hough transform and draw them on the image"""
    global PREV_LEFT_X1, PREV_LEFT_X2, PREV_RIGHT_X1, PREV_RIGHT_X2

    mid_barrier = int(img.shape[1]/2) # Divide image in half to detect right and left lane lines
    left_x = []
    left_y = []
    right_x = []
    right_y = []

    for line in lines:
        x1 = line[0][0]
        y1 = line[0][1]
        x2 = line[0][2]
        y2 = line[0][3]
        slope = find_slope(line[0])

        if 0.3 > slope > -0.3:  # Ignore ~horizontal lines
            continue

        if slope < 0:
            if x1 > mid_barrier:
                continue

            left_x += [x1, x2]  # track left lane coordinates
            left_y += [y1, y2]

        else:
            if x1 < mid_barrier:
                continue

            right_x += [x1, x2]  # track right lane coordinates
            right_y += [y1, y2]

    line_bottom = img.shape[0]  # define the top and bottom of our drawn line
    line_top = img.shape[0] / 2 + 80

    if len(left_x) <= 1 or len(right_x) <= 1:  # end early if new left or right lane coordinates are not available
        if PREV_LEFT_X1 is not None:
            cv2.line(img, (int(PREV_LEFT_X1), int(line_bottom)), (int(PREV_LEFT_X2), int(line_top)), color, thickness)
            cv2.line(img, (int(PREV_LEFT_X2), int(line_bottom)), (int(PREV_RIGHT_X2), int(line_top)), color, thickness)
        return

    left_poly = P.fit(np.array(left_x), np.array(left_y), 1)  # fit a polynomial equation to left and right lane coordinates
    right_poly = P.fit(np.array(right_x), np.array(right_y), 1)

    left_x1 = (left_poly - line_bottom).roots()  # return the roots of the polynomial between the line ceiling and floor previously defined
    right_x1 = (right_poly - line_bottom).roots()

    left_x2 = (left_poly - line_top).roots()
    right_x2 = (right_poly - line_top).roots()

    if PREV_LEFT_X1 is not None:    # normalize our left and right x coordinates to increase line stability
        left_x1 = PREV_LEFT_X1 * 0.7 + left_x1 * 0.3
        left_x2 = PREV_LEFT_X2 * 0.7 + left_x2 * 0.3
        right_x1 = PREV_RIGHT_X1 * 0.7 + right_x1 * 0.3
        right_x2 = PREV_RIGHT_X2 * 0.7 + right_x2 * 0.3

    PREV_LEFT_X1 = left_x1
    PREV_LEFT_X2 = left_x2
    PREV_RIGHT_X1 = right_x1
    PREV_RIGHT_X2 = right_x2

    cv2.line(img, (int(left_x1), int(line_bottom)), (int(left_x2), int(line_top)), color, thickness)
    cv2.line(img, (int(right_x1), int(line_bottom)), (int(right_x2), int(line_top)), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines, [255, 0, 0], 10)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


def process_image(base_image):
    img_height = base_image.shape[0]
    img_width = base_image.shape[1]

    vertices = np.array(
                    [[(0,img_height),
                    (img_width/2-30, img_height/2+60), 
                    (img_width/2+30, img_height/2+60), 
                    (img_width,img_height)]], 
                    dtype=np.int32
    )

    gray = grayscale(base_image)
    image = toHSV(base_image)
    yellow_mask = iso_yellow(image)
    image = iso_white(image)
    #image = iso_yellow(image)
    return image
    # image = gaussian_blur(image, 5)
    # image = canny(image, 50, 150)
    # image = region_of_interest(image, vertices)
    # image = hough_lines(image, 1, np.pi/180, 35, 5, 2)
    # return weighted_img(image, base_image, .8, 1., 0.)


image = mpimg.imread('test_images/solidYellowCurve.jpg')
#image = mpimg.imread('test_images/solidWhiteCurve.jpg')
#image = mpimg.imread('test_images/solidYellowCurve.jpg')
#image = mpimg.imread('test_images/solidYellowCurve2.jpg')
#image = mpimg.imread('test_images/solidYellowLeft.jpg')
#image = mpimg.imread('test_images/whiteCarLaneSwitch.jpg')
plt.imshow(process_image(image)) 
plt.show()

# white_output = 'white.mp4'
# clip1 = VideoFileClip("solidWhiteRight.mp4")
# white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
# white_clip.write_videofile(white_output, audio=False)

# white_output = 'yellow.mp4'
# clip1 = VideoFileClip("solidYellowLeft.mp4")
# white_clip = clip1.fl_image(process_image)
# white_clip.write_videofile(white_output, audio=False)

# challenge_output = 'extra.mp4'
# clip2 = VideoFileClip('challenge.mp4')
# challenge_clip = clip2.fl_image(process_image)
# challenge_clip.write_videofile(challenge_output, audio=False)

# challenge_output = 'home_WhiteDotted.mp4'
# clip2 = VideoFileClip('WhiteDottedLanes.mp4')
# challenge_clip = clip2.fl_image(process_image)
# challenge_clip.write_videofile(challenge_output, audio=False)



