import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import cv2

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
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


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    imshape = img.shape
    print(img.shape)

    Left_xArr = []
    Right_xArr = []
    Left_yArr = []
    Right_yArr = []

    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = float((y2-y1)/(x2-x1))
            if not (np.isnan(slope)) or (np.isinf(slope)) or (slope == 0):
                if  slope < 0:
                    Left_xArr.append(x1)
                    Left_xArr.append(x2)
                    Left_yArr.append(y1)
                    Left_yArr.append(y2)
                else:
                    Right_xArr.append(x1)
                    Right_xArr.append(x2)
                    Right_yArr.append(y1)
                    Right_yArr.append(y2)

    if(len(Left_xArr) <= 0):
        Left_X_min = 0
        Left_X_max = 0
        Left_Y_min = 0
        Left_Y_max = 0
    else:
        Left_X_min = min(Left_xArr)
        Left_X_max = max(Left_xArr)
        Left_Y_min = min(Left_yArr)
        Left_Y_max = max(Left_yArr)

    if(len(Right_xArr) <= 0):
        Right_X_min = 0
        Right_X_max = 0
        Right_Y_min = 0
        Right_Y_max = 0
    else:
        Right_X_min = min(Right_xArr)
        Right_X_max = max(Right_xArr)
        Right_Y_min = min(Right_yArr)
        Right_Y_max = max(Right_yArr)

    cv2.line(img, (Left_X_min, Left_Y_max), (Left_X_max, Left_Y_min), color, thickness)
    cv2.line(img, (Right_X_max, Right_Y_max), (Right_X_min, Right_Y_min), color, thickness)

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


def process_image(image):

    imshape = image.shape

    vertices = np.array([[(0,imshape[0]),
                          (imshape[1]/2-30, imshape[0]/2+50), 
                          (imshape[1]/2+30, imshape[0]/2+50), 
                          (imshape[1],imshape[0])]], 
                          dtype=np.int32)

    gray = grayscale(image)

    blur = gaussian_blur(gray, 5)

    can_image = canny(blur, 65, 200)

    region_image = region_of_interest(can_image, vertices)

    huffy = hough_lines(region_image, 1, np.pi/180, 50, 150, 300)

    final = weighted_img(huffy, image, .8, 1., 0.)

    return final


#image = mpimg.imread('test_images/solidWhiteRight.jpg')
#image = mpimg.imread('test_images/solidWhiteCurve.jpg')
#image = mpimg.imread('test_images/solidYellowCurve.jpg')
#image = mpimg.imread('test_images/solidYellowCurve2.jpg')
#image = mpimg.imread('test_images/solidYellowLeft.jpg')
#image = mpimg.imread('test_images/whiteCarLaneSwitch.jpg')
#plt.imshow(process_image(image))
#print(image.shape)
#plt.show()

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




