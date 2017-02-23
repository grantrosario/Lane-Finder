#**Lane-Finder**
--
##**Finding Lane lines on the road**

[image1]: ./test_images/whiteCarLaneSwitch.jpg "Original"
[image2]: ./test_images/whiteCarLaneSwitch_gray.jpg "Grayscale"
[image3]: ./test_images/whiteCarLaneSwitch_blur.jpg "Gaussian Blur"
[image4]: ./test_images/whiteCarLaneSwitch_canny.jpg "Canny Edge"
[image5]: ./test_images/whiteCarLaneSwitch_region.jpg "Region of Interest"
[image6]: ./test_images/whiteCarLaneSwitch_hough.jpg "Hough Transform"
[image7]: ./test_images/whiteCarLaneSwitch_final.jpg "Weighted Image"

###1. The Pipeline
---
The process_image() pipeline I built consists of 6 main steps:    

-  **Step 1.**  
Convert the image to grayscale. 
![image2]  

- **Step 2.**  
Apply a Gaussian Blur to the image to eliminate insignificant gradient changes.
![image3]
    
- **Step 3.**  
Apply a Canny edge detection transform to the image.
![image4]  
  
- **Step 4.**  
Define a region of interest to block out extraneous edges and irrelevant information.
![image5] 
  
- **Step 5.**  
Apply a Hough tranformation to the image.  
This step involves running the draw_lines() function so I'll explain how it works:
  
-
###draw_lines()
**Note:** *I defined four global variables at the start of the program to keep track of the previous lines x coordinates each time we loop through a new line*  

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by ... first dividing the image vertically in order to separate the right lane lines from the left lane lines by their associating x-coordinates.

I then loop through each line in the image, during which the following steps take place for each line:  
  
Extract the x1, y1, x2, y2 value and calculate the slope of the line using the formula (y2-y1) / (x2-x1). 
  
Track the coordinates of the line only if the slope positive for the right lane, negative for the left lane, and not near horizontal.  
  
Define the top and bottom of the lines we wish to draw on the image  
  
If the current line coordinates are unavailable, draw lines based on previous line cooridnates stored in global variables.  
  
Otherwise, we fit a 1-dimensional polynomial equation to the left and right lane coordinates. 
  
Extract the root of the polynomials.

Normalize the left and right x-coordinates so lines are more stable.  

Store current x-coordinates in global variables to save them for next loop.  
  
Finally, once loop completes, draw lines on image based on newly fitted x-coordinates from polynomial and the top and bottom of the line we defined earlier.

The final image after Hough tranform and draw_lines() will look like this:

![alt text][image6]  

--
- **Step 6.**  
Combine our new image with lines drawn on with or original image by returning the weighted image.  
![alt text][image7]  
  
---

###2. Shortcomings


A potential shortcoming with this pipeline could be observed by the lines losing accuracy when major events happen such as pavement color changing (making lane lines harder to see) or shadows affecting edge detection.

Another shortcoming could be if the road has a sharp turn. This pipeline is fit using a linear polynomial, thus it will not properly guage a drastic turn in the highway.
  
---

###3. Improvements

A few possible improvements could be merging adjacent frames in the video or applying more stringent intercept and slope filtering to increase accuracy when event such as those mentioned previously occur.

It could also be a major benefit to apply logistic regression rather than a linear polynomial fit. This could help fix issues regarding sharp turns.