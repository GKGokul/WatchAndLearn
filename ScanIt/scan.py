# import the necessary packages
import imutils
import numpy as np
import cv2
import matplotlib.pyplot as plt

# STEP 1 - EDGE DETECTION
# Note: Before edge detection, we resize the shape of the image
image = cv2.imread('billimage.jpeg')
image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height=500)

# Convert the image to grayscale
# Blur it
# Find Edges
gray = cv2.GaussianBlur(image, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)

# show the original image and the edge detected image
print("STEP 1: Edge Detection")
cv2.imshow("Image", image)
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()


# STEP 2 - CONTOUR FINDING
# Find the contours in the edges image
(_,cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Show the contour (outline) of the piece of paper
print("STEP 2: Find contours of paper")
cv2.drawContours(image, cnts, -1, (0, 0, 255), 2)
# Show the image after drawing the contours
cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# TODO: Find a way to get only 4 points from the image
pts1 = np.float32([[75,75],[290,25],[100,300],[325,250]])
pts2 = np.float32([[0,0],[500,0],[0,500],[500,500]])

# Do perspective transformation to get the proper view
M = cv2.getPerspectiveTransform(pts1,pts2)

resPer = cv2.warpPerspective(orig,M,(500,500))
plt.subplot(121),plt.imshow(orig,cmap='gray'),plt.title('Input')
plt.subplot(122),plt.imshow(resPer,cmap='gray'),plt.title('Output')
plt.show()