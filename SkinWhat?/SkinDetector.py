# import the necessary packages
import imutils
import numpy as np
import cv2

# Define the upper and lower boundaries of the HSV pixel intensities to be considered 'skin'
lower = np.array([0, 38, 60], dtype = "uint8")
upper = np.array([50, 255, 255], dtype = "uint8")

camera = cv2.VideoCapture(0)

while True:
    # Get the current frame
    _, frame = camera.read()

    # Resize the frame
    # Convert it to the HSV color space,
    # Determine the HSV pixel intensities that fall into the specified boundaries
    frame = imutils.resize(frame, width=500)
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(converted, lower, upper)

    # Apply a series of erosions and dilations to the mask using an elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinMask = cv2.erode(skinMask, kernel, iterations=2)
    skinMask = cv2.dilate(skinMask, kernel, iterations=2)

    # Blur the mask to help remove noise
    # Apply the mask to the frame
    skinMask = cv2.GaussianBlur(skinMask, (5, 5), 0)
    skin = cv2.bitwise_and(frame, frame, mask=skinMask)

    # Show the skin in the image along with the mask
    cv2.imshow("images", np.hstack([frame, skin]))

    # If the key 'q' is pressed, stop the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()