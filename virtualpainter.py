import cv2
import os
import HandTrackingModule as htm
import numpy as np

# Brush and eraser thickness
brushThickness = 15
eraserThickness = 100

# Folder path for header images
folderpath = "C:\\Users\\dell\\Downloads\\minor\\headertrail"
myList = os.listdir(folderpath)

overlaylist = []

# Load all images from the folder and append to overlaylist
for impath in myList:
    image = cv2.imread(f'{folderpath}/{impath}')
    
    # Check if the image is successfully loaded
    if image is not None:
        overlaylist.append(image)
    else:
        print(f"Error loading image: {impath}")

# Check if at least one image is loaded
if len(overlaylist) == 0:
    print("No images found in folder.")
    exit()

header = overlaylist[0]  # Set the first image as the header
drawColor = (255, 0, 255)  # Default color set to red

# Initialize webcam
cap = cv2.VideoCapture(0)  # Change index if needed for different camera
cap.set(3, 1280)  # Set width
cap.set(4, 720)   # Set height
detector = htm.handDetector(detectionCon=0.85)
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

# Smoothing factor for better drawing
smoothening = 5

# Stack for undo and redo functionality
undoStack = []
redoStack = []

while True:
    success, img = cap.read()
    
    if not success:
        print("Failed to capture image from webcam.")
        break

    img = cv2.flip(img, 1)  # Flip the image horizontally for natural behavior
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # Tip of index and middle finger
        x1, y1 = lmList[8][1:]  # Index finger tip
        x2, y2 = lmList[12][1:]  # Middle finger tip
        fingers = detector.fingerUp()

        # Selection Mode: Both index and middle fingers are up
        if fingers[1] == 1 and fingers[2] == 1:
            xp, yp = 0, 0  # Reset previous points
            print("Selection Mode")
            
            # Check if hand is within the header area
            if y1 < 125:
                if 210 < x1 < 262:  # Violet
                    header = overlaylist[0]
                    drawColor = (255, 0, 255)
                    print("Selected Violet")
                elif 372 < x1 < 428:  # Yellow
                    header = overlaylist[1]
                    drawColor = (0, 255, 255)
                    print("Selected Yellow")
                elif 550 < x1 < 604: # Blue
                    header = overlaylist[2]
                    drawColor = (255, 100, 1)
                    print("Selected Blue")
                elif 753< x1 < 845:  # Eraser (Black)
                    header = overlaylist[3]
                    drawColor = (0, 0, 0)
                    print("Selected Eraser")
                elif 931 < x1 < 988:  # Undo
                    header = overlaylist[4]  # Set to the Undo icon in the overlay list
                    if len(undoStack) > 0:
                        redoStack.append(imgCanvas.copy())  # Save current state for redo
                        imgCanvas = undoStack.pop()  # Pop the last state for undo
                        print("Undo Action")
                elif 1035 < x1 < 1098:  # Redo
                    header = overlaylist[5]  # Set to the Undo icon in the overlay list
                    if len(redoStack) > 0:
                        undoStack.append(imgCanvas.copy())  # Save current state for undo
                        imgCanvas = redoStack.pop()  # Pop the redo state
                        print("Redo Action")
                elif 1167 < x1 < 1232:  # Clear Screen
                    header = overlaylist[6]  # Set to the Undo icon in the overlay list
                    imgCanvas = np.zeros((720, 1280, 3), np.uint8)  # Clear the canvas
                    print("Clear Screen Action")
                    
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

        # Drawing Mode: Only index finger is up
        elif fingers[1] == 1 and fingers[2] == 0:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            print("Drawing Mode")

            # Start drawing
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            # Apply smoothing effect
            x1 = int(xp + (x1 - xp) / smoothening)
            y1 = int(yp + (y1 - yp) / smoothening)

            # Save the current state for undo before drawing
            undoStack.append(imgCanvas.copy())
            redoStack.clear()  # Clear redo stack after a new drawing action

            # Erase if the color is black
            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1  # Update previous points

    # Combine canvas with the frame
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    
    # Ensure dimensions match
    if img.shape != imgInv.shape:
        imgInv = cv2.resize(imgInv, (img.shape[1], img.shape[0]))

    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # Overlay the header on the image
    if img.shape[0] >= 125 and img.shape[1] >= 1280:
        img[0:125, 0:1280] = header  # Overlay the header on the image

    # Display the image
    cv2.imshow("Image", img)
    cv2.imshow("Canvas", imgCanvas)
    img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)  # Improved blending for visibility

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
