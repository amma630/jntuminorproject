import cv2
import os
import HandTrackingModule as htm
import numpy as np
# Correct folder path variable name
brushThickness=15
eraserThickness=100
folderpath = "header"
myList = os.listdir(folderpath)
#print(myList)

overlaylist = []

# Load all images from the folder and append to overlaylist
for impath in myList:
    image = cv2.imread(f'{folderpath}/{impath}')
    
    # Check if the image is successfully loaded
    if image is not None:
        overlaylist.append(image)
    else:
        print(f"Error loading image: {impath}")

#print(f"Number of header images loaded: {len(overlaylist)}")

# Check if at least one image is loaded
if len(overlaylist) == 0:
    print("No images found in folder.")
    exit()

header = overlaylist[0]  # Set the first image as the header
drawColor=(255,0,255)

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Set width
cap.set(4, 720)   # Set height
detector = htm.handDetector(min_detection_confidence=0.85)  # Corrected 'detoctionCon' to 'detectionCon'
xp,yp=0,0
imgCanvas=np.zeros((720,1280,3),np.uint8)
while True:
    success, img = cap.read()
    img=cv2.flip(img,1)
    img=detector.findHands(img)
    lmList=detector.findPosition(img,draw=False)
    if len(lmList)!=0:
        
        #print(lmList)
        #tip of index and middle finger
        x1,y1=lmList[8][1:]
        #for middle finger
        x2,y2=lmList[12][1:]
        fingers=detector.fingerUp()
        #print(fingers)
        if fingers[1]==1 and fingers[2]==1:
            xp,yp=0,0
            print("selection Mode")
            if y1<125:
                if 250<x1<450:
                    header=overlaylist[0]
                    drawColor=(255,0,255)
                elif 550<x1<750:
                    header=overlaylist[1]
                    drawColor=(255,0,0)
                elif 800<x1<950:
                    header=overlaylist[2]
                    drawColor=(255,255,0)
                elif 1050<x1<1200:
                    header=overlaylist[3]
                    drawColor=(0,0,0)
            cv2.rectangle(img,(x1,y1-25),(x2,y2+25),drawColor,cv2.FILLED)
        if fingers[1]==1 and fingers[2]==0:
            cv2.circle(img,(x1,y1),15,drawColor,cv2.FILLED)
            print("Drawing Mode")
            if xp==0 and yp==0:
                xp,yp=x1,y1
            if drawColor ==(0,0,0):
                cv2.line(img,(xp,yp),(x1,y1),drawColor,eraserThickness)
                cv2.line(imgCanvas,(xp,yp),(x1,y1),drawColor,eraserThickness)
            
            else:
                cv2.line(img,(xp,yp),(x1,y1),drawColor,brushThickness)
                cv2.line(imgCanvas,(xp,yp),(x1,y1),drawColor,brushThickness)
            xp,yp=x1,y1
    imgGray=cv2.cvtColor(imgCanvas,cv2.COLOR_BGR2GRAY)
    _,imgInv=cv2.threshold(imgGray,50,255,cv2.THRESH_BINARY_INV)
    imgInv=cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    img=cv2.bitwise_and(img,imgInv)
    img=cv2.bitwise_or(img,imgCanvas)
    if not success:
        print("Failed to capture image from webcam.")
        break

    # Ensure the image dimensions match the header
    if img.shape[0] >= 125 and img.shape[1] >= 1280:
        img[0:125, 0:1280] = header  # Overlay the header on the image

    # Display the image
    
    cv2.imshow("Image", img)
    cv2.imshow("Canvas", imgCanvas)
    img=cv2.addWeighted(img,0.5,imgCanvas,0.5,0)
    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
