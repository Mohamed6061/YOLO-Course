import os 
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# -----------------------------------------------------#
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import * 

model = YOLO('../YOLO-weights/yolov8n.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]


cap = cv2.VideoCapture(r"E:\Techno City\Advanced Computer Vision - YOLO Course\Demo Videos\cars.mp4")
mask = cv2.imread("m.png")
# tracking
tracker = Sort(max_age=20)

while True :
    _ , img = cap.read()
    
    imgMask = cv2.bitwise_and(img , mask)
    results = model(imgMask , stream=True )
    
    
    detections = np.empty((0, 5) )
                          
    for r in results :
        boxes = r.boxes 
        for box in boxes :
            cls_name = box.cls[0] #class number
            conf = box.conf[0] # conf
            x1,y1,x2,y2 = box.xyxy[0]

            conf = math.ceil(box.conf[0] * 100) 
            object_name = classNames[int(cls_name)]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            w = x2-x1
            h = y2-y1
            
            if conf > 40 and object_name == "car" or object_name== "bus" or object== "truck" or object== "bicycle" :
                # cvzone.cornerRect(img, (x1,y1,w,h) ,l=7 , colorC=(0,0 ,255) , colorR=(255,0,0))
                # cvzone.putTextRect(img , f"{object_name} {conf}%" , (x1,y1-10),scale=1, thickness=2 ,offset=2)

                currentArray = np.array([x1,y1,x2,y2,conf])
                
                detections = np.vstack((detections, currentArray))
    
    
    resultTracker = tracker.update(detections)
        
    for result in resultTracker :
        x1,y1,x2,y2,ID = result
        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
        
        cvzone.putTextRect(img , f"{ID}%" , (x1,y1-10),scale=1, thickness=2 ,offset=2)
    cv2.imshow("video" , img )
    # cv2.imshow("imgMask" , imgMask )
    if cv2.waitKey(0) & 0xFF ==ord('q') :
        break
    
cap.release()
cv2.destroyAllWindows()
