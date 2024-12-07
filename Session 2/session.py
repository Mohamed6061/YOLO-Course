import os 
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# -----------------------------------------------------#
from ultralytics import YOLO
import cv2
import cvzone
import math

model = YOLO('../YOLO-weights/yolov8n.pt')
# result = model(r'E:\Techno City\AI python\Advanced Computer Vision - YOLO Course\Session 1\img\3.jpg' , show= True)
# cv2.waitKey(0)

# from camera
# cap = cv2.VideoCapture(0)
# cap.set(3, 1280)
# cap.set(4, 480)
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


cap = cv2.VideoCapture(r"E:\Techno City\AI python\Advanced Computer Vision - YOLO Course\Demo Videos\bikes.mp4")
while True :
    _ , img = cap.read()
    results = model(img , stream=True )
    
    for r in results :
        boxes = r.boxes 
        for box in boxes :
            cls_name = box.cls[0]
            conf = box.conf[0]
            x1,y1,x2,y2 = box.xyxy[0]
            
            conf = math.ceil(box.conf[0] * 100) 
            if conf > 5 :
                
                # print(conf)
                object_name = classNames[int(cls_name)]
                # print(f"conf value : {conf} .")
                
                x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
                # cv2.rectangle(img , (x1,y1) ,(x2,y2) ,(0,100,100) ,2 )
                # cv2.putText(img, f"{object_name}", (x1,y1), cv2.FONT_HERSHEY_SIMPLEX , 1 , (0,255,255) , 2)
                w = x2-x1
                h = y2-y1
                cvzone.cornerRect(img, (x1,y1,w,h) ,l=7 , colorC=(0,0 ,255) , colorR=(255,0,0))
                cvzone.putTextRect(img , f"{object_name} {conf}%" , (x1,y1-10),scale=1, thickness=2 ,offset=2)
                
                # cv2.circle(img, (x1,y1) ,3 , (255,0,0) , 3)
                # cv2.circle(img, (x2,y2) ,3 , (0,255,0) , 3)
                # print(x1,y1,x2,y2)
        
    cv2.imshow("video" , img )
    if cv2.waitKey(10) & 0xFF ==ord('q') :
        break
    
cap.release()
cv2.destroyAllWindows()
