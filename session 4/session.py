import os 
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# -----------------------------------------------------#
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import * 

model = YOLO('../YOLO-weights/yolov8n.pt')
tracker = Sort(max_age= 20)

def add_to_file(car_id_inforamtion) :
    with open('counter.txt' , "a") as file :
        file.write(car_id_inforamtion + "\n")

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
limits = [400 , 297 ,673 , 297]
TotalCounts = []

while True :
    _ , frame = cap.read()
    
    frame_ragion = cv2.bitwise_and(frame , mask )
    
    results = model(frame_ragion , stream=True )
    img_graphics = cv2.imread("graphics.png" , cv2.IMREAD_UNCHANGED)
    frame = cvzone.overlayPNG(frame, img_graphics , (0 ,0 ))
    
    detections = np.empty((0,5))
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

            if object_name == "car" or object_name == "bus" :
                # cvzone.cornerRect(frame, (x1,y1,w,h) ,l=7 , colorC=(0,0 ,255) , colorR=(255,0,0))
                # cvzone.putTextRect(frame , f"{object_name}, {conf}%" , (x1,y1-10),scale=1, thickness=2 ,offset=2)
                current_detections = np.array([x1,y1,x2,y2 , conf ])
                detections = np.vstack((detections, current_detections ))
                
    result_tracker =tracker.update(detections)
    cv2.line(frame , (limits[0] , limits[1]) ,(limits[2] , limits[3]), (0,0, 255) , 5 )

    for result in result_tracker :
        x1,y1,x2,y2, ID = result
        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
        w = x2-x1
        h = y2-y1
        cvzone.cornerRect(frame, (x1,y1,w,h) ,l=7 , colorC=(0,0 ,255) , colorR=(255,0,0))
        cvzone.putTextRect(frame , f"ID : {ID}" , (x1,y1-10),scale=1, thickness=2 ,offset=2)

        cx,cy = x1+(w//2) , y1+(h//2)
        cv2.circle(frame , (cx,cy) , 5 , (255 , 0,0) ,-1)



        if limits[0] < cx < limits[2]  and limits[1]-15 < cy < limits[3]+15 :
            cv2.line(frame , (limits[0] , limits[1]) ,(limits[2] , limits[3]), (0,255, 0) , 5 )
            if TotalCounts.count(ID) == 0 :
                TotalCounts.append(ID)
                car_id_inforamtion = f"Car id {ID} , car index : {len(TotalCounts)}"
                add_to_file(car_id_inforamtion)
                
            
      
    # cvzone.putTextRect(frame , f"Numbers of Cars : {len(TotalCounts)}" , (50,50),scale=4, thickness=2 ,offset=2)
    cv2.putText(frame, str(len(TotalCounts)) , (255,100) , cv2.FONT_HERSHEY_SIMPLEX , 4 , (255,0,0) , 5)
            
    cv2.imshow("video" , frame )
    
    # cv2.imshow("video with make" , frame_ragion )
    # cv2.imshow("frameMask" , frameMask )
    
    if cv2.waitKey(1) & 0xFF ==ord('q') :
        break

cap.release()
cv2.destroyAllWindows()
