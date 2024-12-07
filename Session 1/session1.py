import os 
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# -----------------------------------------------------#

from ultralytics import YOLO
import cv2

model = YOLO('../YOLO-weights/yolov8l.pt')

result = model('img/8.jpg' , show=True)
cv2.waitKey(0)