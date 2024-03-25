import cv2
import mediapipe as mp
import time 

cap = cv2.VideoCapture('face.mp4')
pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.75)


while True:
    success, img = cap.read()
    #window size
    height, width, _ = img.shape
    max_width = 1000  
    max_height = 800 
    scale = min(max_width / width, max_height / height)
    img = cv2.resize(img, None, fx=scale, fy=scale) 

    #detecting draw 
    imgRGB = cv2.cvtColor(img , cv2.COLOR_RGB2BGR)
    result = faceDetection.process(imgRGB)
    if result.detections:
        for id, detection in enumerate(result.detections):
            bboxC = detection.location_data.relative_bounding_box
            ih,iw,ic = img.shape
            bbox = int(bboxC.xmin * iw) , int(bboxC.ymin * ih), \
                   int(bboxC.width * iw) , int(bboxC.height * ih)
            cv2.rectangle(img, bbox , (17, 219, 13), 2)
    #FPS frames per second
    cTime = time.time()
    fps = 1/ (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f"FPS : {int(fps)}", (20,70), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 2)

    cv2.imshow("Image",img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

