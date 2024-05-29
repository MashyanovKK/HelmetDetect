import cv2


cap = cv2.VideoCapture('video.mp4')
i=0
while True:
    ret,frame = cap.read()
    if not ret:
        break
    cv2.imwrite(f'images_from_video_{i}.jpg',frame)
    cv2.waitKey(1)