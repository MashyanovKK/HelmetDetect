import cv2


cap = cv2.VideoCapture('video.mp4')
i=0
j=0
while True:
    ret,frame = cap.read()
    if not ret:
        break
    if j==29:
        cv2.imwrite(f'images_from_video1/{i}.jpg',frame)
        cv2.waitKey(1)
        j=0
    j+=1
    i+=1