from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import numpy as np
import os


class Camera:

    def __init__(self):
        self.images = ''
        self.get_images_list()

    def get_images_list(self):
        files = os.listdir('images')
        files = [i for i in files if i.find('.txt') == -1]
        self.images = files
    
    def get_images(self):
        for i in self.images:
            yield cv2.imread(f'images/{i}')


class Detector:
    def __init__(self):
        self.model = YOLO("best.pt")
        self.CONFIDENCE_THRESHOLD = 0.7
        self.keyDict = {0:'person', 1:'helmet'}

    def detection(self, frame):
        detections = self.model(frame, verbose=False)[0] 
        detected = detections.boxes.data.tolist()
        
        results = {1:[], 0:[]}
        for data in detected:
                    if data[4] < self.CONFIDENCE_THRESHOLD:
                        continue
                    xmin,ymin, xmax, ymax, class_id = int(data[0]), int(data[1]), int(data[2]), int(data[3]), int(data[5])
                    results[class_id].append([xmin, ymin, xmax, ymax, float(data[4])])
                    if class_id == 1:
                        cv2.rectangle(frame, (xmin,ymin), (xmax, ymax), (255, 0, 0), 2)
                        cv2.putText(frame, f'{self.keyDict[class_id]}, confidence: {data[4]}', (xmin,ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        frame = self.analyze(frame, results)
        return frame

    def analyze(self, frame, detections):
        
        red = (0, 0, 255)
        green = (0, 255, 0)
        if not len(detections[1]):
            for i in detections[0]:
                cv2.rectangle(frame, (i[0], i[1]), (i[2], i[3]), red, 2)
                cv2.putText(frame, f'{self.keyDict[0]}, confidence: {i[4]}', (i[0], i[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, red, 2)
        else:
            for person in detections[0]:
                hats = []
                personCenter = person[0]+(person[2] - person[0])/2, person[1]+(person[3] - person[1])/2
                for hat in detections[1]:
                    hatCenter = (hat[0]+(hat[2] - hat[0])/2, hat[1]+(hat[3] - hat[1])/2)
                    hatDist = np.sqrt((personCenter[0] - hatCenter[0])**2+(personCenter[1] - hatCenter[1])**2)
                    hats.append(hat.copy()+[hatDist])

                hats = sorted(hats, key = lambda x : x[5],reverse=False)
                hat = hats[0]
                hatCenter = (hat[0]+(hat[2] - hat[0])/2, hat[1]+(hat[3] - hat[1])/2)
                if hatCenter[0] >= person[0] and hatCenter[0] <= person[2] and  hatCenter[1] >= person[1] and hatCenter[1] <= person[3]:
                    cv2.rectangle(frame, (person[0], person[1]), (person[2], person[3]), green, 2)
                    cv2.putText(frame, f'{self.keyDict[0]}, confidence: {person[4]}', (person[0], person[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, green, 2)
                else:
                    '''
                    TODO:
                    additional classifier
                    '''
                    
                    cv2.rectangle(frame, (person[0], person[1]), (person[2], person[3]), red, 2)
                    cv2.putText(frame, f'{self.keyDict[0]}, confidence: {person[4]}', (person[0], person[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, red,2 )
        return frame       


class Main:
    def __init__(self):
        self.camera = Camera()
        self.detector = Detector()
        generator=self.camera.get_images()
        i = 0
        while True:
            try:
                detected_frame = self.detector.detection(next(generator))
                cv2.imshow('image', detected_frame)
                cv2.imwrite(f'results/{i}.jpg', detected_frame)
                i+=1
                cv2.waitKey(1000)
            except StopIteration:
                break

if __name__ == '__main__':
    main = Main()