from ultralytics import YOLO
import cv2
import numpy as np
import os


# Класс Camera для работы с изображениями из директории
class Camera:

    def __init__(self):
        self.images = ''
        self.get_images_list()  # Получаем список изображений при инициализации

    def get_images_list(self):
        # Получаем список файлов в директории 'images', исключая текстовые файлы
        files = os.listdir('images')
        files = [i for i in files if i.find('.txt') == -1]
        self.images = files
    
    def get_images(self):
        # Генератор для чтения изображений из списка
        for i in self.images:
            yield cv2.imread(f'images/{i}')


# Класс Detector для детекции объектов на изображениях
class Detector:
    def __init__(self):
        self.model = YOLO("best.pt")  # Загружаем модель YOLO
        self.CONFIDENCE_THRESHOLD = 0.7  # Устанавливаем порог уверенности для детекции
        self.keyDict = {0: 'person', 1: 'helmet'}  # Словарь классов объектов

    def detection(self, frame):
        # Выполняем детекцию на кадре и получаем результаты
        detections = self.model(frame, verbose=False)[0] 
        detected = detections.boxes.data.tolist()
        
        results = {1: [], 0: []}  # Словарь для хранения результатов детекции
        for data in detected:
            if data[4] < self.CONFIDENCE_THRESHOLD:
                continue  # Пропускаем объекты, уверенность в которых ниже порога
            xmin, ymin, xmax, ymax, class_id = int(data[0]), int(data[1]), int(data[2]), int(data[3]), int(data[5])
            results[class_id].append([xmin, ymin, xmax, ymax, float(data[4])])
            if class_id == 1:
                # Рисуем прямоугольник вокруг обнаруженной каски и добавляем текст
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
                cv2.putText(frame, f'{self.keyDict[class_id]}, confidence: {data[4]}', (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        frame = self.analyze(frame, results)
        return frame

    def analyze(self, frame, detections):
        # Анализируем детекции для определения, носит ли человек каску
        red = (0, 0, 255)
        green = (0, 255, 0)
        if not len(detections[1]):
            # Если каски не обнаружены, рисуем красные прямоугольники вокруг всех людей
            for i in detections[0]:
                cv2.rectangle(frame, (i[0], i[1]), (i[2], i[3]), red, 2)
                cv2.putText(frame, f'{self.keyDict[0]}, confidence: {i[4]}', (i[0], i[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, red, 2)
        else:
            for person in detections[0]:
                helmets = []
                personCenter = person[0] + (person[2] - person[0]) / 2, person[1] + (person[3] - person[1]) / 2
                for helmet in detections[1]:
                    helmetCenter = (helmet[0] + (helmet[2] - helmet[0]) / 2, helmet[1] + (helmet[3] - helmet[1]) / 2)
                    helmetDist = np.sqrt((personCenter[0] - helmetCenter[0]) ** 2 + (personCenter[1] - helmetCenter[1]) ** 2)
                    helmets.append(helmet.copy() + [helmetDist])

                helmets = sorted(helmets, key=lambda x: x[5], reverse=False)
                helmet = helmets[0]
                helmetCenter = (helmet[0] + (helmet[2] - helmet[0]) / 2, helmet[1] + (helmet[3] - helmet[1]) / 2)
                if helmetCenter[0] >= person[0] and helmetCenter[0] <= person[2] and helmetCenter[1] >= person[1] and helmetCenter[1] <= person[3]:
                    # Если каска находится внутри прямоугольника человека, рисуем зеленый прямоугольник
                    cv2.rectangle(frame, (person[0], person[1]), (person[2], person[3]), green, 2)
                    cv2.putText(frame, f'{self.keyDict[0]}, confidence: {person[4]}', (person[0], person[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, green, 2)
                else:
                    # Если каска не находится внутри прямоугольника человека, рисуем красный прямоугольник и запускаем дополнительный классификатор YOLO
                    helmet_frame = frame[person[1]:person[3], person[0]:person[2]]  # Извлекаем регион интереса (ROI)
                    helmet_detections = self.model(helmet_frame, verbose=False)[0]  # Выполняем детекцию на ROI
                    helmet_detected = helmet_detections.boxes.data.tolist()
                    for helmet_data in helmet_detected:
                        if helmet_data[4] >= self.CONFIDENCE_THRESHOLD and int(helmet_data[5]) == 1:
                            # Если каска найдена в ROI, рисуем зеленый прямоугольник
                            cv2.rectangle(frame, (person[0], person[1]), (person[2], person[3]), green, 2)
                            cv2.putText(frame, f'{self.keyDict[0]}, confidence: {person[4]}', (person[0], person[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, green, 2)
                            break
                    else:
                        # Если каска не найдена, рисуем красный прямоугольник
                        cv2.rectangle(frame, (person[0], person[1]), (person[2], person[3]), red, 2)
                        cv2.putText(frame, f'{self.keyDict[0]}, confidence: {person[4]}', (person[0], person[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, red, 2)
        return frame       


# Класс Main для запуска основного процесса
class Main:
    def __init__(self):
        self.camera = Camera()  # Инициализация камеры
        self.detector = Detector()  # Инициализация детектора
        generator = self.camera.get_images()  # Получаем генератор изображений
        i = 0
        while True:
            try:
                detected_frame = self.detector.detection(next(generator))  # Детекция объектов на следующем кадре
                cv2.imshow('image', detected_frame)  # Отображаем изображение
                cv2.imwrite(f'results/{i}.jpg', detected_frame)  # Сохраняем результат
                i += 1
                cv2.waitKey(1000)  # Ожидаем 1 секунду
            except StopIteration:
                break  # Завершаем цикл, когда изображения заканчиваются

# Точка входа в программу
if __name__ == '__main__':
    main = Main()