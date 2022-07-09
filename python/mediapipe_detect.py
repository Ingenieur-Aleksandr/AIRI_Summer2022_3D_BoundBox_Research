import cv2
import mediapipe as mp
import time

mp_objectron = mp.solutions.objectron
mp_drwawing = mp.solutions.drawing_utils

with open('python/source/paths.txt', 'r') as paths:
    PATH_IN, PATH_OUT = list(map(lambda x: x.replace('\n', ''), paths.readlines()))


cap = cv2.VideoCapture('')
out = cv2.VideoWriter('outpy_3.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 15, (720, 1280))

with mp_objectron.Objectron(
    static_image_mode=False,
    max_num_objects=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.8,
    model_name='Shoe'
) as objectron:
    while True:
        ret, frame = cap.read()
        if ret:
            frame.flags.writeable = False
            results = objectron.process(frame)
            frame.flags.writeable = True

        #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if results.detected_objects:
                for detected_objects in results.detected_objects:
                    mp_drwawing.draw_landmarks(frame, detected_objects.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
                    mp_drwawing.draw_axis(frame, detected_objects.rotation, detected_objects.translation)

            out.write(frame)

        else:
            break