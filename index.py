# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 19:19:44 2021
@author: Sonu
"""

import time
import cv2
from flask import Flask, render_template, Response
import mediapipe as mp
import numpy as np
from keras.models import load_model



app = Flask(__name__)

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('templates/index.html')

def extract_keypoints(results):
    rh = np.array([[res.x, res.y, res.z] for res in results.landmark]).flatten() 
    #lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    #return np.concatenate([rh,lh])
    return rh

def gen():
    previous_time = 0
    # creating our model to draw landmarks
    mpDraw = mp.solutions.drawing_utils
    # creating our model to detected our pose
    mp_hands = mp.solutions.hands
    #pose = my_pose.Pose()

    actions = np.array(['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'])
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.9
    model = load_model('models/right_a-z.h5')

    """Video streaming generator function."""
    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        # converting image to RGB from BGR cuz mediapipe only work on RGB
        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5) as hands:
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(imgRGB)

            if result.multi_hand_landmarks:
                for idx, hand_handedness in enumerate(result.multi_handedness):
                    #print(hand_handedness.classification[0].label)
                    if hand_handedness.classification[0].label=='Left':
                        for hand_landmarks in result.multi_hand_landmarks:
                            mpDraw.draw_landmarks(
                                img,
                                hand_landmarks,
                                mp_hands.HAND_CONNECTIONS)
                            keypoints = extract_keypoints(hand_landmarks)
                            sequence.append(keypoints)
                            sequence = sequence[-30:]

                            if len(sequence) == 30:
                                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                                print(actions[np.argmax(res)])

                                if res[np.argmax(res)] > threshold: 
                                    sentence.append(actions[np.argmax(res)])
                                    sequence.clear()

                                if len(sentence) > 10: 
                                    sentence = sentence[-10:]
            else:
                cv2.putText(img, 'Hand is not detected', (120,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255, 0), 4, cv2.LINE_AA)
                sequence.clear()
                    
            cv2.putText(img, ' '.join(sentence), (3,30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        
            current_time = time.time()
            fps = 1 / (current_time - previous_time)
            previous_time = current_time

            # Writing FrameRate on video
            cv2.putText(img, 'fps:'+str(int(fps)), (550, 450), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

        #cv2.imshow("Pose detection", img)
        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        key = cv2.waitKey(20)
        if key == 27:
            break


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__=="__main__":
    app.run(debug=True)







