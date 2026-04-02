import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.spatial import distance
import threading
import winsound
from ultralytics import YOLO
from fer import FER
from facenet_pytorch import InceptionResnetV1
import torch
import os



###############################
# Alarm function
###############################

alarm_active = False
alarm_thread_running = False


def sound_alarm():

    global alarm_active
    global alarm_thread_running

    alarm_thread_running = True

    while alarm_active:
        winsound.Beep(1200, 600)

    alarm_thread_running = False


def start_alarm():

    global alarm_active
    global alarm_thread_running

    if not alarm_thread_running:

        alarm_active = True

        threading.Thread(
            target=sound_alarm,
            daemon=True
        ).start()


def stop_alarm():

    global alarm_active
    alarm_active = False


###############################
# EAR calculation function
###############################

def calculate_EAR(eye):

    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])

    return (A + B) / (2.0 * C)


###############################
# MAR calculation function
###############################

def calculate_MAR(mouth):

    vertical = distance.euclidean(mouth[0], mouth[1])
    horizontal = distance.euclidean(mouth[2], mouth[3])

    return vertical / horizontal

###############################
# FACE EMBEDDING FUNCTION
###############################

def get_face_embedding(frame):

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    h, w, _ = frame_rgb.shape

    face_crop = frame_rgb[
        int(h*0.2):int(h*0.8),
        int(w*0.2):int(w*0.8)
    ]

    face_crop = cv2.resize(face_crop,(160,160))

    face_tensor = torch.tensor(face_crop / 255.0).permute(2,0,1).unsqueeze(0).float()

    face_tensor = torch.nn.functional.interpolate(
        face_tensor,
        size=(160,160),
        mode='bilinear'
    )

    embedding = facenet_model(face_tensor).detach()

    return embedding

###############################
# MediaPipe initialization
###############################

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


###############################
# YOLO PHONE DETECTION MODEL
###############################

phone_model = YOLO("yolov8s.pt")
phone_model.to("cpu")


###############################
# Emotion Detection Model
###############################

emotion_detector = FER(mtcnn=True)

###############################
# DRIVER IDENTITY RECOGNITION MODEL (FaceNet)
###############################
facenet_model = InceptionResnetV1(pretrained='vggface2').eval()

KNOWN_FACE_PATH = "driver_face.pt"

driver_authenticated = False
reference_embedding = None

###############################
# Landmark indexes
###############################

LEFT_EYE = [33,160,158,133,153,144]
RIGHT_EYE = [362,385,387,263,373,380]
MOUTH = [13,14,78,308]

NOSE = 1
CHIN = 152
LEFT_IRIS = 468
RIGHT_IRIS = 473


###############################
# Detection variables
###############################

baseline_EAR = []
FRAME_COUNTER = 0
ALARM_ON = False

phone_detected = False
phone_detected_frames = 0

CONSEC_FRAMES = 20
last_emotion = "Neutral"
distraction_counter = 0
blink_counter = 0
total_blinks = 0
gaze_stable_frames = 0
fatigue_score = 0

BLINK_WARNING_LIMIT = 30
EXTREME_BLINK_LIMIT = 45

baseline_head_distance = None
threshold = 0

###############################
# CNN EYE STATE MODEL (MobileNetV2)
###############################

import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2

device = torch.device("cpu")

cnn_eye_model = mobilenet_v2(pretrained=True)

cnn_eye_model.classifier[1] = torch.nn.Linear(
    cnn_eye_model.last_channel,
    2
)

cnn_eye_model = cnn_eye_model.to(device)
cnn_eye_model.eval()
cnn_eye_model.load_state_dict(torch.load("eye_model.pt"))


eye_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])



eye_closed_prob = 0

#########################################
# LSTM TEMPORAL FATIGUE MODEL IMPORTS
#########################################

from collections import deque
import torch.nn as nn

#########################################
# LSTM TEMPORAL FATIGUE PREDICTOR MODEL
#########################################

class FatigueLSTM(nn.Module):

    def __init__(self):

        super(FatigueLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=2,   # EAR + eye_closed_prob
            hidden_size=32,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(32, 1)

        self.sigmoid = nn.Sigmoid()


    def forward(self, x):

        _, (hidden, _) = self.lstm(x)

        output = self.fc(hidden[-1])

        return self.sigmoid(output)

#########################################
# TEMPORAL FATIGUE BUFFER (LSTM INPUT)
#########################################

sequence_length = 75

ear_sequence = deque(maxlen=sequence_length)

cnn_sequence = deque(maxlen=sequence_length)

fatigue_lstm_model = FatigueLSTM()

fatigue_lstm_model.eval()

###############################
# FATIGUE LOG INITIALIZATION
###############################

LOG_FILE = os.path.join(os.getcwd(), "fatigue_log.csv")

if not os.path.isfile(LOG_FILE):

    pd.DataFrame(columns=[
        "timestamp",
        "EAR",
        "MAR",
        "fatigue_score",
        "emotion",
        "phone_detected",
        "lane_drift_probability"
    ]).to_csv(LOG_FILE, index=False)

###############################
# LOAD DRIVER REFERENCE FACE
###############################

if os.path.exists(KNOWN_FACE_PATH):

    reference_embedding = torch.load(KNOWN_FACE_PATH)

    print("Known driver loaded successfully.")

###############################
# Start webcam
###############################

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)

print("Driver Monitoring System Started")

frame_count = 0
no_driver_counter = 0


while True:

    ret, frame = cap.read()

    if not ret:
        break


    ###############################
    # NIGHT MODE IMAGE ENHANCEMENT
    ###############################

    gray_check = np.mean(frame)

    if gray_check < 80:   # only enhance in low light

        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)

        limg = cv2.merge((cl,a,b))
        frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


    ###############################
    # FACE DETECTION
    ###############################

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)


    ###############################
    # DRIVER PRESENCE CHECK
    ###############################

    if not results.multi_face_landmarks:

        no_driver_counter += 1

        if no_driver_counter > 40:

            cv2.putText(
                frame,
                "DRIVER NOT VISIBLE!",
                (20,260),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,0,255),
                3
            )

            start_alarm()

        cv2.imshow("Driver Monitoring System", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        continue


    no_driver_counter = 0
    stop_alarm()


    ###############################
    # MULTIPLE FACE CHECK
    ###############################

    face_count = len(results.multi_face_landmarks)

    if face_count > 1:

        cv2.putText(
            frame,
            "MULTIPLE FACES DETECTED",
            (20,420),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            3
        )


    ###############################
    # EXTRACT LANDMARKS
    ###############################

    mesh_points = np.array([
        [p.x * frame.shape[1], p.y * frame.shape[0]]
        for p in results.multi_face_landmarks[0].landmark
    ])

    ###############################
    # DRIVER AUTHENTICATION CHECK
    ###############################

    current_embedding = None

    if reference_embedding is None and frame_count > 30:

        reference_embedding = get_face_embedding(frame)

        torch.save(reference_embedding, KNOWN_FACE_PATH)

        cv2.putText(
            frame,
            "Driver Registered",
            (20,420),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            3
        )

    elif frame_count % 120 == 0:

        current_embedding = get_face_embedding(frame)

        similarity = torch.cosine_similarity(
            reference_embedding,
            current_embedding
        ).item()

        if similarity > 0.75:

            driver_authenticated = True
            stop_alarm()

            cv2.putText(
                frame,
                "Driver Verified",
                (20,420),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                3
            )

        else:

            driver_authenticated = False

            if frame_count > 120:

                cv2.putText(
                    frame,
                    "Unknown Driver!",
                    (20,420),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,0,255),
                    3
                )

                start_alarm()

    ###############################
    # EMOTION DETECTION (Optimized)
    ###############################

    if frame_count % 30 == 0:

        emotion_result = emotion_detector.detect_emotions(frame)

        if emotion_result:

            emotions = emotion_result[0]["emotions"]

            dominant_emotion = max(emotions, key=emotions.get)

            last_emotion = dominant_emotion


    if "last_emotion" in locals():

        cv2.putText(
            frame,
            f"Emotion: {last_emotion}",
            (400,260),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255,200,0),
            2
        )

        if last_emotion in ["angry", "sad"]:

            cv2.putText(
                frame,
                "STRESS DETECTED",
                (400,300),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,0,255),
                3
            )


    ###############################
    # YOLO PHONE DETECTION
    ###############################

    frame_count += 1

    if frame_count % 30 == 0:

        yolo_results = phone_model.predict(
            frame,
            conf=0.25,
            device="cpu",
            verbose=False
        )

        phone_detected = False

        for r in yolo_results:
            for box in r.boxes:

                cls = int(box.cls[0])

                confidence = float(box.conf[0])

                if phone_model.names[cls] == "cell phone" and confidence > 0.6:

                    phone_detected = True
                    phone_detected_frames = 5

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)

                    cv2.putText(
                        frame,
                        "PHONE DETECTED",
                        (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0,0,255),
                        2
                    )

    else:

        if phone_detected_frames > 0:
            phone_detected = True
            phone_detected_frames -= 1


    ###############################
    # REGION EXTRACTION
    ###############################

    left_eye = mesh_points[LEFT_EYE]
    right_eye = mesh_points[RIGHT_EYE]
    mouth = mesh_points[MOUTH]

    nose = mesh_points[NOSE]
    chin = mesh_points[CHIN]

    left_iris = mesh_points[LEFT_IRIS]
    right_iris = mesh_points[RIGHT_IRIS]

    ###############################
    # CNN EYE STATE DETECTION
    ###############################

    if frame_count % 5 == 0:

        x_min = int(min(left_eye[:,0].min(), right_eye[:,0].min()))
        x_max = int(max(left_eye[:,0].max(), right_eye[:,0].max()))

        y_min = int(min(left_eye[:,1].min(), right_eye[:,1].min()))
        y_max = int(max(left_eye[:,1].max(), right_eye[:,1].max()))

        h, w, _ = frame.shape

        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)

        eye_crop = frame[y_min:y_max, x_min:x_max]

        if eye_crop.size != 0:

            eye_tensor = eye_transform(eye_crop).unsqueeze(0).to(device)

            with torch.no_grad():

                output = cnn_eye_model(eye_tensor)

                probabilities = torch.softmax(output, dim=1)

                eye_closed_prob = probabilities[0][1].item()

    

    ###############################
    # EAR / MAR
    ###############################

    leftEAR = calculate_EAR(left_eye)
    rightEAR = calculate_EAR(right_eye)

    EAR = (leftEAR + rightEAR)/2

    MAR = calculate_MAR(mouth)


    cv2.putText(frame,f"EAR: {round(EAR,3)}",(20,50),
                cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

    cv2.putText(frame,f"MAR: {round(MAR,3)}",(20,80),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.putText(
        frame,
        f"Eye Closed Prob: {round(eye_closed_prob,2)}",
        (20,120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,255,0),
        2
    )
    
    #########################################
    # LSTM TEMPORAL FATIGUE PREDICTION
    #########################################

    ear_sequence.append(EAR)

    cnn_sequence.append(eye_closed_prob)


    if len(ear_sequence) == sequence_length:

        temporal_input = torch.tensor(
            list(zip(ear_sequence, cnn_sequence)),
            dtype=torch.float32
        ).unsqueeze(0)


        fatigue_prediction = fatigue_lstm_model(
            temporal_input
        ).item()


        cv2.putText(
            frame,
            f"LSTM Fatigue Risk: {fatigue_prediction:.2f}",
            (20, 320),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255,255,0),
            2
        )


        if fatigue_prediction > 0.65:

            cv2.putText(
                frame,
                "MICROSLEEP RISK!",
                (350,260),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,0,255),
                3
            )

            start_alarm()

    ###############################
    # GAZE DETECTION (STABLE VERSION)
    ###############################

    gaze_direction = "CENTER"

    # Horizontal gaze detection using iris vs eye corners
    eye_center_x = (left_eye[0][0] + right_eye[3][0]) / 2
    iris_center_x = (left_iris[0] + right_iris[0]) / 2

    if iris_center_x < eye_center_x - 15:
        gaze_direction = "LEFT"

    elif iris_center_x > eye_center_x + 15:
        gaze_direction = "RIGHT"


    # Vertical gaze detection using iris vertical offset
    eye_center_y = (left_eye[1][1] + right_eye[1][1]) / 2
    iris_center_y = (left_iris[1] + right_iris[1]) / 2

    if iris_center_y > eye_center_y + 12:
        gaze_direction = "DOWN"


    cv2.putText(
        frame,
        f"Gaze: {gaze_direction}",
        (400,180),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,255,0),
        2
    )


    ###############################
    # DISTRACTION TIMER (STABLE VERSION)
    ###############################

    if gaze_direction != "CENTER":
        gaze_stable_frames += 1
    else:
        gaze_stable_frames = 0

    # Trigger distraction counter only if gaze stays away consistently
    if gaze_stable_frames > 15:
        distraction_counter += 1
    else:
        distraction_counter = 0


    # Show alert only if distraction persists ~2 seconds
    if distraction_counter > 60:

        cv2.putText(
            frame,
            "DISTRACTION ALERT!",
            (350,220),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            3
        )
        start_alarm()


    ###############################
    # HEAD DROP DETECTION
    ###############################

    head_distance = distance.euclidean(nose,chin)

    if baseline_head_distance is None:
        baseline_head_distance=head_distance

    elif baseline_head_distance and head_distance < baseline_head_distance * 0.75:

        cv2.putText(frame,"HEAD DROP ALERT!",
                    (20,240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,(0,0,255),3)

        start_alarm()


    ###############################
    # PHONE FUSION ALERT
    ###############################

    if phone_detected:

        cv2.putText(frame,"PHONE USAGE CONFIRMED!",
                    (20,360),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,(0,0,255),3)

        if gaze_direction=="DOWN":

            cv2.putText(frame,"HIGH DISTRACTION RISK!",
                        (20,390),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,(0,0,255),3)

            start_alarm()


    ###############################
    # CALIBRATION
    ###############################

    if len(baseline_EAR)<30:

        baseline_EAR.append(EAR)

        cv2.putText(frame,"Calibrating...",
                    (20,110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,(0,255,255),2)

    else:

        threshold=np.mean(baseline_EAR)*0.75

        if EAR<threshold:

            FRAME_COUNTER+=1
            blink_counter+=1

            if len(baseline_EAR) >= 30 and FRAME_COUNTER >= CONSEC_FRAMES:
                start_alarm()

                cv2.putText(frame,"DROWSINESS ALERT!",
                            (20,150),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,(0,0,255),3)

        else:

            if blink_counter>=3:
                total_blinks+=1

            blink_counter=0
            FRAME_COUNTER=0

            
    ###############################
    # FATIGUE SCORE ENGINE
    ###############################

    fatigue_score=0

    if len(baseline_EAR)>=30:

        if EAR < threshold or eye_closed_prob > 0.70:
            fatigue_score += 40

        if MAR>0.6:
            fatigue_score+=20

        if total_blinks>BLINK_WARNING_LIMIT:
            fatigue_score+=20

        if baseline_head_distance and head_distance<baseline_head_distance*0.75:
            fatigue_score+=20


    score_color=(0,255,0)

    if fatigue_score>=40:
        score_color=(0,255,255)

    if fatigue_score>=60:
        score_color=(0,0,255)


    cv2.putText(frame,
                f"Fatigue Level: {fatigue_score}%",
                (20,300),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                score_color,
                3)
    
    ###############################
    # GLOBAL SAFE STATE CHECK
    ###############################

    head_drop_detected = (
        baseline_head_distance and
        head_distance < baseline_head_distance * 0.75
    )

    if (
        fatigue_score < 40
        and not phone_detected
        and distraction_counter < 60
        and not head_drop_detected
    ):
        stop_alarm()
    
    ###############################
    # VIRTUAL STEERING DRIFT ESTIMATION
    ###############################

    lane_drift_probability = 0

    if len(baseline_EAR) >= 30 and EAR < threshold:

        lane_drift_probability += 25

    if MAR > 0.6:

        lane_drift_probability += 15

    if gaze_direction != "CENTER":

        lane_drift_probability += 20

    if phone_detected:

        lane_drift_probability += 20

    if baseline_head_distance and head_distance < baseline_head_distance*0.75:

        lane_drift_probability += 20


    drift_color = (0,255,0)

    if lane_drift_probability > 40:

        drift_color = (0,255,255)

    if lane_drift_probability > 60:

        drift_color = (0,0,255)


    cv2.putText(
        frame,
        f"Lane Drift Risk: {lane_drift_probability}%",
        (20,340),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        drift_color,
        3
    )


    ###############################
    # BLINK RATE WARNINGS
    ###############################

    blink_color=(0,255,255)

    if total_blinks>BLINK_WARNING_LIMIT:

        blink_color=(0,0,255)

        cv2.putText(frame,"HIGH BLINK RATE!",
                    (400,90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,(0,0,255),2)


    if total_blinks>EXTREME_BLINK_LIMIT:

        cv2.putText(frame,"EXTREME FATIGUE BLINK LEVEL!",
                    (350,130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,(0,0,255),2)

        start_alarm()


    cv2.putText(frame,
                f"Blinks: {total_blinks}",
                (400,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                blink_color,
                2)


    ###############################
    # YAWNING DETECTION
    ###############################

    if MAR>0.6:

        cv2.putText(frame,"Yawning Detected!",
                    (20,200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,(255,0,0),3)

    ###############################
    # SAVE FATIGUE EVENT LOG
    ###############################

    log_data = {

        "timestamp": datetime.now(),

        "EAR": round(EAR,3),

        "MAR": round(MAR,3),

        "fatigue_score": fatigue_score,

        "emotion": last_emotion,

        "phone_detected": phone_detected,

        "lane_drift_probability": lane_drift_probability
    }

    if frame_count % 5 == 0 and 'EAR' in locals():
    
        print("Logging fatigue data...")

        pd.DataFrame([log_data]).to_csv(
            LOG_FILE,
            mode='a',
            header=False,
            index=False
        )

    cv2.imshow("Driver Monitoring System",frame)

    if cv2.waitKey(1)&0xFF==ord("q"):
        break

    

cap.release()
cv2.destroyAllWindows()