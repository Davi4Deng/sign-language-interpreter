import cv2
import numpy as np
import mediapipe as mp
import time
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input

# 定义模型相关参数
imageSize = 50
map_characters = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'del', 27: 'nothing', 28: 'space'
}

# Load the trained model
def load_trained_model():
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(imageSize, imageSize, 3))
    x = base_model.output
    x = Flatten()(x)
    predictions = Dense(len(map_characters), activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze the pre-training weight and train only the custom layer
    for layer in base_model.layers:
        layer.trainable = False

    return model

model = load_trained_model()

# Mediapipe Hand detection Settings
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.4, min_tracking_confidence=0.4)

# OpenCV Settings
cap = cv2.VideoCapture(0)
pTime = time.time()  # Initialize to the current time

while True:
    ret, frame = cap.read()
    if not ret:
        print("Unable to capture camera image")
        break

    # Mediapipe Process image 
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Mediapipe Using RGB images
    results = hands.process(rgb_frame)

    # Extract the area when the hand is detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand key points and lines on the image
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Gets the hand frame
            h, w, c = frame.shape
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)

            # Extend the frame to include the entire hand
            padding = 100
            x_min = max(0, x_min - padding)
            x_max = min(w, x_max + padding)
            y_min = max(0, y_min - padding)
            y_max = min(h, y_max + padding)

            # Control recognition every few seconds
            cTime = time.time()
            if cTime - pTime >= 3:  # Prediction every 3 seconds
                pTime = cTime  

                # Extract hand area
                if (x_max - x_min > 0) and (y_max - y_min > 0):
                    hand_roi = frame[y_min:y_max, x_min:x_max]

                    # Pretreat extracted hand area
                    hand_roi = cv2.resize(hand_roi, (imageSize, imageSize))
                    img_array = img_to_array(hand_roi)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = preprocess_input(img_array)

                    # Model prediction
                    predictions = model.predict(img_array)
                    predicted_class = np.argmax(predictions)
                    predicted_label = map_characters[predicted_class]

                    # Show forecast results
                    print(f"Prediction: {predicted_label}")
                    cv2.putText(frame, f"Prediction: {predicted_label}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    # Show forecast results
    cv2.imshow("Hand Sign Recognition", frame)

    # Press the 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
