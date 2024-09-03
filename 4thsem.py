from fer import FER                                                                        #(dataset)
import cv2                                                                                 #(only python)
from deepface import DeepFace

model = DeepFace.build_model("Emotion")                                                     #(build model is a function)

emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    
    ret, frame = cap.read()
    
    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                                    #(colour detector variable)

    # Detect faces in the frame                                                             #(cascade is a classifier)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))  #(x and y axis min is 30 min size of object)
                                                                                            #(adjecent x and y axis is neighbour)
                                                                                            #(tells how much the object's size is reduced in each image)
    for (x, y, w, h) in faces:                                                              #(x and y axis)

        face_roi = gray_frame[y:y + h, x:x + w]                                             #(region of interest)
        resized_face = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)         

        # Normalize the resized face image
        normalized_face = resized_face / 255.0                                              #(255 is maximum pixel intensity for an 8-bit image.)

        # Reshape the image to match the input shape of the model
        reshaped_face = normalized_face.reshape(1, 48, 48, 1)

        # Predict emotions using the pre-trained model
        preds = model.predict(reshaped_face)[0]
        emotion_idx = preds.argmax()                                                          #(maximum times which emotion is showed)
        emotion = emotion_labels[emotion_idx]                                                 #(maximum emotion is printed)

        # Draw rectangle around face and label with predicted emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        print(emotion)
        
    # Display the resulting frame
    cv2.imshow('Real-time Emotion Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):                                                       #(exit key)
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()