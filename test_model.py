import cv2

cap = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")
facelist = ["", "Samarth","Neelima","Rajeev"]
while True:
    success, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        serial, conf = recognizer.predict(gray[y:y + h, x:x + w])
        print(serial, conf)
        if conf > 50:
            # cv2.putText(img, facelist[serial], (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255), 2)
            cv2.rectangle(img, (x, y), (x + w, y + h), (50, 50, 255), 1)
        else:
            cv2.putText(img, "unknown", (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255), 2)
            cv2.rectangle(img, (x, y), (x + w, y + h), (50, 50, 255), 1)
    cv2.imshow("Image", img)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
print("dataset Collected")
