import cv2

cap = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

id = input("enter your id")
count = 0
while True:
    success, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        count = count + 1
        cv2.imwrite('datasets/user.' + str(id) + "." + str(count) + ".jpg", gray[y:y + h, x:x + w])
        cv2.rectangle(img, (x, y), (x + w, y + h), (50, 50, 255), 1)
    cv2.imshow("Image", img)
    k = cv2.waitKey(1)
    if count > 2:
        break
cap.release()
cv2.destroyAllWindows()
print("dataset Collected")
