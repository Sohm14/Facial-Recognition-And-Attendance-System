import cv2
import face_recognition
import pickle
import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://face-attendance-system-c1854-default-rtdb.firebaseio.com/",
    'storageBucket':"face-attendance-system-c1854.appspot.com"
})

folderPath = 'Images'
PathList = os.listdir(folderPath)
imgList = []
studentIds = []
for path in PathList:
    imgList.append(cv2.imread(os.path.join(folderPath, path)))
    studentIds.append(os.path.splitext(path)[0])
    filename=f'{folderPath}/{path}'
    bucket=storage.bucket()
    blob=bucket.blob(filename)
    blob.upload_from_filename(filename)


def findencodings(imageslist):
    encodeList = []
    for img in imgList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


print("Encoding started")
encodeListKnown = findencodings(imgList)
print("Encoding finished")
encodeListKnownWithIds = [encodeListKnown, studentIds]
file = open("EncodeFile.p", 'wb')
pickle.dump(encodeListKnownWithIds, file)
file.close()
print("File saved")


