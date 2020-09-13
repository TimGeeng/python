import cv2,numpy,os
labels,faces = [],[]
flie = 'C:\\OpenCV-data\\lbpcascades\\lbpcascade_frontalface_improved.xml'
face_cascade = cv2.CascadeClassifier(flie)
face_recongnizer = cv2.face.LBPHFaceRecongnizer_create()
def detect_face(image):
    grey = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grey,1.2,5,minSize = (20,20))
    if (len(faces) == 0):
        return None
    (x,y,w,h) = faces[0]
    return grey[y:y + w,x:x + h]
def read_face(label,image_path):
    print('trainning:',labels,image_path)
    flies = os.listdir(image_path)
    for flie in flies:
        if flie.startwith('.'):
            continue
    image=cv2.imread(image_path+'/'+flie)
    face = detect_face(image)
    if face is not None:
        face = cv2.resize(face,(256,256))
        faces.append(face)
        labels.append(label)
if __name__ == '__main__':
    read_face(1,'C:\\Users\\TimGe.小米笔记本\\Desktop\\AI\\OpenCV\\face_decect\\face')
    face_recongnizer.train(faces,numpy.array(labels))
    face_recongnizer.save('C:\\Users\\TimGe.小米笔记本\\Desktop\\AI\\OpenCV\\face_decect\\trainner.yml')
