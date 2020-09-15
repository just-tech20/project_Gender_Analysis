import cv2

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
gender_list = ['Male', 'Female']

face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)
bool = cap.isOpened()
print(bool)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)

        #Get Face 
        face_img = img[y:y+h, h:h+w].copy()  # roi -- region of image  or Technically-- Region of Interest

        # dnn-- deep neural network |  blob = cv.dnn.blobFromImage(image, scalefactor, size, mean, swapRB, crop) 
        # scaleFactor: This function compensates a false perception in size that occurs when one face appears to be bigger than the other simply because it is closer to the camera.
        # minNeighbors: Detection algorithm that uses a moving window to detect objects, it does so by defining how many objects are found near the current one before it can declare the face found.
        blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)  
            
        # .prototxt — The definition of CNN goes in here. This file defines the layers in the neural network, each layer’s inputs, outputs and functionality.
        # .caffemodel — This contains the information of the trained neural network (trained model).  
        # This way function cv::dnn::readNet can automatically detects a model's format.
        gender_net = cv2.dnn.readNetFromCaffe('data/deploy_gender.prototxt','data/gender_net.caffemodel')  
 																										   
        
        #Predict Gender
        gender_net.setInput(blob)  # Pass the input to our caffe model
        gender_preds = gender_net.forward()  # Passing our input image to through the network
        print(gender_preds)
        gender = gender_list[gender_preds[0].argmax()]  # argmax() return the position of largest value whereas max() returns the value

        cv2.waitKey(30) & 0xff
        cv2.putText(img,gender,(x+4,y-5),0,.7, (0,255,255),2, cv2.LINE_AA)

    cv2.imshow('Image',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cv2.release()
cv2.destroyAllWindows()
