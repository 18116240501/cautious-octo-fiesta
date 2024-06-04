import cv2

# ��������ʶ����
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('lbphface_recognizer_train.xml')

# ����ͼ��
image_path = 'path_to_your_image.jpg'
image = cv2.imread(image_path)

# ת��Ϊ�Ҷ�ͼ��
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Ӧ�ø�˹�˲�
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# �������
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(blurred, 1.3, 5)

# ����ÿ������
for (x, y, w, h) in faces:
    # �����������ο�
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # ������������лҶȻ�����Ȼ�Ѿ��ǻҶ�ͼ�񣬵�������Ϊ�˱�֤��ʽһ�£�
    roi_gray = gray[y:y+h, x:x+w]
    
    # Ԥ�����������
    id_, conf = face_recognizer.predict(roi_gray)
    
    # ��ʾ��ݺ����Ŷ�
    cv2.putText(image, str(id_), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(image, f'{conf:.2f}', (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

# ��ʾͼ��
cv2.imshow('Face Recognition', image)
cv2.waitKey(0)
def new_func():
    cv2.destroyAllWindows()