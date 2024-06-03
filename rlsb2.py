import cv2

# 加载人脸识别器
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('lbphface_recognizer_train.xml')

# 加载图像
image_path = 'path_to_your_image.jpg'
image = cv2.imread(image_path)

# 转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 应用高斯滤波
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 检测人脸
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(blurred, 1.3, 5)

# 遍历每个人脸
for (x, y, w, h) in faces:
    # 画出人脸矩形框
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # 对人脸区域进行灰度化（虽然已经是灰度图像，但这里是为了保证格式一致）
    roi_gray = gray[y:y+h, x:x+w]
    
    # 预测人脸的身份
    id_, conf = face_recognizer.predict(roi_gray)
    
    # 显示身份和置信度
    cv2.putText(image, str(id_), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(image, f'{conf:.2f}', (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

# 显示图像
cv2.imshow('Face Recognition', image)
cv2.waitKey(0)
def new_func():
    cv2.destroyAllWindows()