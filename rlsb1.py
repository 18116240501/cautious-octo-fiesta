import cv2 as cv
import matplotlib.pyplot as plt

# 1. ��ȡͼƬ��ת��Ϊ�Ҷ�ͼ
img = cv.imread("16.jpg")
if img is None:
    print("Error: Could not open or find the image.")
    exit()
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 2. ����OpenCV�������۾�ʶ��ķ�����
face_cas = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
if not face_cas.empty():
    print("Loaded face cascade classifier")
else:
    print("Error loading face cascade classifier")
    exit()

eyes_cas = cv.CascadeClassifier("haarcascade_eye.xml")
if not eyes_cas.empty():
    print("Loaded eyes cascade classifier")
else:
    print("Error loading eyes cascade classifier")
    exit()

# 3. �������
faceRects = face_cas.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
for (x, y, w, h) in faceRects:
    # �������
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
    
    # 4. ��ʶ����������н����۾��ļ��
    roi_gray = gray[y:y+h, x:x+w]
    eyes = eyes_cas.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        # �۾�λ����Ҫӳ���ԭͼ������ϵͳ
        eye_x = x + ex
        eye_y = y + ey
        cv.rectangle(img, (eye_x, eye_y), (eye_x + ew, eye_y + eh), (0, 255, 0), 2)

# 5. ��ʾ�����
plt.figure(figsize=(8, 6), dpi=100)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))  # ת��ΪRGB�Ա�matplotlib��ȷ��ʾ
plt.title('�����')
plt.axis('off')  # �ر�������
plt.show()