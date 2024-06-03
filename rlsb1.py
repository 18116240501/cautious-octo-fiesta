import cv2 as cv
import matplotlib.pyplot as plt

# 1. 读取图片并转换为灰度图
img = cv.imread("16.jpg")
if img is None:
    print("Error: Could not open or find the image.")
    exit()
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 2. 加载OpenCV人脸和眼睛识别的分类器
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

# 3. 检测人脸
faceRects = face_cas.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
for (x, y, w, h) in faceRects:
    # 框出人脸
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
    
    # 4. 在识别出的人脸中进行眼睛的检测
    roi_gray = gray[y:y+h, x:x+w]
    eyes = eyes_cas.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        # 眼睛位置需要映射回原图的坐标系统
        eye_x = x + ex
        eye_y = y + ey
        cv.rectangle(img, (eye_x, eye_y), (eye_x + ew, eye_y + eh), (0, 255, 0), 2)

# 5. 显示检测结果
plt.figure(figsize=(8, 6), dpi=100)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))  # 转换为RGB以便matplotlib正确显示
plt.title('检测结果')
plt.axis('off')  # 关闭坐标轴
plt.show()