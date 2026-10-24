from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# تحميل الموديل الأساسي
model = YOLO("best_yolo11_model.pt")

# اختبر أي صورة من الصورتين (car_test.jpg أو car_test2.png)
img_path = "paintscratch.png" 

# 👈 السر هنا: conf=0.12 مع استبعاد مربع no_damage (classes 1 to 7)
results = model.predict(img_path, conf=0.12, classes=[1, 2, 3, 4, 5, 6, 7])[0]

# رسم النتائج
res_img = results.plot()
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB))
plt.title(f"Detected {len(results.boxes)} Damages!", fontsize=14, fontweight='bold', color='darkred')
plt.axis("off")
plt.show()


