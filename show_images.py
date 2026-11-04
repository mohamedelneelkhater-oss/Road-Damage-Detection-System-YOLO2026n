import os
import random
import cv2
import matplotlib.pyplot as plt

# 1. حدد مسار مجلد الصور الخاص بالبيانات لديك (مثال لمسار شائع في YOLO)
# قم بتغييره حسب المسار الفعلي للمجلد في جهازك أو مستودعك
image_dir = 'datasets/road-damage/train/images' 

# 2. جلب جميع أسماء الصور من المجلد
all_images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# التأكد من وجود صور كافية في المجلد
if len(all_images) < 30:
    print(f"المجلد يحتوي على {len(all_images)} صورة فقط، سيتم عرضها بالكامل.")
    sampled_images = all_images
else:
    # اختيار 30 صورة عشوائياً بدون تكرار
    sampled_images = random.sample(all_images, 30)

# 3. إعداد شبكة العرض (Grid) بمقاس 5 أسطر × 6 أعمدة ليعطينا إجمالي 30 صورة
fig, axes = plt.subplots(5, 6, figsize=(20, 16))
axes = axes.ravel() # تحويل المصفوفة إلى أحادية ليسهل المرور عليها

for i, img_name in enumerate(sampled_images):
    img_path = os.path.join(image_dir, img_name)
    
    # قراءة الصورة بواسطة OpenCV
    img = cv2.imread(img_path)
    # تحويل الألوان من BGR (الخاصة بـ OpenCV) إلى RGB لتعرضها Matplotlib بشكل صحيح
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # عرض الصورة في الخانة المخصصة لها
    axes[i].imshow(img)
    axes[i].set_title(img_name[:15], fontsize=8) # عرض أول 15 حرف من اسم الصورة كعنوان
    axes[i].axis('off') # إخفاء محاور الإحداثيات (الـ x والـ y) للمظهر الجمالي

# إخفاء المربعات المتبقية الفارغة إذا كانت الصور أقل من 30
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show() # هذا الأمر سيقوم بفتح نافذة منبثقة تفاعلية بالصور على جهازك مباشرة
