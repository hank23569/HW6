import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
import numpy as np
import requests
from PIL import Image
from io import BytesIO

# 第一步：載入預訓練 VGG16 模型
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False  # 凍結預訓練層

# 添加自訂分類層
x = Flatten()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dense(2, activation='softmax')(x)  # 針對兩個類別進行分類
model = Model(inputs=base_model.input, outputs=x)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 第二步：設定數據生成器（使用醫療口罩數據集）
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# 調整此路徑為你下載的數據集路徑
train_dir = r'C:\Users\電機系味宸漢\Desktop\hw6\Face-Mask-Detection\dataset'  # 確保這是正確的資料夾
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)
val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# 訓練模型
model.fit(train_generator, validation_data=val_generator, epochs=5)

# 第三步：建立分類函式，接受圖片 URL 作為輸入
def classify_image(image_url, model, class_names):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    img = img.resize((224, 224))  # 調整圖片大小
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions)
    return class_names[class_idx]

# 測試分類程式
image_url = "https://na.cx/i/eqzQJYw.jpg"  # 測試用圖片網址
class_names = list(train_generator.class_indices.keys())  # 獲取類別名稱
predicted_class = classify_image(image_url, model, class_names)
print(f"預測的類別是：{predicted_class}")
