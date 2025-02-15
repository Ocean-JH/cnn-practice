import os
from keras.src.models import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import matplotlib.pyplot as plt
import numpy as np
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.optimizers import Adam
from keras.src.callbacks import EarlyStopping


# 1. 数据加载与预处理（使用ImageDataGenerator）
def load_data(image_folder, img_size=(128, 128), batch_size=32, val_size=0.2, test_size=0.1):
    # 使用 ImageDataGenerator 加载数据
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=val_size + test_size,
        rotation_range=30,
        zoom_range=0.2,
        horizontal_flip=True
    )

    # 加载训练数据和验证数据
    train_gen = datagen.flow_from_directory(
        image_folder,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    val_test_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=test_size / (val_size + test_size))
    val_gen = val_test_datagen.flow_from_directory(
        image_folder,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    test_gen = val_test_datagen.flow_from_directory(
        image_folder,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    # 获取标签映射
    label_map = train_gen.class_indices
    return train_gen, val_gen, test_gen, label_map


# 2. 模型构建
def build_model(input_shape=(128, 128, 3)):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),

        Dense(4, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001, decay=1e-6), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# 3. 训练与评估
def train_model(model, train_gen, val_gen, epochs=25):
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=[early_stopping]
    )
    return history


# 4. 模型保存
def save_model(model, model_path='dog_emotion_model.keras'):
    model.save(model_path)
    print(f"Model saved at {model_path}")


# 5. 可视化训练过程
def plot_training_history(history):
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


# 6. 测试与评估
def eva_model(model, test_gen):
    loss, accuracy = model.evaluate(test_gen)
    print(f"eva Accuracy: {accuracy:.4f}, eva Loss: {loss:.4f}")

    # 测试集预测
    predictions = model.predict(test_gen)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = test_gen.classes

    # 可视化测试结果
    class_labels = list(test_gen.class_indices.keys())
    fig, axes = plt.subplots(1, len(test_gen.filenames), figsize=(12, 4))
    if len(test_gen.filenames) == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        img = plt.imread(test_gen.filenames[i])
        ax.imshow(img)
        true_label = class_labels[true_labels[i]]
        pred_label = class_labels[predicted_labels[i]]
        ax.set_title(f"True: {true_label}\nPred: {pred_label}")
        ax.axis("off")
    plt.show()


# 7. 完整训练流程
def train_and_evaluate(image_folder, img_size=(128, 128), epochs=25, batch_size=32):
    # 加载数据
    train_gen, val_gen, test_gen, label_map = load_data(image_folder, img_size, batch_size)

    # 构建模型
    model = build_model(input_shape=(img_size[0], img_size[1], 3))

    # 训练模型
    history = train_model(model, train_gen, val_gen, epochs)

    # 保存模型
    save_model(model)

    # 可视化训练过程
    plot_training_history(history)

    # 测试模型（使用验证集作为测试集）
    eva_model(model, test_gen)


if __name__ == '__main__':
    # 运行训练与评估
    image_folder = r'D:\python_files\dog_emotion'
    train_and_evaluate(image_folder, epochs=25, batch_size=32)
