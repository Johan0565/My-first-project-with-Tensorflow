import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10
from threading import activeCount
from tensorflow.keras.models import load_model
model = load_model('my_model.h5')  # Загрузка обученной модели

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

def display_image(index):
    plt.imshow(train_images[index])
    plt.title(f"Label: {train_labels[index]}")
    plt.show()

# Обучение модели  30 эпох.
#history = model.fit(train_images, train_labels, epochs=30, validation_data=(test_images, test_labels))

# Оценка модели на тестовых данных
#test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
#print(f'Test accuracy: {test_acc}')
#print(f'Test loss: {test_loss}')

# Выбираем одно изображение из тестового набора
index = 900
image = test_images[index]

# Модель возвращает вероятность для каждого класса
predictions = model.predict(np.expand_dims(image, axis=0))
predicted_label = np.argmax(predictions)  # Находим индекс с наибольшей вероятностью

# Выводим предсказанный класс и истинный класс
print(f"Predicted label: {predicted_label}")
print(f"True label: {test_labels[index][0]}")

plt.imshow(image)
plt.title(f"Predicted: {predicted_label}, True: {test_labels[index][0]}")
plt.show()


