# -*- coding: utf-8 -*-
"""
@author: Dogus14
"""

from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv2D, Flatten, MaxPooling2D, BatchNormalization
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt



"""Data Yükleme ve Önişlem"""
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Pikseller normalize edilir. M,n-Max Normalization
x_train = (x_train.astype(np.float32)-127.5)/127.5
print(x_train.shape)
#x_train = x_train.astype(np.float32) / 255.0

#Data gorsellestirme
plt.imshow(x_train[14])

#veri duzenleme
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)



""" Parametreler"""
iterasyon_sayisi = 25
paket_boyutu = 64
ogrenme_orani = 1e-3
sinif_sayisi = 10 #10 rakam sinifi



"""Model oluşturma"""
class CNN:
    @staticmethod
    def olustur(width, height, depth, classes):
        #Modeli yukleyelim
        model = Sequential()
        girdi_tipi = (height, width, depth)
        
        #Eger kanal oncelikli calisiliyorsa girdi tipini buna uygun olarak guncelle.
        if K.image_data_format() == "channel_first":
            girdi_tipi = (depth, height, width)
        #Modeli olusturalim
        model.add(Conv2D(30, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal',input_shape=girdi_tipi))
        model.add(Conv2D(30, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.20))
        model.add(Conv2D(60, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))
        model.add(Conv2D(60, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(120, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(120, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
		#Siniflandirici softmax secilir. Multiclass
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model



"""Modelin yüklenmesi"""
print("Model yükleniyor...")
net = CNN.olustur(width = 28, height = 28, depth = 1, classes = 10)
opt = RMSprop(lr = ogrenme_orani, decay=ogrenme_orani / iterasyon_sayisi)
net.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
print(net.summary())
  


"""Veriler düzenlenir ve doğrulama datası oluşturulur."""
y_train = to_categorical(y_train, sinif_sayisi)
y_test = to_categorical(y_test, sinif_sayisi)
X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state=50)




"""Veri artırma tekniği uygulanır."""
datagen = ImageDataGenerator(
        featurewise_center=False,  # girdilerin ortalamasi 0 olarak ayarlanmasin.
        samplewise_center=False,  # her goruntu orneginin ortalamasi 0 olarak ayarlanmasim.
        featurewise_std_normalization=False,  # girdiler veri kumesinin std sapmasina bolunmesin
        samplewise_std_normalization=False,  # her goruntu pikseli kendi std sapmasina bolunmesin
        rotation_range=20, # goruntulerin rastgele dondurulme acisi
        zoom_range = 0.2, # goruntulere uygulanan rastgele zoom orani
        width_shift_range=0.1,  # goruntulerin yatay eksende rastgele kaydirilarak burkulma orani
        height_shift_range=0.1,  # goruntulerin dikey eksende rastgele kaydirilarak burkulma orani
        horizontal_flip=False,  # goruntuler yatay eksende rastgele dondurulmesin.
        vertical_flip=False)  # goruntuler dikey eksende rastgele dondurulmesin.



"""Modelin eğitilmesi ve değerlendirilmesi"""
datagen.fit(x_train)
h = net.fit_generator(datagen.flow(x_train,y_train, batch_size=paket_boyutu),
                              epochs = iterasyon_sayisi, validation_data = (X_val,Y_val),
                              verbose = 1, steps_per_epoch = X_train.shape[0] // paket_boyutu)



loss, acc = net.evaluate(x_test, y_test, verbose=0)
print("Toplam hata: {0:.2f}, Dogruluk: {1:.2f}".format(loss, acc))

