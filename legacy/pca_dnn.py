import numpy as np
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import cv2

dir_path = os.path.dirname(os.path.realpath(__file__))
gaf_path = dir_path + '\\data\\images\\gaf'
data = gaf_path + '\\images.npy'
labels = gaf_path + '\\labels.npy'

x = np.load(data).astype(int) / 255
y = np.load(labels).astype(int)
idx_of_sparsely_spiny = np.where(y == 2)
x = np.delete(x, idx_of_sparsely_spiny, axis=0)
y = np.delete(y, idx_of_sparsely_spiny)

b = np.zeros((x.shape[0], x.shape[1], x.shape[2]))
g = np.zeros((x.shape[0], x.shape[1], x.shape[2]))
r = np.zeros((x.shape[0], x.shape[1], x.shape[2]))

for i in range(len(x)):
    b[i], g[i], r[i] = cv2.split(x[i][:])

b = b.reshape((x.shape[0], x.shape[1]*x.shape[2]))
g = g.reshape((x.shape[0], x.shape[1]*x.shape[2]))
r = r.reshape((x.shape[0], x.shape[1]*x.shape[2]))


print("Red channel")
print(r.shape)
print("\nGreen channel")
print(g.shape)
print("\nBlue channel")
print(b.shape)


pca_r = PCA(n_components=1)
pca_r_trans = pca_r.fit_transform(r)

pca_g = PCA(n_components=1)
pca_g_trans = pca_g.fit_transform(g)

pca_b = PCA(n_components=1)
pca_b_trans = pca_b.fit_transform(b)

pca_data = np.concatenate((pca_r_trans, pca_g_trans, pca_b_trans), axis=1)


print("Explained variances by each channel")
print("-----------------------------------")
print("Red:", np.sum(pca_r.explained_variance_ratio_ ) * 100)
print("Green:", np.sum(pca_g.explained_variance_ratio_ ) * 100)
print("Blue:", np.sum(pca_b.explained_variance_ratio_ ) * 100)

# y = to_categorical(y, num_classes=2)
x_train, x_val, y_train, y_val = train_test_split(pca_data, y, train_size=0.9, random_state=1)
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.78, random_state=1)

idx = 0
for lr in [0.01, 0.001, 0.0005, 0.0001, 0.01]:
    for wd in [0.0005, 0.001, 0.01, 0.1]:
        for drop in [0, 0.2, 0.5, 0.6]:
            for size in [600, 200, 100, 10]:
                norm1 = BatchNormalization()
                dense1 = Dense(x_train.shape[1], activation='relu', kernel_regularizer=l2(wd), bias_regularizer=l2(wd))
                drop1 = Dropout(drop)
                norm2 = BatchNormalization()
                dense2 = Dense(size, activation='relu', kernel_regularizer=l2(wd), bias_regularizer=l2(wd))
                drop2 = Dropout(drop)
                dense3 = Dense(size//2, activation='relu', kernel_regularizer=l2(wd), bias_regularizer=l2(wd))
                prediction = Dense(1, activation='sigmoid')
                model = Sequential([norm1, dense1, drop1, norm2, dense2, drop2, dense3, prediction])

                opt = Adam(learning_rate=lr, decay=lr/100)
                model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
                history = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_val, y_val), verbose=0)
                loss, acc = model.evaluate(x_test, y_test, verbose=0)
                print("idx: " + str(idx), "accuracy: " + str(acc))
                idx += 1
                if acc > 0.7:
                    print(lr, wd, drop, size)
                    break

# plot history
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

print(model.predict(x_test))
print(y_test)

# TODO optimize







