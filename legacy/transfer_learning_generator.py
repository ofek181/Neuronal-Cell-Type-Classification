from keras.layers import Dense,GlobalAveragePooling2D, Dropout, Flatten
from keras.applications import ResNet50
from keras.applications.resnet_v2 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model


BATCH_SIZE = 32
DATA = 'gaf'
CLASSES = ['spiny', 'aspiny']

n_classes = len(CLASSES)

base_model = ResNet50(weights='imagenet', include_top=False)

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
# x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
# x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
preds = Dense(n_classes, activation='softmax')(x)


model = Model(inputs=base_model.input, outputs=preds)


train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.2)

train_generator = train_datagen.flow_from_directory('data/images/gaf_gray/png/mouse',
                                                    target_size=(224, 224),
                                                    color_mode='rgb',
                                                    batch_size=BATCH_SIZE,
                                                    # class_mode='categorical',
                                                    classes=CLASSES,
                                                    shuffle=True,
                                                    subset='training')

validation_generator = train_datagen.flow_from_directory('data/images/gaf_gray/png/mouse',
                                                         target_size=(224, 224),
                                                         color_mode='rgb',
                                                         batch_size=BATCH_SIZE,
                                                         # class_mode='categorical',
                                                         classes=CLASSES,
                                                         shuffle=True,
                                                         subset='validation')

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])


step_size_train = train_generator.n//train_generator.batch_size

model.fit_generator(generator=train_generator,
                    steps_per_epoch=step_size_train,
                    epochs=100, validation_data = validation_generator,
                    validation_steps = validation_generator.samples // train_generator.batch_size)
