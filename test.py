model = Sequential([
  layers.Rescaling(1./255, input_shape=(180, 180, 1)),
  layers.Conv2D(16, 3, strides=2, padding='same', activation='relu'),
  layers.Conv2D(32, 3, strides=2, padding='same', activation='relu'),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(128, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu', activity_regularizer=regularizers.L2(0.01)),
  layers.BatchNormalization()
  layers.Dense(1, activation='sigmoid', activity_regularizer=regularizers.L2(0.01))
])