from CNN import CNNmodel
import pandas as pd
import numpy as np
import keras.optimizers as opt
#incorporated PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('fer2013/fer2013.csv')

# learning_rate = float(input("Enter Learning Rate: "))
# epochs = int(input("Enter number of epochs: "))
# # batch_size = int(input("Enter batch size: "))

# for i in range(5):
#     print()

data = np.array(data)

output = data[:, 0]
y = np.zeros((len(output), 7))
for i in range(len(output)):
    y[i][output[i]] = 1

X = data[:, 1]
X = [np.fromstring(x, dtype='int', sep=' ') for x in X]

X = np.array([np.fromstring(x, dtype='int', sep=' ').reshape(2304)
              for x in data[:, 1]])

# print(X.shape)
# X[:0.90, 0.95]
print(X.shape)
print(y.shape)
X_train, X_test, X_validation = X[0:int(0.80 * len(X)),:], X[int(0.80 * len(X)):int(
    0.90 * len(X)),:], X[int(0.90 * len(X)):int(len(X)),:]
y_train, y_test, y_validation = y[0:int(0.80 * len(X)),:], y[int(0.80 * len(X)):int(
    0.90 * len(X)),:], y[int(0.90 * len(X)):int(len(X)),:]

n = 576
pca = PCA(n_components=n)
scaler = StandardScaler()
# print(X_train.shape)
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_validation = scaler.transform(X_validation)
# print(X_train)
# print(X_test)
# print(X_validation)
s1 = X_train.shape[0]
s2 = X_test.shape[0]
s3 = X_validation.shape[0]
X_train = pca.fit_transform(X_train).reshape(s1, 24, 24, 1)
X_test = pca.fit_transform(X_test).reshape(s2, 24, 24, 1)
X_validation = pca.fit_transform(X_validation).reshape(s3, 24, 24, 1)

model = CNNmodel(num_emotions=7)
optimizer = opt.Adadelta(lr=0.01, decay=1e-6)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'],
              )

model.fit(X_train, y_train, validation_data=(
    X_validation, y_validation), epochs=50, verbose=2, batch_size=128)

scores = model.evaluate(X_test, y_test, verbose=0)
print(scores[1] * 100)
model_json = model.to_json()
with open("model.json", "w") as f:
    f.write(model_json)
model.save_weights("model.h5")
