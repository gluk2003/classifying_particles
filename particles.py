import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import tensorflow as tf
import tf2onnx
import json

tf.random.set_seed(43)

data = pd.read_csv("training.csv")
features = np.asarray(list(set(data.columns) - {'Label',}))

x = np.asarray(data[features], dtype=np.float32)
le = preprocessing.LabelEncoder()
le.fit(data.Label)
y = le.transform(data.Label).astype(np.float32)
labels = np.array(le.classes_, dtype=np.str)

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.5, random_state=42)

INPUT_DIM  = x_train.shape[1]
HIDDEN_DIM = 200
OUTPUT_DIM = len(labels)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(HIDDEN_DIM, activation="relu"),
    tf.keras.layers.Dense(OUTPUT_DIM, activation="softmax", name="output"),
])

learning_rate = 1e-3
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, tf.keras.utils.to_categorical(y_train, num_classes=len(labels)),
    epochs=50, batch_size=32,
    validation_data=(x_val, tf.keras.utils.to_categorical(y_val, num_classes=len(labels))))

y_pred = model.predict(x_test)
roc_auc = roc_auc_score(y_true=tf.keras.utils.to_categorical(y_test, num_classes=len(labels)), y_score=y_pred, multi_class='ovo')
print("ROC AUC score:", roc_auc)

fpr = dict()
tpr = dict()
roc_auc_dict = dict()
for i, label in enumerate(labels):
    fpr[label], tpr[label], _ = roc_curve(y_true=tf.keras.utils.to_categorical(y_test, num_classes=len(labels))[:, i], y_score=y_pred[:, i])
    roc_auc_dict[label] = roc_auc_score(y_true=tf.keras.utils.to_categorical(y_test, num_classes=len(labels))[:, i], y_score=y_pred[:, i])
    plt.plot(fpr[label], tpr[label], label=label)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig("roc.png")
plt.show()

y_pred_classes = np.argmax(y_pred, axis=1)
conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred_classes)

plt.matshow(conf_matrix, cmap=plt.cm.Blues)
plt.colorbar()
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, rotation=45)
plt.yticks(tick_marks, labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion.png')
plt.show()


plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.savefig('train.png')
plt.show()


metrics = {"test_auc": {}}
for i, label in enumerate(labels):
    label_indices = np.where(y_test == i)[0]
    label_y_test = tf.keras.utils.to_categorical(y_test[label_indices], num_classes=len(labels))
    label_y_pred = y_pred[label_indices]
    label_roc_auc = roc_auc_dict[label]
    metrics["test_auc"][label] = round(label_roc_auc, 2)

with open("particles.json", "w") as f:
    json.dump(metrics, f, indent=2)
    

spec = (tf.TensorSpec((None, INPUT_DIM), tf.float32, name="input"),)
output_path = "particles.onnx"

"""

не получилось сохранить модель (какая-то проблема с совместимостью библиотек tf и tf2onnx)
выдает ошибку 'FuncGraph' object has no attribute '_captures'


"""

model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_path)


 
