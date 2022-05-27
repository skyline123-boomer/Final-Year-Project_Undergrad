import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics, Model, Input
from tensorflow.keras.layers import Dense, Dropout, add
import numpy as np
import scipy.io as sio
import random
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import  accuracy_score

def preprocess(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, y):

    x1 = tf.cast(x1, dtype=tf.float32)
    x2 = tf.cast(x2, dtype=tf.float32)
    x3 = tf.cast(x3, dtype=tf.float32)
    x4 = tf.cast(x4, dtype=tf.float32)
    x5 = tf.cast(x5, dtype=tf.float32)
    x6 = tf.cast(x6, dtype=tf.float32)
    x7 = tf.cast(x7, dtype=tf.float32)
    x8 = tf.cast(x8, dtype=tf.float32)
    x9 = tf.cast(x9, dtype=tf.float32)
    x10 = tf.cast(x10, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.int32)
    return x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, y

data = sio.loadmat('pattern_real.mat')['data_real']
data = tf.convert_to_tensor(data, dtype=tf.float32)
print(data.shape)
batchsz = 50

idx1 = tf.range(5600)
idx1 = tf.random.shuffle(idx1)

idx2 = tf.range(2400)
idx2 = tf.random.shuffle(idx2)

x = data[:,:1820]
y = data[:,1820:1884]

mean = tf.reduce_mean(x)
std = np.std(x, ddof=1)
std = tf.convert_to_tensor(std, dtype=tf.float32)
x = (x - mean) / std

x0 = x[:700,:]
x1 = x[1000:1700,:]
x2 = x[2000:2700,:]
x3 = x[3000:3700,:]
x4 = x[4000:4700,:]
x5 = x[5000:5700,:]
x6 = x[6000:6700,:]
x7 = x[7000:7700,:]
x_train = tf.concat([x0,x1,x2,x3,x4,x5,x6,x7],axis=0)
x_train = tf.gather(x_train, idx1)

x0 = x[700:1000,:]
x1 = x[1700:2000,:]
x2 = x[2700:3000,:]
x3 = x[3700:4000,:]
x4 = x[4700:5000,:]
x5 = x[5700:6000,:]
x6 = x[6700:7000,:]
x7 = x[7700:8000,:]
x_val = tf.concat([x0,x1,x2,x3,x4,x5,x6,x7],axis=0)
x_val = tf.gather(x_val, idx2)

y0 = y[:700,:]
y1 = y[1000:1700,:]
y2 = y[2000:2700,:]
y3 = y[3000:3700,:]
y4 = y[4000:4700,:]
y5 = y[5000:5700,:]
y6 = y[6000:6700,:]
y7 = y[7000:7700,:]
y_train = tf.concat([y0,y1,y2,y3,y4,y5,y6,y7],axis=0)
y_train = tf.gather(y_train, idx1)

y0 = y[700:1000,:]
y1 = y[1700:2000,:]
y2 = y[2700:3000,:]
y3 = y[3700:4000,:]
y4 = y[4700:5000,:]
y5 = y[5700:6000,:]
y6 = y[6700:7000,:]
y7 = y[7700:8000,:]
y_val = tf.concat([y0,y1,y2,y3,y4,y5,y6,y7],axis=0)
y_val = tf.gather(y_val, idx2)

x_train1 = x_train[:,:182]
x_train2 = x_train[:,182:364]
x_train3 = x_train[:,364:546]
x_train4 = x_train[:,546:728]
x_train5 = x_train[:,728:910]
x_train6 = x_train[:,910:1092]
x_train7 = x_train[:,1092:1274]
x_train8 = x_train[:,1274:1456]
x_train9 = x_train[:,1456:1638]
x_train10 = x_train[:,1638:1820]

x_val1 = x_val[:,:182]
x_val2 = x_val[:,182:364]
x_val3 = x_val[:,364:546]
x_val4 = x_val[:,546:728]
x_val5 = x_val[:,728:910]
x_val6 = x_val[:,910:1092]
x_val7 = x_val[:,1092:1274]
x_val8 = x_val[:,1274:1456]
x_val9 = x_val[:,1456:1638]
x_val10 = x_val[:,1638:1820]

train_db = tf.data.Dataset.from_tensor_slices((x_train1,x_train2,x_train3,x_train4,x_train5,x_train6,x_train7,x_train8,x_train9,x_train10,y_train))
train_db = train_db.map(preprocess).shuffle(5000).batch(batchsz)
val_db = tf.data.Dataset.from_tensor_slices((x_val1,x_val2,x_val3,x_val4,x_val5,x_val6,x_val7,x_val8,x_val9,x_val10,y_val))
val_db = val_db.map(preprocess).shuffle(1000).batch(batchsz)

sample = next(iter(train_db))
print('batch:', sample[0].shape, sample[10].shape)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'logs/' + current_time
summary_writer = tf.summary.create_file_writer(log_dir)

base_inp = Input(shape=(182,))
ss = Dense(128, activation='relu')(base_inp)
ss = Dense(80, activation='relu')(ss)
ss = Dense(64, activation='relu')(ss)
ss = Dropout(0.2)(ss)
base_model = Model(inputs=base_inp, outputs=ss)

inp1 = Input(shape=(182,))
inp2 = Input(shape=(182,))
inp3 = Input(shape=(182,))
inp4 = Input(shape=(182,))
inp5 = Input(shape=(182,))
inp6 = Input(shape=(182,))
inp7 = Input(shape=(182,))
inp8 = Input(shape=(182,))
inp9 = Input(shape=(182,))
inp10 = Input(shape=(182,))

out1 = base_model(inp1)
out2 = base_model(inp2)
out3 = base_model(inp3)
out4 = base_model(inp4)
out5 = base_model(inp5)
out6 = base_model(inp6)
out7 = base_model(inp7)
out8 = base_model(inp8)
out9 = base_model(inp9)
out10 = base_model(inp10)

merged_vector = keras.layers.concatenate([out1, out2, out3, out4, out5, out6, out7, out8, out9, out10], axis=-1)
ss = Dense(300, activation='relu')(merged_vector)
ss = Dropout(0.2)(ss)
predictions = Dense(64, activation='sigmoid')(ss)
new_model = Model(inputs=[inp1, inp2, inp3, inp4, inp5, inp6, inp7, inp8, inp9, inp10], outputs=predictions)
new_model.summary()

optimizers = optimizers.Adam(lr=1e-3)
def main():
    test_loss_list=[]
    train_loss_list=[]
    for epoch in range(101):
        test_loss_ce_epoch = 0
        train_loss_ce_epoch = 0
        for step, (x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,y) in enumerate(train_db):


            with tf.GradientTape() as tape:

                logits = new_model([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10])

                loss_ce = tf.losses.binary_crossentropy(y, logits, from_logits = False)
                loss_ce = tf.reduce_mean(loss_ce)
                train_loss_ce_epoch += loss_ce

            grads = tape.gradient(loss_ce, new_model.trainable_variables)
            optimizers.apply_gradients(zip(grads, new_model.trainable_variables))

            if step % 50 == 0:
                print(epoch, step, 'loss', float(loss_ce))
                with summary_writer.as_default():
                    tf.summary.scalar('train-loss', float(loss_ce), step=epoch)
        train_loss_ce_epoch = train_loss_ce_epoch / 3
        train_loss_list.append(train_loss_ce_epoch)


        total_correct = 0
        total_number = 0
        for (x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,y) in val_db:
            logits2 = new_model([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10])

            loss_ce2 = tf.losses.binary_crossentropy(y, logits2, from_logits=False)
            loss_ce2 = tf.reduce_mean(loss_ce2)
            test_loss_ce_epoch += loss_ce2

            mask = logits2 > 0.5
            A = tf.ones([50,64])
            B = tf.zeros([50,64])
            pred = tf.where(mask, A, B)

            for i in range(50):
                prob = accuracy_score(y[i], pred[i])
                if prob == 1:
                    total_correct += 1

            total_number += x1.shape[0]

        acc = total_correct / total_number
        print(epoch, 'validation acc', acc)

        test_loss_list.append(test_loss_ce_epoch)

        with summary_writer.as_default():
            tf.summary.scalar('validation-acc', float(acc), step=epoch)

    plt.plot(train_loss_list)
    plt.plot(test_loss_list)
    plt.title('Transferred Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()