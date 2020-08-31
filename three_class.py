#匯入資料庫
import jieba
import pandas as pd # 引用套件並縮寫為 pd
import os
import numpy as np
import tensorflow as tf
import time
from datetime import timedelta
import csv
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from sklearn import metrics

#Read Csv
train_paths = ['./1_政治.csv', './1_科技.csv', './1_運動.csv']
train_x_raw, train_y_raw = [], []
for train_path in train_paths:
    cnt = 0
    with open(train_path, newline='',encoding='utf-8') as csvfile:
        # 讀取 CSV 檔案內容
        rows = csv.reader(csvfile)
        # 以迴圈輸出每一列
        for row in rows:
            cnt+=1
            if cnt==5001:
                break
            else:
                train_x_raw.append(row[2])
                train_y_raw.append(row[3])

test_paths = ['./測試資料.csv']
test_x_raw, test_y_raw = [], []
for test_path in test_paths:
    index = test_paths.index(test_path)
    with open(test_path, newline='', encoding='utf-8') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            test_x_raw.append(row[2])
            test_y_raw.append(row[3])
            
test_x_raw, test_y_raw = test_x_raw[1:], test_y_raw[1:]

#Label to int & Save clean data
label_dict = {
    "政治": '0',
    "科技": '1',
    "運動": '2'
}
def to_int(label):
    return label_dict[label]

train_x_clean_path, train_y_clean_path = "./data/train_x_clean", "./data/train_y_clean"
test_x_clean_path, test_y_clean_path = "./data/test_x_clean", "./data/test_y_clean"
if(not os.path.isfile("{0}.npy".format(train_x_clean_path))):
    print("Data clean file is not exist \n >> Prepare Data clean file")    
    train_x_clean, train_y_clean = train_x_raw, [to_int(i) for i in train_y_raw]
    test_x_clean, test_y_clean = test_x_raw, [to_int(i) for i in test_y_raw]
    
    np.save(train_x_clean_path, train_x_clean) ; np.save(train_y_clean_path, train_y_clean) # train
    np.save(test_x_clean_path, test_x_clean) ; np.save(test_y_clean_path, test_y_clean) # test
else:
    print("train/test data clean file is exist")    
    print(">> Loding  \n   train_x_clean from {0}.npy \n   train_y_clean from {1}.npy".format(train_x_clean_path, train_y_clean_path))    
    print(">> Loding  \n   test_x_clean from {0}.npy \n   test_y_clean from {1}.npy".format(test_x_clean_path, test_y_clean_path))    
    train_x_clean, train_y_clean = np.load("{0}.npy".format(train_x_clean_path)), np.load("{0}.npy".format(train_y_clean_path))
    test_x_clean, test_x_clean = np.load("{0}.npy".format(test_x_clean_path)), np.load("{0}.npy".format(test_y_clean_path))

#Seg Data
train_x_seg_path, train_y_seg_path = "./data/train_x_seg", "./data/train_y_seg"
test_x_seg_path, test_y_seg_path = "./data/test_x_seg", "./data/test_y_seg"
if(not os.path.isfile("{0}.npy".format(train_x_seg_path))):
    print("Seg Train/Test data file is not exist")   
    print(">> Prepare process Seg Train/Test data file") 
    train_x_seg, test_x_seg = [' '.join(list(i)) for i in train_x_clean], [' '.join(list(i)) for i in test_x_clean]
    train_y_seg, test_y_seg = train_y_clean, test_y_clean
    np.save(train_x_seg_path, train_x_seg); np.save(train_y_seg_path, train_y_seg)
    np.save(test_x_seg_path, test_x_seg) ; np.save(test_y_seg_path, test_y_seg)
else:
    print("Seg Train/Test data file is exist")   
    print(">> Loding  \n   train_x_seg from {0}.npy \n   train_y_seg from {1}.npy".format(train_x_seg_path, train_y_seg_path))    
    print(">> Loding  \n   test_x_seg from {0}.npy \n   test_y_seg from {1}.npy".format(test_x_seg_path, test_y_seg_path))    
    train_x_seg, train_y_seg = np.load("{0}.npy".format(train_x_seg_path), allow_pickle=True), np.load("{0}.npy".format(train_y_seg_path), allow_pickle=True)
    test_x_seg, test_y_seg = np.load("{0}.npy".format(test_x_seg_path), allow_pickle=True), np.load("{0}.npy".format(test_y_seg_path), allow_pickle=True)

#參數設定
class Config():
    max_sequence_length = 600 # 最長序列長度為n個字
    min_word_frequency = 1 # 出現頻率小於n的話 ; 就當成罕見字
    
    vocab_size = None
    category_num = None
    
    embedding_dim_size = 64 # 詞向量維度
    num_filters = 128  # kernal數 #default 256
    kernel_size = 5  # kernal尺寸
    hidden_dim = 32  # FC神經元數 #default 64
    
    learning_rate = 0.001 # 學習率
    keep_prob = 0.5 
    
    batch_size = 64 # mini-batch
    epoch_size = 30 # epoch
    
    save_path = './model/cnn_best_validation' # 模型儲存檔名
    
config = Config()


train_X_path, train_y_path = "./data/train_X", "./data/train_y"
val_X_path, val_y_path = "./data/val_X", "./data/val_y"
test_X_path, test_y_path = "./data/test_X", "./data/test_y"
if(not os.path.isfile("{0}.npy".format(train_X_path))):
    print("Train/Val/Test data file is not exist")   
    print(">> Prepare process Train/Val/Test data file") 
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(config.max_sequence_length, min_frequency=config.min_word_frequency)
    train_x_pad = np.array(list(vocab_processor.fit_transform(train_x_seg)))
    train_y_pad = tf.keras.utils.to_categorical(train_y_seg)
    
    test_X = np.array(list(vocab_processor.fit_transform(test_x_seg)))
    test_y = tf.keras.utils.to_categorical(test_y_seg)
    
    config.vocab_size = len(vocab_processor.vocabulary_)
    
    with open('./data/vocab.txt', 'wt', encoding="utf-8") as w_file:
        for vocab in vocab_processor.vocabulary_._reverse_mapping:
            w_file.write(vocab + "\n")      
    print("Total vocab size: {0}".format(config.vocab_size))
    
    train_X, val_X, train_y, val_y = train_test_split(train_x_pad, train_y_pad, test_size = 0.3, random_state = 1)
    print('>> Full Train Input Data Shape : {0} ; Full Train Input Label Shape : {1}'.format(train_X.shape, train_y.shape))
    print('>> Full Val Input Data Shape : {0} ; Full Val Input Label Shape : {1}'.format(val_X.shape, val_y.shape))
    print('>> Full Test Input Data Shape : {0} ; Full Test Input Label Shape : {1}'.format(test_X.shape, test_y.shape))
    np.save(train_X_path, train_X); np.save(train_y_path, train_y)
    np.save(val_X_path, val_X) ; np.save(val_y_path, val_y)
    np.save(test_X_path, test_X) ; np.save(test_y_path, test_y)
else:
    print("Train/Val/Test data file is exist")   
    train_X, train_y = np.load("{0}.npy".format(train_X_path)),  np.load("{0}.npy".format(train_y_path))
    val_X, val_y = np.load("{0}.npy".format(val_X_path)),  np.load("{0}.npy".format(val_y_path))
    test_X, test_y = np.load("{0}.npy".format(test_X_path)),  np.load("{0}.npy".format(test_y_path))
    config.vocab_size = sum(1 for line in open("./data/vocab.txt",encoding='utf-8'))

config.category_num = train_y.shape[1]
print('>> Train Data Shape : {0} ; Train Label Shape : {1}'.format(train_X.shape, train_y.shape))
print('>> Val Data Shape : {0} ; Val Label Shape : {1}'.format(val_X.shape, val_y.shape))
print('>> Test Data Shape : {0} ; Test Label Shape : {1}'.format(test_X.shape, test_y.shape))

#模型架構
class TextCNN(object):
    def __init__(self, config):
        self.config = config
        
        # 四個等待輸入的data
        self.batch_size = tf.placeholder(tf.int32, [] , name = 'batch_size')
        self.keep_prob = tf.placeholder(tf.float32, [], name = 'keep_prob')
        
        # Initial
        self.x = tf.placeholder(tf.int32, [None, self.config.max_sequence_length] , name = 'x')
        self.y_label = tf.placeholder(tf.float32, [None, self.config.category_num], name = 'y_label')
        
        self.cnn()
    
    def cnn(self):
        """RNN模型"""
        # 詞向量映射
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim_size])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.x)
            
        #原始single layer
        with tf.name_scope("cnn"):
            # single layer
            # CNN layer
            conv = tf.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kernel_size, name='conv')
            # global max pooling layer
            gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')
        
        with tf.name_scope("score"):
            # 全連接層，後面接dropout以及relu激活
            fc = tf.layers.dense(gmp, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            # 分類器
            self.logits = tf.layers.dense(fc, self.config.category_num, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 損失函數，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_label)
            self.loss = tf.reduce_mean(cross_entropy)
            # 優化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 準確率
            correct_pred = tf.equal(tf.argmax(self.y_label, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#CNN訓練
def get_time_dif(start_time):
    """得到已使用時間"""
    end_time = time.time()
    time_dif = end_time - start_time
    
    return timedelta(seconds = int(round(time_dif)))

def feedData(x_batch, y_batch, keep_prob, batch_size, model):
    feed_dict = {
        model.x: x_batch,
        model.y_label: y_batch,
        model.keep_prob: keep_prob,
        model.batch_size: batch_size
    }
    return feed_dict

best_val_acc = -1.0 # 最佳驗證集準確度
last_improved = 0 # 紀錄上一次提升batch 
require_improvement = 30  # 如果超过n輪未提升，提前结束訓練
total_batch = 0  # 總批次
print_per_batch = 10

tf.reset_default_graph()

model = TextCNN(config)
start_time = time.time()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    
    flag = False
    for epoch in range(config.epoch_size):
        print('Epoch: {0}'.format(epoch + 1))
        for step in range(0, train_X.shape[0], config.batch_size):
            batch_x, batch_y = train_X[step:step + config.batch_size], train_y[step:step + config.batch_size]
            
            if total_batch % print_per_batch == 0:  
                train_loss, train_acc = sess.run([model.loss, model.acc], feed_dict = feedData(batch_x, batch_y, 1.0, batch_x.shape[0], model))
                val_loss, val_acc = sess.run([model.loss, model.acc], feed_dict = feedData(val_X, val_y, 1.0, val_X.shape[0], model))
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    last_improved = total_batch
                    saver.save(sess = sess, save_path = config.save_path)
                    improved_str = '*'
                else:
                    improved_str = ''
                    
                time_dif = get_time_dif(start_time)                               
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.3}, Train Acc: {2:>7.2%}, Val Loss: {3:>6.3}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(total_batch, train_loss, train_acc, val_loss, val_acc, time_dif, improved_str))
            
            # train
            sess.run(model.optim, feed_dict = feedData(batch_x, batch_y, 1.0, batch_x.shape[0], model))
            total_batch += 1
            
            if total_batch - last_improved > require_improvement:
                # 驗證集準確度長期不提升，提前结束訓練
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break  # 跳出循环

        if flag:  # 同上
            break
    print("訓練完成...")

#測試集
tf.reset_default_graph()
model = TextRNN(config)
start_time = time.time()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess = sess, save_path = config.save_path)  # 讀取保存的模型
    
    test_loss, test_acc, test_predict_label = sess.run([model.loss, model.acc, model.y_pred_cls], feed_dict = feedData(test_X, test_y, 1.0, test_X.shape[0], model))
    time_dif = get_time_dif(start_time)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}, Time: {2}'
    print(msg.format(test_loss, test_acc, time_dif))
    print("測試完成...") 
    
    # 評估
    test_label = np.argmax(test_y, 1)
    print(">> Precision, Recall and F1-Score...")
    print(metrics.classification_report(test_label, test_predict_label))
    
    # 混淆矩陣
    print(">> Confusion Matrix...")
    cm = metrics.confusion_matrix(test_label, test_predict_label)
    print(cm)