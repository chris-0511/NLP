#匯入函式庫
import jieba
import pandas as pd # 引用套件並縮寫為 pd
import os
import numpy as np
import re
import tensorflow as tf
import time
from datetime import timedelta
import csv
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

#參數設定
class Config():
    max_sequence_length = 500 # 最長序列長度為n個字
    min_word_frequency = 3 # 出現頻率小於n的話 ; 就當成罕見字
    
    vocab_size = None
    category_num = None
    
    choose_model = 'lstm' # 想要使用的模型 ex lstm; rnn; gru
    embedding_dim_size =300 # 詞向量維度
    num_layer = 1 # 層數
    num_units = [128] # 神經元
    learning_rate = 0.0001 # 學習率         
    keep_prob = 0.8 
    
    batch_size = 64 # mini-batch
    epoch_size = 30 # epoch
    
    save_path = 'best_validation' # 模型儲存檔名
    
config = Config()

#資料載入
stopwords=[]
with open(r'stop.txt', 'r', encoding='utf-8') as f:
    for line in f:
        if len(line)>0:
            stopwords.append(line.strip())
# 開啟 CSV 檔案
data_train_x1,data_train_y1,data_valid_x1,data_valid_y1,data_test_x1,data_test_y1=[],[],[],[],[],[]

with open(r'train.csv', newline='',encoding='utf-8') as csvfile:
    # 讀取 CSV 檔案內容
    rows = csv.reader(csvfile)
    # 以迴圈輸出每一列
    for row in rows:
        data_train_x1.append(row[3])
        data_train_x= data_train_x1[1:]
        data_train_y1.append(row[5])
        data_train_y=data_train_y1[1:]
with open(r'test.csv', newline='',encoding='utf-8') as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
      data_valid_x1.append(row[3])
      data_valid_x=data_valid_x1[1:]
      data_valid_y1.append(row[5])
      data_valid_y=data_valid_y1[1:]
with open(r'valid.csv', newline='',encoding='utf-8') as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
      data_test_x1.append(row[1])
      data_test_x=data_test_x1[1:]
      data_test_y1.append(row[2])
      data_test_y=data_test_y1[1:]
#前處理
def clean_text(text_string):
    text_string = re.sub(r'[^\u4e00-\u9fa5]+', '', text_string)
    return(text_string)
# Clean texts
data_train_x = [clean_text(x) for x in data_train_x]
data_valid_x = [clean_text(x) for x in data_valid_x]
data_test_x = [clean_text(x) for x in data_test_x]

def clean_text(text_string):
    text_string = re.sub(r'[^\u4e00-\u9fa5]+', '', text_string)
    return(text_string)
# Clean texts
data_train_x = [clean_text(x) for x in data_train_x]
data_valid_x = [clean_text(x) for x in data_valid_x]
data_test_x = [clean_text(x) for x in data_test_x]
print(len(data_train_x))
print(len(data_train_y))
print(len(data_valid_x))
print(len(data_valid_y))
print(len(data_test_x))
print(len(data_test_y))


clean_train_x = []
clean_train_y = []
clean_vaild_x = []
clean_valid_y = []
clean_test_x = []
clean_test_y = []
print(f'清洗前trian:{len(data_train_x)},清洗前trian_target:{len(data_train_y)}')
for i in range(len(data_train_x)):
    if data_train_x[i]!='':
        clean_train_x.append(data_train_x[i])
        clean_train_y.append(data_train_y[i])
print(f'清洗後trian:{len(clean_train_x)},清洗後trian_target:{len(clean_train_y)}')

print(f'清洗前test:{len(data_valid_x)},清洗前test:{len(data_valid_y)}')
for i in range(len(data_valid_x)):
    if data_valid_x[i]!='':
        clean_vaild_x.append(data_valid_x[i])
        clean_valid_y.append(data_valid_y[i])
print(f'清洗後trian:{len(clean_vaild_x)},清洗後trian_target:{len(clean_valid_y)}')

print(f'清洗前valid:{len(data_test_x)},清洗前valid:{len(data_test_y)}')
for i in range(len(data_test_x)):
    if data_test_x[i]!='':
        clean_test_x.append(data_test_x[i])
        clean_test_y.append(data_test_y[i])
print(f'清洗後trian:{len(clean_test_x)},清洗後trian_target:{len(clean_test_y)}')

if(not os.path.isfile("seg_train_x.npy")):
    print("Train/Val/Test data file is not exist")   
    seg_train_x = []
    seg_valid_x = []
    seg_test_x = []
    for i in range(len(clean_train_x)):
        seg_train_x.append(' '.join([j for j in jieba.cut_for_search(clean_train_x[i]) if j not in stopwords]))
    for i in range(len(clean_vaild_x)):
        seg_valid_x.append(' '.join([j for j in jieba.cut_for_search(clean_vaild_x[i]) if j not in stopwords]))
    for i in range(len(clean_test_x)):
        seg_test_x.append(' '.join([j for j in jieba.cut_for_search(clean_test_x[i]) if j not in stopwords]))
    seg_train_y = clean_train_y
    seg_valid_y = clean_valid_y
    seg_test_y = clean_test_y
    np.save("seg_train_y", seg_train_y)
    np.save("seg_valid_y", seg_valid_y)
    np.save("seg_test_y", seg_test_y)

    np.save("seg_train_x", seg_train_x)
    np.save("seg_valid_x", seg_valid_x)
    np.save("seg_test_x", seg_test_x)
else:
    print("Train/Val/Test data file is exist")   
    seg_train_x, seg_train_y = np.load("seg_train_x.npy"),  np.load("seg_train_y.npy")
    seg_valid_x, seg_valid_y = np.load("seg_valid_x.npy"),  np.load("seg_valid_y.npy")
    seg_test_x, seg_test_y = np.load("seg_test_x.npy"),  np.load("seg_test_y.npy")



if(not os.path.isfile("train_x.npy")):
    print("Train/Val/Test data file is not exist")   
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(config.max_sequence_length, min_frequency=config.min_word_frequency)
    train_x = np.array(list(vocab_processor.fit_transform(seg_train_x)))
    train_y = tf.keras.utils.to_categorical(seg_train_y)
    valid_x = np.array(list(vocab_processor.fit_transform(seg_valid_x)))
    valid_y = tf.keras.utils.to_categorical(seg_valid_y)
    test_x = np.array(list(vocab_processor.fit_transform(seg_test_x)))
    test_y = tf.keras.utils.to_categorical(seg_test_y)
    config.vocab_size = len(vocab_processor.vocabulary_)
    
    with open('vocab.txt', 'wt',encoding="utf-8") as w_file:
        for vocab in vocab_processor.vocabulary_._reverse_mapping:
            w_file.write(vocab + "\n")      
    print("Total vocab size: {0}".format(config.vocab_size))

    np.save("train_x", train_x); np.save("train_y", train_y)
    np.save("val_x", valid_x) ; np.save("val_y", valid_y)
    np.save("test_x", test_x) ; np.save("test_y", test_y)
else:
    print("Train/Val/Test data file is exist")   
    train_x, train_y = np.load("train_x.npy"),  np.load("train_y.npy")
    valid_x, valid_y = np.load("val_x.npy"),  np.load("val_y.npy")
    test_x, test_y = np.load("test_x.npy"),  np.load("test_y.npy")
    
    config.vocab_size = sum(1 for line in open("vocab.txt",encoding='utf-8'))

config.category_num = train_y.shape[1]
print('>> Train Data Shape : {0} ; Train Label Shape : {1}'.format(train_x.shape, train_y.shape))
print('>> Val Data Shape : {0} ; Val Label Shape : {1}'.format(valid_x.shape, valid_y.shape))
print('>> Test Data Shape : {0} ; Test Label Shape : {1}'.format(test_x.shape, test_y.shape))

#模型架構
class TextRNN(object):
    def __init__(self, config):
        self.config = config
        
        # 四個等待輸入的data
        self.batch_size = tf.placeholder(tf.int32, [] , name = 'batch_size')
        self.keep_prob = tf.placeholder(tf.float32, [], name = 'keep_prob')
        
        # Initial
        self.x = tf.placeholder(tf.int32, [None, self.config.max_sequence_length] , name = 'x')
        self.y_label = tf.placeholder(tf.float32, [None, self.config.category_num], name = 'y_label')
        self.choose_model = config.choose_model
        self.rnn()
    # Get LSTM Cell
    def cell(self, num_units):
        #BasicLSTMCell activity => default tanh
        if self.choose_model == "lstm":
            #可以設定peephole等屬性
            LSTM_cell = rnn.LSTMCell( initializer = tf.random_uniform_initializer(-0.1, 0.1,seed=2 )) 
        elif self.choose_model == "basic":
            #最基礎的，沒有peephole
            LSTM_cell = rnn.BasicLSTMCell(num_units = num_units, forget_bias = 1.0, state_is_tuple = True) 
        else:
            LSTM_cell = rnn.GRUCell(num_units)

        return rnn.DropoutWrapper(LSTM_cell, output_keep_prob = self.keep_prob)
    
    def rnn(self):
        """RNN模型"""
        # 詞向量映射
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim_size])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.x)
            
        # RNN Layers
        with tf.name_scope('layers'):
            with tf.name_scope('RNN'):
                LSTM_cells = rnn.MultiRNNCell([self.cell(int(self.config.num_units[_])) for _ in range(self.config.num_layer)])
                # x_shape = tf.reshape(self.x, [-1, self.config.truncate, self.config.vectorSize])
                
            with tf.name_scope('output'):
                init_state = LSTM_cells.zero_state(self.batch_size, dtype = tf.float32)
                outputs, final_state = tf.nn.dynamic_rnn(LSTM_cells, inputs = embedding_inputs, 
                                                        initial_state = init_state, time_major = False, dtype = tf.float32)
                
        # Output Layer
        with tf.name_scope('output_layer'):
            # 全連接層，後面接dropout以及relu激活
            fc1 = tf.layers.dense(outputs[:, -1, :], int(self.config.num_units[len(self.config.num_units)-1]))
            fc1 = tf.contrib.layers.dropout(fc1, self.keep_prob)
            fc1 = tf.nn.relu(fc1)
                
            # 分類器
            y = tf.layers.dense(fc1, self.config.category_num, name = 'y')
        
        self.y_pred_cls = tf.argmax(y, axis = 1) #預測類別
        with tf.name_scope('cross_entropy'):
            with tf.name_scope('total'):
                self.y=tf.nn.softmax(y)   
                self.softmax = tf.nn.softmax_cross_entropy_with_logits(labels = self.y_label, logits = y)
                self.cross_entropy = tf.reduce_mean(self.softmax)

        with tf.name_scope('train'):
            self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.cross_entropy)

        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                self.correction_prediction = tf.equal(self.y_pred_cls, tf.argmax(self.y_label, axis = 1))
            with tf.name_scope('accuracy'):
                self.accuracy = tf.reduce_mean(tf.cast(self.correction_prediction, tf.float32))

#建立LSTM網路訓練
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
require_improvement = 300  # 如果超过n輪未提升，提前结束訓練
total_batch = 0  # 總批次
print_per_batch = 100

tf.reset_default_graph()

model = TextRNN(config)
start_time = time.time()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    
    flag = False
    for epoch in range(config.epoch_size):
        print('Epoch: {0}'.format(epoch + 1))
        shuffled_ix = np.random.permutation(np.arange(len(train_x)))
        train_x = train_x[shuffled_ix]
        train_y = train_y[shuffled_ix]
        for step in range(0, train_x.shape[0], config.batch_size):
            batch_x, batch_y = train_x[step:step + config.batch_size], train_y[step:step + config.batch_size]
            
            if total_batch % print_per_batch == 0:  
                train_loss, train_acc = sess.run([model.cross_entropy, model.accuracy], feed_dict = feedData(batch_x, batch_y, 1.0, batch_x.shape[0], model))
                val_loss, val_acc = sess.run([model.cross_entropy, model.accuracy], feed_dict = feedData(valid_x, valid_y, 1.0, valid_x.shape[0], model))
                
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
            sess.run(model.train_step, feed_dict = feedData(batch_x, batch_y, 1.0, batch_x.shape[0], model))
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
from sklearn import metrics
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
tf.reset_default_graph()
model = TextRNN(config)
start_time = time.time()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess = sess, save_path = config.save_path)  # 讀取保存的模型
    shuffled_ix = np.random.permutation(np.arange(len(test_x)))
    test_x = test_x[shuffled_ix]
    test_y = test_y[shuffled_ix]
    test_loss, test_acc, test_predict_label,y,y_label = sess.run([model.cross_entropy, model.accuracy, model.y_pred_cls,model.y,model.y_label], feed_dict = feedData(test_x, test_y, 1.0, test_x.shape[0], model))
    time_dif = get_time_dif(start_time)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}, Time: {2}'
    print(msg.format(test_loss, test_acc, time_dif))
    print("測試完成...") 
    test_label = np.argmax(test_y, 1)
    # 混淆矩陣
    print(">> Confusion Matrix...")
    cm = metrics.confusion_matrix(test_label, test_predict_label)
    print(cm)
    print(y)
