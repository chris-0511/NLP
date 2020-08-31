#匯入函式庫
from collections import Counter
import jieba
jieba.set_dictionary('dict.txt')  # 如果是使用繁體文字，請記得去下載繁體字典來使用
import numpy as np
import pandas as pd
from numpy.linalg import norm

#匯入資料
# 把檔案讀出來(原始資料: https://society.hccg.gov.tw/ch/home.jsp?id=43&parentpath=0,5)
df_QA = pd.read_json('ProcessedData.json', encoding='utf8')
# 我們這次只會使用到question跟ans這兩個欄位
df_question = df_QA[['question', 'ans']].copy()  ## 不要更動到原始的DataFrame
df_question.drop_duplicates(inplace=True)  ## 丟掉重複的資料
df_question.head(5)  ## show出來


#前處理
all_terms = []
def preprocess(item):  ##定義前處理的function
    # 請把將每一行用jieba.cut進行分詞(記得將cut_all設定為True)
    # 同時建立所有詞彙的list(all_terms)
    #=============your works starts===============#
    terms = [t for t in jieba.cut(item, cut_all=True)]  ## 把全切分模式打開，可以比對的詞彙比較多
    all_terms.extend(terms)  ## 收集所有出現過的字
    #==============your works ends================#
    return terms

df_question['processed'] = df_question['question'].apply(preprocess)
print(df_question.iloc[0])
# question                                      小孩出生後應於何時申請育兒津貼?
# ans          1.幼兒家長在戶政事務所完成新生兒出生登記後，即可向所轄區公所社政課提出育兒津貼申請。2.在...
# processed                  [小孩, 出生, 後, 應於, 何時, 申請, 育兒, 津貼, , ]
# Name: 0, dtype: object

df_question.head()

# 建立詞彙表
termindex = list(set(all_terms))

print("len(termindex)", len(termindex))
print(termindex[:10])


# 建立IDF vector
Doc_Length = len(df_question)  ## 計算出共有幾篇文章
Idf_vector = []  ## 初始化IDF向量
for term in termindex:  ## 對index中的詞彙跑回圈
    num_of_doc_contains_term = 0  ## 計算有機篇文章出現過這個詞彙
    for terms in df_question['processed']:
        if term in terms:
            num_of_doc_contains_term += 1
    idf = np.log(Doc_Length/num_of_doc_contains_term)  ## 計算該詞彙的IDF值
    Idf_vector.append(idf)
print(len(Idf_vector))
print(termindex[:10])
print(Idf_vector[:10])

# 建立document vector
def terms_to_vector(terms):  ## 定義把terms轉換成向量的function
    ## 建立一條與termsindex等長、但值全部為零的向量(hint:dtype=np.float32)
    #=============your works starts===============#
    vector = np.zeros_like(termindex, dtype=np.float32)  
    #==============your works ends================#
    
    for term, count in Counter(terms).items():
        # 計算vector上每一個字的tf值
        #=============your works starts===============#
        vector[termindex.index(term)] = count
        #==============your works ends================#

    # 計算tfidf，element-wise的將vector與Idf_vector相乘
    ## hint: 如果兩個vector的型別都是np.array，把兩條vector相乘，就會自動把向量中的每一個元素成在一起，建立出一條新的向量
    #=============your works starts===============#
    vector = vector * Idf_vector
    #==============your works ends================#
    return vector

df_question['vector'] = df_question['processed'].apply(terms_to_vector)  ## 將上面定義的function，套用在每一筆資料的terms欄位上

#餘弦相似性
def cosine_similarity(vector1, vector2):  ## 定義cosine相似度的計算公式
    # 使用np.dot與norm計算cosine score
    #=============your works starts===============#
    score = np.dot(vector1, vector2)  / (norm(vector1) * norm(vector2))
    #==============your works ends================#
    return score#越大越像

sentence1 = df_question.loc[0]  ##取出第零個的問題
sentence2 = df_question.loc[2]  ##取出第二個的問題
sentence10 = df_question.loc[10]
print(sentence1['question'])
print(sentence2['question'])
print(sentence10['question'])

print(cosine_similarity(sentence1['vector'], sentence2['vector']))  ##計算兩者的相似度
print(cosine_similarity(sentence1['vector'], sentence10['vector']))  ##計算兩者的相似度

#資料檢索
def retrieve(testing_sentence, return_num=3):  ## 定義出檢索引擎
    testing_vector = terms_to_vector(preprocess(testing_sentence))  ## 把剛剛的前處理、轉換成向量的function，應用在使用者輸入的問題上
    idx_score_mapping = [(idx, cosine_similarity(testing_vector, vec)) for idx, vec in enumerate(df_question['vector'])]
    top3_idxs = np.array(sorted(idx_score_mapping, key=lambda x:x[1], reverse=True))[:3, 0]

    return df_question.loc[top3_idxs, ['question', 'ans']]

idxs = retrieve("老人年金").index
print(idxs)
# Float64Index([100.0, 111.0, 321.0], dtype='float64')
print(df_question.loc[idxs, 'question'])


#Use Scikit learn
from sklearn.feature_extraction.text import TfidfVectorizer

#TF-IDF
tfidf = TfidfVectorizer()
df_question['sklearn_vector'] = list(tfidf.fit_transform(df_question['processed'].apply(lambda x:" ".join(x)).values).toarray())
print(df_question.loc[:10, 'sklearn_vector'].apply(sum).values)

#資料檢索
def sklearn_retrieve(testing_sentence, return_num=3):  ## 定義出檢索引擎
    # 請使用前面定義的tfidf.transform與preprocess兩個function，計算出testing_sentence的向量
    # 注意tfidf.transform必須是兩個維度的array
    # 且out為sparse metric，必需.toarray()轉換為一般np.array()
    # 計算其與資料庫每一的問句的相似度
    # 依分數進行排序，找到分數最高的三個句子
    #=============your works starts===============#
    testing_vector = tfidf.transform([" ".join(preprocess(testing_sentence))]).toarray()[0]
    idx_score_mapping = [(idx, cosine_similarity(testing_vector, vec)) for idx, vec in enumerate(df_question['sklearn_vector'])]
    top3_idxs = np.array(sorted(idx_score_mapping, key=lambda x:x[1], reverse=True))[:return_num, 0]
    #==============your works ends================#
    return df_question.loc[top3_idxs, ['question', 'ans']]

print(retrieve("老人年金")['question'])
print(sklearn_retrieve("老人年金")['question'])