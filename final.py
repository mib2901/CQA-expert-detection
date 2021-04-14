import numpy as np
from nltk.corpus import stopwords
import pandas as pd
from gensim.models import Word2Vec as w2v
from nltk.stem import PorterStemmer
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import BatchNormalization
from keras.layers import Dense

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

X = []
punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
punc += '1234567890'

model = w2v.load("skipgram_v2.model")
wv = model.wv

vocab = wv.key_to_index

vocab1 = pd.read_csv('vocabulary.csv')

vocab_to_int = {}

for i in range(len(vocab1)):
    vocab_to_int[vocab1['Word'][i]] = int(vocab1['Id'][i])

model = Sequential()
model.add(LSTM(256, input_shape = (244, 1)))
model.add(BatchNormalization())
model.add(Dense(1, activation = "sigmoid"))
model.compile(loss = "mse", optimizer = "sgd", metrics = ["accuracy"])

model.load_weights("./formality_model_v1.h5")

def transform_text(A):
    str = []
    
    A = A.lower()
    temp = ''

    for j in range(len(A)):
        if A[j] not in punc:
            temp += A[j]

    temp = temp.split(' ')

    for i in range(len(temp)):
        if temp[i] != '' and temp[i] not in stop_words:
            p = ps.stem(temp[i])
            if p in vocab:
                str.append(p)
    return str

def similarity(Q, A):
    result = 0
    for i in A:
        x1 = wv[i]
        r = 0
        for j in Q:
            x2 = wv[j]
            if r < np.dot(x1, x2)/(np.linalg.norm(x1)* np.linalg.norm(x2)):
                r = np.dot(x1, x2)/(np.linalg.norm(x1)* np.linalg.norm(x2))
        result += r
    if len(A) > 0:
        result = result / len(A)
    result = round(result, 4)
    return result

def get_result(Q, A):
    q = transform_text(Q)

    result = []
    alt_result = []
    
    for i in range(len(A)):
        a = transform_text(A[i])
        
        X_train = []

        temp = [0] * len(vocab_to_int)

        for j in a:
            if j in vocab_to_int:
                temp[vocab_to_int[j]] += 1
        X_train.append(temp)
        
        X_train = np.reshape(X_train, (1,len(vocab_to_int),1))
        
        x = model.predict(X_train)
        
        alt_result.append(x[0])
        
        result.append(similarity(q, a) + (x[0] / 2))
        
    return result

Q = 'Who were the members of the rock band the Beatles?'

A = ['The Beatles were an English rock band formed in Liverpool in 1960. The group, whose best-known line-up comprised John Lennon, Paul McCartney, George Harrison and Ringo Starr, are regarded as the most influential band of all time.', 'Beetles are a group of insects that form the order Coleoptera, in the superorder Endopterygota. Their front pair of wings are hardened into wing-cases, elytra, distinguishing them from most other insects.', 'Beetle Spur (84°10′S 172°0′ECoordinates: 84°10′S 172°0′E) is a rock spur 2 nautical miles (4 km) north of Mount Patrick in the Commonwealth Range. It descends from a small summit peak on the range to the east side of Beardmore Glacier.', 'The Volkswagen Beetle—officially the Volkswagen Type 1, informally in German the Käfer (meaning "beetle"),[11] and known by many other nicknames in other languages—is a two-door, rear-engine economy car, intended for five occupants.', 'Beetle is a British party game in which one draws a beetle in parts. The game may be played solely with pen, paper and a die or using a commercial game set, some of which contain custom scorepads and dice and others which contain pieces which snap together to make a beetle/bug.']

result = get_result(Q, A)

for i in result:
    print(i[0], end = ' ')

print()