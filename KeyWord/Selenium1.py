def set_values():
    print("Language:")
    print("(1) Español")
    print("(2) English")
    language = input()
    if int(language) == 1:
        print ("Cuál es su Keyword?")
        Keyword= input()
        print(" \n Tu palabra es:", Keyword, "\n")
        LANG = "es"
        CONT="CO"
        return Keyword, LANG, CONT
    if int(language) == 2:
        print("What is your keyword?")
        Keyword= input()
        print("Your word is:", Keyword)
        LANG = "en"
        CONT="US"
        return Keyword, LANG, CONT

def news():
    if CONT == 'US':
        print('Do you want to add another Keyword?')
        print('(1) Yes')
        print('(2) No')
        choice = input()
    if CONT == 'CO':
        print('¿Desea añadir otra Keyword?')
        print('(1) Si')
        print('(2) No')
        choice = input()
    return choice


def new_keyword():
    a,b,c = set_values()
    erasecsvdata()
    downdata(a, b, c)
    data = reading_data(a)
    data = clean_zeros(data)
    keywording = Keymarcas(data.Keyword)
    print('{} Keywords'.format(len(keywording)))
    getalldata(keywording,a)
    dates= rec_data(a)
    data = pd.concat([ keywords_data, dates])
    Key = keys(data)
    Key = textkey(Key)
    return Key,data

from pathlib import Path
path = Path.cwd()
import shutil, os, sys

def erasecsvdata():
    csv_files=list(filter(lambda x: '.csv' in x, os.listdir(path)))
    if CONT == 'CO':
        print ('Estos son los CSV en tu carpeta')
        for i in range(len(csv_files)):
            print ('({}) -> {}'.format(i+1, csv_files[i]))
        print ('Escoge los numeros de los CSV que quieres eliminar o 0 para pasar (ejemplo 1,3)')
        numero = input()
    if CONT == 'US':
        print('This are the CSV on your folder')
        for i in range(len(csv_files)):
            print('({} -> {})'.format(i+1, csv_files[i]))
        print('Choose the numbers of the CSV that you want to delete or choose 0 to pass (example 1,3)')
        numero = input()
    A = numero.split(',')
    A = [int(integer) for integer in A]
    for j in range(len(csv_files)):
        #print(csv_files[j])
        if j+1 in A:
            os.remove(csv_files[j])
    print('Los archivos restantes son:')
    print (list(filter(lambda x: '.csv' in x, os.listdir(path))))


from selenium import webdriver
from selenium.webdriver import Chrome
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.keys import Keys

import time


def downdata(Keyword,LANG, CONT):
    if os.path.exists( Keyword +'.csv') == False:
        options = webdriver.ChromeOptions()
        prefs = {"download.default_directory": str(path)}
        options.add_experimental_option("prefs", prefs)
        browser = webdriver.Chrome(executable_path='chromedriver', options = options) 
        browser.get('https://cocolyze.com/en/google-keyword-planner-tool#null')
    
        username = browser.find_element_by_id('keyword')
        username.send_keys(Keyword)


        select_Country = Select(browser.find_element_by_id('country'))
        select_Country.select_by_value(CONT)

        select_lang = Select(browser.find_element_by_id("lang"))
        select_lang.select_by_value(LANG)

        button = browser.find_element_by_id('submitBtn')
        button.click()


        time.sleep(15)

        ids = browser.find_elements_by_xpath("//*[@href]")
        for ii in ids:
            #print (ii.text)
            if ii.text == 'CSV':
                download = ii
        #print (download.text)
        download.click()


        time.sleep(5)
        browser.close()
        os.rename('suggestion.csv', Keyword+'.csv')


def getalldata(Keywording,Keyword):
    dire = "Data_"+ Keyword
    try:
    # Create target Directory
        os.mkdir(dire)
        print("Directory Created") 
    except FileExistsError:
        print("Directory already exists")
    options = webdriver.ChromeOptions()
    prefs = {"download.default_directory": str(path)}
    options.add_experimental_option("prefs", prefs)
    browser = webdriver.Chrome(executable_path='chromedriver', options = options) 
    browser.get('https://cocolyze.com/en/google-keyword-planner-tool#null')
    for i in range(len(Keywording)):
        if os.path.exists('Data_'+ Keyword +'/'+ Keywording[i]+str(i+1) +'.csv'):
            continue
        elif os.path.exists('Data_'+ Keyword+'/'+ Keywording[i] +'.csv')== False :
            time.sleep(1)
            username = browser.find_element_by_id('keyword')
            username.clear()
            username.send_keys(Keywording[i])

            select_Country = Select(browser.find_element_by_id('country'))
            select_Country.select_by_value(CONT)
            select_lang = Select(browser.find_element_by_id("lang"))
            select_lang.select_by_value(LANG)
            button = browser.find_element_by_id('submitBtn')
            button.click()
            time.sleep(15)
            ids = browser.find_elements_by_xpath("//*[@href]")
            for ii in ids:
                #print (ii.text)
                if ii.text == 'CSV':
                    download = ii
            #print (download.text)
            download.click()

            time.sleep(3)
  
            os.rename('suggestion.csv',  Keywording[i]+str(i+1)+'.csv')
    
            shutil.move(Keywording[i]+str(i+1)+'.csv', str(path)+"\Data_"+ Keyword)
            time.sleep(2)
    print("All data downloaded")
    browser.close()

Keyword, LANG, CONT = set_values()

erasecsvdata()

downdata(Keyword,LANG, CONT)

import pandas as pd
def reading_data(Keyword):
    col_list = ["Keyword","Search Volume","CPC"]
    keyword_Data=pd.read_csv(Keyword+'.csv', sep =";",usecols=col_list)
    print(keyword_Data.head())
    return keyword_Data
keyword_Data = reading_data(Keyword)

def clean_zeros(keyword_Data):
    col_names= keyword_Data.columns.tolist()
    for column in col_names:
        print("Null Data in {} = {}".format(column,keyword_Data[column].isnull().sum()))
        print("\n")
    Total = len(keyword_Data)
    for column in col_names[1:-1]:
         fixed = keyword_Data.loc[keyword_Data[column] > 0]
            
    Ft= len(fixed)
    rang = (Ft/Total)*100
   
    if CONT == 'CO':
        if rang < 60:
            print ('Se presenta más del 40% de datos sucios, se recomienda usar otra palabra')
        else :
            print('Tienes un buen dataset, continua')
    if CONT == 'US': 
        if rang < 60:
            print('The Data has 40% of Null data, you should use another Keyword')
        else:
            print('Great Dataset! Continue')
    
    return fixed



keyword_Data = clean_zeros(keyword_Data)

keyword_Data.head(30)


Marcas=['Nike', 'Adidas','Puma','Nautica','Levis', 'Under Armour', 'Zara', 'Diadora', 'Carolina']


from fuzzywuzzy import process, fuzz


# In[15]:


def get_matches(query, choices, limit=30):
    results = process.extract(query, choices, limit = limit, scorer = fuzz.partial_ratio)
    return results 


def Keymarcas(Keyword):
    keywords=[]
    for marca in Marcas:
        data = get_matches(marca,Keyword)
        for i in range(len(data)):
            if data[i][1] > 90:
                keywords.append(data[i][0]) 
    for i in range(len(keywords)):
        keywords[i]=keywords[i].replace('.','')
    return keywords

keywords = Keymarcas(keyword_Data.Keyword)

keywords

getalldata(keywords ,Keyword)

def rec_data(Keyword):
    path_2 = 'Data_'+Keyword
    files = [file for file in os.listdir(path_2) if not file.startswith('.')] # Ignore hidden files
    col_list = ["Keyword","Search Volume","CPC"]
    keywords_data = pd.DataFrame()

    for file in files:
        current_data = pd.read_csv(path_2+"/"+file, sep =";", usecols = col_list)
        keywords_data = pd.concat([keywords_data, current_data])
    for  i in range(len(keywords)):
        keywords_data= pd.concat([keywords_data,keyword_Data.loc[keyword_Data['Keyword']==keywords[2]]])
    
    keywords_data.to_csv("all_data_"+Keyword+".csv", index=False)
    return keywords_data
keywords_data = rec_data(Keyword)


keywords_data.head()


def keys(keywords_data):
    keywords_data = keywords_data.sort_values(by='Search Volume', ascending=False)
    keywords_data= keywords_data.reset_index(drop=True)

    keywords_data= keywords_data.loc[keywords_data['Search Volume']> keywords_data['Search Volume'].mean()]

    Key= keywords_data['Keyword'].values.tolist()
    Key = list(dict.fromkeys(Key))
    return Key
Key = keys(keywords_data)

def textkey(Key):
    Key = [text.lower() for text in Key]
    return Key
Key = textkey(Key)
Key[:30]

while True:
    choice = news()
    if int(choice) == 1:
        Key,data = new_keyword()
    else:
        break

Key = Keymarcas(Key)

Key[:30]

import re
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils


Key_Text= ', '.join(Key)
processed_text = re.sub('[^a-zA-Z]',r' ', Key_Text)


chars = sorted(list(set(Key_Text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
char_to_int

n_chars = len(Key_Text)
n_vocab = len(chars)
print ("Total Characters: ", n_chars)
print ("Total Vocab: ", n_vocab)

seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
    seq_in =  Key_Text[i:i + seq_length]
    seq_out = Key_Text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print ("Total Patterns: ", n_patterns)

X = np.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)


model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

try:
    os.mkdir("checkpoint")
    print("Directory Created") 
except FileExistsError:
    print("Directory already exists")

filepath="checkpoint\weights-improvement-{loss:.3f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit(X, y, epochs=25, batch_size=128, callbacks=callbacks_list)

def get_best_modeled():
    csv_files=list(filter(lambda x: '.hdf5' in x, os.listdir("checkpoint")))
    A = []
    B =[]
    C= np.zeros(len(csv_files))
    for i in range(len(csv_files)):
        A.append(csv_files[i].split('-'))
        B.append(A[i][2].split('.h'))
        C[i] = float(B[i][0])
    mini = np.min(C)
    return mini
mini = get_best_modeled()


filename = "checkpoint\weights-improvement-"+str(mini)+".hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

char_to_ints = dict((i, c) for i, c in enumerate(chars))

start = np.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print ("Seed:")
print ("\"", ''.join([char_to_ints[value] for value in pattern]), "\"")
# generate characters
for i in range(10):
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(n_vocab)
    prediction = model.predict(x, verbose=0)
    index = np.argmax(prediction)
    result = char_to_ints[index]
    seq_in = [char_to_ints[value] for value in pattern]
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
print ("\n Done.")


from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import get_file
import random
import io


# In[44]:


print('corpus length:', len(processed_text))

chars = sorted(list(set(processed_text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))


maxlen = 20
step = 3
sentences = []
next_chars = []
for i in range(0, len(processed_text) - maxlen, step):
    sentences.append(processed_text[i: i + maxlen])
    next_chars.append(processed_text[i + maxlen])
print('nb sequences:', len(sentences))



print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1



# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars), activation='softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)



model.summary()



def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, _):
    # Function invoked at end of each epoch. Prints generated text.
    print("****************************************************************************")
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, 30)
    for temperature in [0.2, 0.5, 1.0]:
        print('----- temperature:', temperature)

        generated = ''
        sentence = processed_text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(50):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, temperature)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()



import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Fit the model
print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

model.fit(x, y,
          batch_size=128,
          epochs=25,
          callbacks=[print_callback])






