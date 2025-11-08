#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import spacy
import random 
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical 
from keras.models import Sequential
from keras.layers import Dense,LSTM,Embedding
from pickle import dump,load
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ### Project Name: Building an LSTM Text Generation Model
# 
# #### Description
# In this project, I'll use the book **"Alice in Wonderland"** to train a text generation model based on **LSTM (Long Short-Term Memory)**.  
# The model predicts the next word based on a sequence of previous words. Sequences from the book, will form our training data.
# 
# ---
# 
# #### Table of Contents
# 
# ##### Part 1: Preprocessing
# 1. Read the text file of *Alice in Wonderland*.  
# 2. Use **spaCy** to split the text into words and punctuation, and remove the punctuation.  
# 3. Divide the text into sequences.  
# 4. Use **Keras' Tokenizer** to assign each token an index based on its frequency in the text.  
# 5. Create the training data by splitting the entire text into sequences and a predicted word.
# 
# ##### Part 2: Building and Training the Model
# 6. Build the **LSTM model**.  
# 7. Train the model using the sequences prepared in Part 1.
# 
# ##### Part 3: Text Generation
# 8. Write a function that predicts the next word given a sequence, using the trained model.  
# 9. Generate new text using the model.
# 

# ### 📚 Data
# 
# The text of **Alice in Wonderland** was used from the **Gutenberg Project**.  
# It can be found and downloaded at: [https://www.gutenberg.org/ebooks/11](https://www.gutenberg.org/ebooks/11)
# 

# In[ ]:


# Part 1: Preprocessing


# In[2]:


# 1. Read the txt file of Alice in Wonderland

def read_file(filepath):
    with open(filepath, encoding='utf-8') as f:
        str_text = f.read()
    return str_text

Alice = read_file('Alice_in_Wonderland.txt')


# In[4]:


# 2. We use spacy to split the text into words and punctuations, and remove the punctuation 

# Load the medium English model, disabling unnecessary components
nlp = spacy.load('en_core_web_md', disable=['parser', 'tagger', 'ner'])


# In[5]:


nlp.max_length = 1000000


# In[13]:


# we separate punctuations and new lines from the actual text using the following function

def separate_punc(doc_text):
    return [token.text.lower() for token in nlp(doc_text) if token.text not in 
           '\n    \n \n\n \n\n\n!"-#$%&()--.*+,-/:;<=>?@[\\]^_`{|}~\t\n']


# In[15]:


tokens = separate_punc(Alice)


# In[16]:


len(tokens)


# In[17]:


# 3. We devide the text into sequences of 25 words, our network predicts #26

train_len = 25+1
text_sequences = []
for i in range(train_len, len(tokens)):
    seq = tokens[(i- train_len):i]
    text_sequences.append(seq)

text_sequences.append(seq)


# In[19]:


# 4. We use keras's tokenizer to give each token we have an index based on its frequency in the tex


# In[20]:


tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_sequences)


# In[21]:


sequences = tokenizer.texts_to_sequences(text_sequences)


# In[40]:


vocabulary_size = len(tokenizer.word_counts)
vocabulary_size


# In[28]:


# 5. Using our Tokenization we create the training data:
# splitting the entire text into sequences and a predicted word

sequences = np.array(sequences)
sequences


# In[34]:


X = sequences[:,:-1] # taking the sequences without the predicted word (in the last col): X


# In[31]:


y = sequences[:,-1] # the predicted word (label)


# In[32]:


# We turn the to-be-predicted word in each sequence and convert it to one-hot encoding
# This is for the soft max function in the end of the model 
y = to_categorical(y,num_classes=vocabulay_size+1)


# In[36]:


seq_len = X.shape[1]


# In[ ]:


# Part 2: Building and Training the model 


# In[53]:


# 6. We build our LSTM Model

def create_model(vocabulary_size,seq_len):
    model = Sequential()
    model.add(Embedding(vocabulary_size,seq_len,input_length = seq_len))
    model.add(LSTM(seq_len*6,return_sequences = True))
    model.add(LSTM(seq_len*6))
    model.add(Dense(seq_len*6,activation = 'relu'))
    
    model.add(Dense(vocabulary_size,activation = 'softmax'))
    
    model.compile(loss = 'categorical_crossentropy',optimizer = 'adam', metrics=['accuracy'])
    model.summary()
    return model


# In[54]:


model = create_model(vocabulary_size+1,seq_len)


# In[89]:


# 7. We use the sequences we prepared to train it 
# Since Traning with the parameters bellow took ~3 hours, I have saved my trained model
# and you can just load it
#model.fit(X,y,batch_size = 128, epochs = 200,verbose = 1)


# In[58]:


#model.save('My_Alice_LSTM_Model.h5')
#dump(tokenizer,open('My_Alice_Tokenizer','wb'))


# In[83]:


model = load_model('My_Alice_LSTM_Model.h5')
tokenizer = load(open('My_Alice_Tokenizer','rb'))


# In[ ]:


# Part 3: Text Generation
# 8. We write a function that predict the next word given a sequence, using our model


# In[85]:


def generate_text(model,tokenizer,seq_len,seed_text,num_gen_words):
    output_text = []
    input_text = seed_text
    for i in range(num_gen_words):
        # convert the seed to tokens 
        encoded_text = tokenizer.texts_to_sequences([input_text])[0] 
        # padding or cutting the sequence to be 25 tokens
        pad_encoded = pad_sequences([encoded_text],maxlen = seq_len,truncating='pre')
        #predict the next word
        pred = model.predict(pad_encoded, verbose=0)
        pred_word_ind = np.argmax(pred, axis=-1)[0]
        pred_word = tokenizer.index_word[pred_word_ind]
        # we update the input text since we predict more than one word
        input_text += ' '+pred_word
        # adding the predicted words to the output variable 
        output_text.append(pred_word)
    return ' '.join(output_text)
    


# In[86]:


# 9. We generate text 
# here we choose a random sequence for the book to be our input
random.seed(55)
random_pick = random.randint(0,len(text_sequences))
random_seed_text = text_sequences[random_pick]


# In[87]:


seed_text = ' '.join(random_seed_text)


# In[88]:


# and here we generate the text
generate_text(model,tokenizer,seq_len,seed_text= seed_text, num_gen_words=25)

