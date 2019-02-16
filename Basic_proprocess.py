
# coding: utf-8

# In[1]:


import codecs


# In[2]:


train_file_name = 'origin_data/origin_train.txt'
valid_file_name = 'origin_data/origin_valid.txt'
text_file_name = 'origin_data/origin_test.txt'


# In[3]:


valid_labels = []
valid_texts = []

with codecs.open(valid_file_name, 'r', encoding='utf8') as f:
    for line in f:
        splits = str.split(line, ' ||| ')
        valid_labels.append(splits[0])
        valid_texts.append(splits[1])


# In[4]:


train_labels = []
train_texts = []

with codecs.open(train_file_name, 'r', encoding='utf8') as f:
    for line in f:
        splits = str.split(line, ' ||| ')
        train_labels.append(splits[0])
        train_texts.append(splits[1])


# In[5]:


test_texts = []

with codecs.open(text_file_name, 'r', encoding='utf8') as f:
    for line in f:
        splits = str.split(line, ' ||| ')
        test_texts.append(splits[1])


# ### Remove Punctuations Stop Words, Add POS tags, Lower cases, Lemmatize

# In[192]:


import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.tag.mapping import tagset_mapping
import math
import string


# In[193]:


from nltk.tokenize import word_tokenize


# In[214]:


tokenizer = word_tokenize
lemmatizer = WordNetLemmatizer()
pos_map = tagset_mapping('en-ptb', 'universal')


# In[215]:


def lemmatize(word, tag, lemmatizer=lemmatizer):
    word = word.lower()
    if tag == 'NOUN':
        word = lemmatizer.lemmatize(word, pos='n')
    elif tag == 'VERB':
        word = lemmatizer.lemmatize(word, pos='v')
    elif tag == 'ADJ':
        word = lemmatizer.lemmatize(word, pos='a')
    elif tag == 'ADV':
        word = lemmatizer.lemmatize(word, pos='r')
    elif tag == 'NUM':
        try:
            word = float(word)
            word = str(int(math.log10(abs(word)+1)))
        except:
            pass
    return word


# In[250]:


def preprocess(sent, tokenizer=tokenizer):
    words = []
    for word in str.split(sent, ' '):
        if word == '@.@':
            words.append('.')
            continue
        if word == '@,@':
            words.append(',')
            continue
        if word != '@-@':
            words.append(word)
    sent = ' '.join(words)
    words = tokenizer(sent)
    pos_tags = nltk.pos_tag(words)
    new_words = []
    for pos_tag in pos_tags:
        word, tag = pos_tag
        tag = pos_map[tag]
        word = lemmatize(word, tag)
            
#         if word not in stop_words:
        new_words.append('_'.join([word, tag]))

    return ' '.join(new_words)


# In[251]:


with codecs.open('valid.txt', 'w', encoding='utf8') as f:
    for line in valid_texts:
        f.write(preprocess(line)+'\n')


# In[252]:


count = 0
with codecs.open('train_no_oov.txt', 'w', encoding='utf8') as f:
    for line in train_texts:
        f.write(preprocess(line)+'\n')
        count += 1
        print('{}/{}'.format(count, len(train_texts)),end='\r')


# In[253]:


with codecs.open('test.txt', 'w', encoding='utf8') as f:
    for line in test_texts:
        f.write(preprocess(line)+'\n')


# ### Show Set of Labels

# In[65]:


labels = set(train_labels)


# In[70]:


num2labels = list(labels)
print(num2labels)


# In[71]:


labels2num = {}
for i,label in enumerate(num2labels):
    labels2num[label] = i


# In[72]:


labels2num


# In[75]:


with open('valid_label.txt', 'w') as f:
    for label in valid_labels:
        if label == 'Media and darama':
            label = 'Media and drama'
        f.write(str(labels2num[label])+'\n')


# In[76]:


with open('train_label.txt', 'w') as f:
    for label in train_labels:
        f.write(str(labels2num[label])+'\n')


# ### Convert Infrequent Words to OOV

# In[6]:


from collections import Counter


# In[7]:


train_file_name = 'train_no_oov.txt'
valid_file_name = 'valid.txt'
text_file_name = 'test.txt'


# In[8]:


vocab = Counter()

with codecs.open(train_file_name, 'r', encoding='utf8') as f:
    for line in f:
        words = str.split(line, ' ')
        for word in words:
            word = str.split(word, '_')[0]
            vocab[word] += 1


# In[32]:


len(vocab)


# In[33]:


idx = -95015
vocab.most_common()[idx:idx+10]


# In[34]:


### get frequent words
good_vocab = vocab.most_common()[:idx]


# In[35]:


good_vocab = [word for word,_ in good_vocab]


# In[36]:


f2 = open('train.txt', 'w', encoding='utf8')
count = 0

with codecs.open(train_file_name, 'r', encoding='utf8') as f:
    for line in f:
        
        if len(line.strip()) == 0:
            f2.write('<None>' + '\n')
            count += 1
            print('{}'.format(count), end='\r')
        
        else:
            words = str.split(line, ' ')
            words[-1] = words[-1].strip()
            new_line = []
            for word in words:
                try:
                    word,tag = str.split(word, '_')
                except:
                    continue
                if word in good_vocab:
                    new_line.append(word)
                else:
                    new_line.append(tag)
            f2.write(' '.join(new_line) + '\n')

            count += 1
            print('{}'.format(count), end='\r')
            
#             if count == 100:
#                 break
        
f2.close()

