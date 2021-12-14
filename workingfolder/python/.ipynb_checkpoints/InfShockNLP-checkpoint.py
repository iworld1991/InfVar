# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ## News-based Shock to Inflation 
#
# - The codes in this notebook construct topic-specific shock to inflation based on news articles. 
# - News media: New York Times. (Wall Streat Journal for future) 
# - Currently, the sample period is 2009-2019, and there are,in total, 4670 articles.

# +
import numpy as np
import pandas as pd
import nltk  
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import re

from IPython.display import Markdown, display
def printmd(string):
    display(Markdown(string))
    
import matplotlib.pyplot as plt
# %matplotlib inline

import math as mt
# -

# ### 1. Data Pre-processing

test_txt = open('../TextData/InfNYT1.txt').read()

# + {"code_folding": []}
raw_txt =''
for txt_id in range(10):
    myfile = ('../TextData/InfNYT'+str(txt_id+1)+'.txt')
    #print(myfile)
    txt_temp = open(myfile).read()
    #print('lenght of the txt is '+ str(len(txt_temp)))
    raw_txt += txt_temp
print('Total length of the text is '+str(len(raw_txt)))

# + {"code_folding": [0]}
## split raw texts into articles 
divider = '\n____________________________________________________________\n'
articles = raw_txt.split(divider)

# + {"code_folding": []}
#articles[30]
# -

len(articles)

# + {"code_folding": [0]}
## create empty dataframe to store articles information
index = np.arange(len(articles))
columns = ['author',
          'text',
          'subject',
          'location',
          'company',
          'people',
           'organization',
          'title',
          'doctype',
           'date']

dt = pd.DataFrame(index=index,columns = columns)

# + {"code_folding": [0]}
## new codes that extract information about each article
author_str0 = '\nAuthor:'
author_str1 = '\nPublication info:'
author_str = [author_str0,author_str1]

text_str0 = '\nFull text:'
text_str1 = '\nSubject:'
text_str =[text_str0,text_str1]

subject_str0 = '\nSubject:'
subject_str1 = '\nLocation:'
subject_str = [subject_str0,subject_str1]

location_str0 = '\nLocation:'
location_str1 = '\nPeople:'
location_str = [location_str0,location_str1]

people_str0 = '\nPeople:'
people_str1 = '\nCompany / organization:'
people_str = [people_str0,people_str1]

organ_str0 = '\nCompany / organization:'
organ_str1 = '\nURL:'
organ_str = [organ_str0,organ_str1]

title_str0 = '\nTitle:'
title_str1 = ': \xa0'
title_str = [title_str0,title_str1]

doc_type_str0 = '\nDocument type:'
doc_type_str1 = '\nProQuest document ID:'
doc_type_str = [doc_type_str0,doc_type_str1]

date_str0 = '\nLast updated:'
date_str1 ='\nDatabase:'
date_str = [date_str0,date_str1]

str_list = {'author':author_str,
           'text':text_str,
           'subject':subject_str,
           'location':location_str,
            'organization':organ_str,
           'people':people_str,
           'title':title_str,
           'doctype':doc_type_str,
           'date':date_str}

# first, check if all articles contain these strings. 

#for article in articles:
#    for string in str_list:
#        i = 0
#        if string[0] not in article:
#            i+=1
#        if i >0:
#            print("There is "+str(i)+ ' article, for which there is no such a string' )


# second, locate them 

for i,article in enumerate(articles):
    for info,string in str_list.items():
        if string[0] in article and string[1] in article:
            loc_str = article.find(string[0])
            loc_str1= article.find(string[1])
            #print(string)
            #print(str(loc_str) + ' is where the string starts')
            extract = article[loc_str:loc_str1].split(string[0])[1]
            extract = extract.strip('\n')
            dt[info][i] = extract  
# -

dt.head()

# +
## dates 

#dates = pd.to_datetime(dt['date'])
#dt['date'] = dates

# + {"code_folding": [0]}
## types of the documents
print( "The types of documents include "+ str( set( dt['doctype'] ) ) )

# + {"code_folding": []}
## number of authors
print("Three are " + str( len(set(dt['author']) ) ) + " unique authors for " + str(len(articles)) + " articles.")

# + {"code_folding": [0]}
## subjects 
subject_lst = []

for i in range(len(dt)):
    if str(dt['subject'][i]) != 'nan':
        subjects = dt['subject'][i].split(';')
        subject_lst += subjects
    subject_lst = [subject.strip("'") for subject in subject_lst]
    subject_set =   set(subject_lst)
print("There are " + str( len(subject_set) ) +" unique subjects")

subject_ct = [subject_lst.count(subject) for subject in subject_set]
subject_set_sort = sorted(zip (subject_ct,subject_set), reverse=True )

# the most common subjects 
subject_set_sort[:30]
# -

## most common subjects
sub_freq = nltk.FreqDist(subject_lst)
list(sub_freq.most_common(10))
plt = sub_freq.plot(20)

# + {"code_folding": [0]}
##location

location_lst = []

for i in range(len(dt)):
    if str(dt['location'][i]) != 'nan':
        locations = dt['location'][i].split(';')
        location_lst += locations
    location_lst = [location.strip("'") for location in location_lst]
    location_set =   set(location_lst)
print("There are " + str( len(location_set) ) +" unique locations")

location_ct = [location_lst.count(location) for location in location_set]
location_set_sort = sorted(zip (location_ct,location_set), reverse=True )

# the most common locations 
location_set_sort[:30]

# + {"code_folding": [0]}
##organizations/companies

organization_lst = []

for i in range(len(dt)):
    if str(dt['organization'][i]) != 'nan':
        organizations = dt['organization'][i].split(';')
        organization_lst += organizations
    organization_lst = [organization.strip("'") for organization in organization_lst]
    organization_set =   set(organization_lst)
print("There are " + str( len(organization_set) ) +" unique organizations")

organization_ct = [organization_lst.count(organization) for organization in organization_set]
organization_set_sort = sorted(zip (organization_ct,organization_set), reverse=True )

# the most common locations 
organization_set_sort[:30]
# -



# ## 2. Tokenizing

# + {"code_folding": [0]}
# tokenize the raw texts
tokenizer = nltk.RegexpTokenizer(r'\w+')  # ignore putuations 
toks = tokenizer.tokenize(raw_txt)
# convert to lower case
toks = [w.lower() for w in toks]

## clean tokens

### remove stop words
stop_words = set(stopwords.words('english'))
toks = [w for w in toks if not w in stop_words]
toks.sort()

### lemmatizing

lemmatizer = WordNetLemmatizer()
toks = [lemmatizer.lemmatize(w) for w in toks]

### exclude numbers

#toks = [w for w in toks if not r'[0-9]*']

### exclude news paper information

source_inf = ['publication','proquest','copyright']
toks = [w for w in toks if not w in source_inf]
# -

# count frequency 
freq = nltk.FreqDist(toks)
print('Number of unique tokens is '+str(len(freq)))

list(freq.most_common(100))

plt = freq.plot(20)

## ngrams 
toks.sort()
#bigram = list(nltk.bigrams(toks))
#bigram[-100:-1]

# +
# tokenize each articles
# -


