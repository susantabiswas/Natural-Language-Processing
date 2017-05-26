#Word Prediction using Quadgram

"""
      Loads the corpus line by line into the memory .Uses function to encode keys for the hash table for minimising 
      space usage by the keys of the dictionary
"""

from nltk.util import ngrams
from collections import defaultdict
from collections import OrderedDict
import nltk
import string


#returns: string
#arg: string
#remove punctuations and make the string lowercase
def removePunctuations(sen):

    #split the string into word tokens
    temp_l = sen.split()
    i = 0

    #changes the word to lowercase and removes punctuations from it
    for word in temp_l :
      for l in word :
          if l in string.punctuation:
              word = word.replace(l," ")
      temp_l[i] = word.lower()
      i=i+1   
    
    #spliting is being don here beacause in sentences line here---so after punctuation removal it should 
    #become "here so"   
    content = " ".join(temp_l)
    
    return content

####################################################################################


#return: string
#arg:list,list,dict
#for encoding keys for the dictionary
#for encoding keys ,index has been used for each unique word   
#for mapping keys with their index
def encodeKey(s,index,vocab_dict):
    key = ''
    #print (s)
    for t in s:
        #print (t)
        if t not in vocab_dict:
            vocab_dict[t] = index[0]
            index[0] = index[0] + 1

        key = key + str(vocab_dict[t]) + '#'  
    #print(key)
    return key

#######################################################################################

#arg: list
#return: string,dict
#for decoding keys 
def decodeKey(s,vocab_dict):
    key = ''
    l = []
    item = list(vocab_dict.items())
      
    temp_l =  s.split('#')
    del temp_l[len(temp_l)-1]
    
    index = 0
    for c in temp_l:
        if c != ' ':
            index = int(c)
            l.append(item[index][0])

    key = ' '.join(l)    
    return key
#######################################################################################


#returns : void
#arg: string,dict,dict,dict,list
#loads the corpus for the dataset and makes the frequency count of quadgram and trigram strings
def loadCorpus(filename,tri_dict,quad_dict,vocab_dict,index):
    w1 = ''    #for storing the 3rd last word to be used for next token set
    w2 = ''    #for storing the 2nd last word to be used for next token set
    w3 = ''    #for storing the last word to be used for next token set
    i = 0
    sen = ''
    token = []

    with open(filename,'r') as file:
        #read the data line by line
        for line in file:
            token = line.split()
            i = 0
            for word in token :
                for l in word :
                    if l in string.punctuation:
                        word = word.replace(l," ")
                token[i] = word.lower()
                i=i+1   

            content = " ".join(token)
            token = content.split()

            if not token:
                continue
            
            #first add the previous words
            if w2!= '':
                token.insert(0,w2)
            if w3!= '':
                token.insert(1,w3)
            
            
            #tokens for trigrams
            temp1 = list(ngrams(token,3))

            if w1!= '':
                token.insert(0,w1)

            #tokens for quadgrams
            temp2 = list(ngrams(token,4))
            
            #count the frequency of the trigram sentences
            for t in temp1:
                sen = encodeKey(t,index,vocab_dict)
                tri_dict[sen] += 1

            #count the frequency of the quadgram sentences
            for t in temp2:
                sen = encodeKey(t,index,vocab_dict)
                quad_dict[sen] += 1


            #then take out the last 3 words
            n = len(token)

            w1 = token[n -3]
            w2 = token[n -2]
            w3 = token[n -1]
#######################################################################################


#returns : float
#arg : string sentence,string word,dict,dict
def findprobability(s,w,tri_dict,quad_dict):
    c1 = 0 # for count of sentence 's' with word 'w'
    c2 = 0 # for count of sentence 's'
    s1 = s + w
    
    if s1 in quad_dict:
        c1 = quad_dict[s1]
    if s in tri_dict:
        c2 = tri_dict[s]
   
    if c2 == 0:
        return 0
    return c1/c2
#######################################################################################    


#returns : void
#arg: string,dict,dict,dict,list
def doPrediction(sen,tri_dict,quad_dict,vocab_dict,index):
    
    #remove punctuations and make it lowercase
    temp_l = sen.split()
    i = 0
    
    for word in temp_l :
        for l in word :
            if l in string.punctuation:
                word = word.replace(l," ")
        temp_l[i] = word.lower()
        i=i+1   
        
    content = " ".join(temp_l)
    temp_l = content.split() 
    
    #encode the sentence before checking
    sen = encodeKey(temp_l,index,vocab_dict)
    
    max_prob = 0
    #when there is no probable word available
    #now for guessing the word which should exist we use quadgram
    right_word = 'apple' 
    
    for word in vocab_dict:
        #print(word)
        #encode the word before checking
        dict_l = []
        dict_l.append(word)
        word = encodeKey(dict_l,index,vocab_dict)
        
        prob = findprobability(sen,word,tri_dict,quad_dict)
        
        if prob > max_prob:
            max_prob = prob
            right_word = word
    
    #decode the right word       
    right_word = decodeKey(right_word,vocab_dict)
    
    print('Word Prediction is :',right_word)

#######################################################################################



def main():

    tri_dict = defaultdict(int)
    quad_dict = defaultdict(int)
    vocab_dict = OrderedDict()   #for mapping of words with their index ==> key:word value:index of key in dict\n",
    index = [0]   #list for assigning index value to keys\n",

    loadCorpus('mycorpus.txt',tri_dict,quad_dict,vocab_dict,index)

    cond = False
    #take input
    while(cond == False):
        sen = input('Enter the string\n')
        sen = removePunctuations(sen)
        temp = sen.split()
        if len(temp) < 3:
            print("Please enter atleast 3 words !")
        else:
            cond = True
            temp = temp[-3:]
            sen = " ".join(temp)
    
    doPrediction(sen,tri_dict,quad_dict,vocab_dict,index)


if __name__ == '__main__':
    main()
    
