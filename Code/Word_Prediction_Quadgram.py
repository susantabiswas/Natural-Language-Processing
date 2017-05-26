#import the modules necessary
from nltk.util import ngrams
from collections import defaultdict
import nltk
import string
import time

start_time = time.time()


#returns : void
#arg: string,dict,dict,dict
#loads the corpus for the dataset and makes the frequency count of quadgram and trigram strings
def loadCorpus(file_path,tri_dict,quad_dict,vocab_dict):

	w1 = ''    #for storing the 3rd last word to be used for next token set
	w2 = ''    #for storing the 2nd last word to be used for next token set
	w3 = ''    #for storing the last word to be used for next token set
	token = []

	#open the corpus file and read it line by line
	with open(file_path,'r') as file:
	    for line in file:

	        #split the line into tokens
	        token = line.split()
	        i = 0

	        #for each word in the token list ,remove pucntuations and change to lowercase
	        for word in token :
	            for l in word :
	                if l in string.punctuation:
	                    word = word.replace(l," ")
	            token[i] = word.lower()
	            i=i+1

	        #make the token list into a string    
	        content = " ".join(token)
	        token = content.split()
	        #word_len = word_len + len(token)  

	        if not token:
	            continue
	        
	        #since we are reading line by line some combinations of word might get missed for pairing
	        #for trigram
	        #first add the previous words
	        if w2!= '':
	            token.insert(0,w2)
	        if w3!= '':
	            token.insert(1,w3)
	        
	        
	        
	        #tokens for trigrams
	        temp1 = list(ngrams(token,3))

	        #insert the 3rd last word from previous line for quadgram pairing
	        if w1!= '':
	            token.insert(0,w1)
	        
	        #add new unique words to the vocaulary set if available
	        for word in token:
	            if word not in vocab_dict:
	                vocab_dict[word] = 1
	            else:
	                vocab_dict[word]+= 1
	                
	        #tokens for quadgrams
	        temp2 = list(ngrams(token,4))
	       
	        
	        #count the frequency of the trigram sentences
	        for t in temp1:
	            sen = ' '.join(t)
	            tri_dict[sen] += 1

	        #count the frequency of the quadgram sentences
	        for t in temp2:
	            sen = ' '.join(t)
	            quad_dict[sen] += 1


	        #then take out the last 3 words
	        n = len(token)

	        #store the last few words for the next sentence pairing
	        w1 = token[n -3]
	        w2 = token[n -2]
	        w3 = token[n -1]

####################################################################################


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



#returns : float
#arg : string sentence,string word,dict,dict
def findprobability(s,w,tri_dict,quad_dict):
    c1 = 0 # for count of sentence 's' with word 'w'
    c2 = 0 # for count of sentence 's'
    s1 = s + ' ' + w
    
    if s1 in quad_dict:
        c1 = quad_dict[s1]
    if s in tri_dict:
        c2 = tri_dict[s]
    
    if c2 == 0:
        return 0
    return c1/c2
    
####################################################################################

#returns : void
#arg: string,dict,dict,dict
def doPrediction(sen,tri_dict,quad_dict,vocab_dict):
    
    sen = removePunctuations(sen)

    max_prob = 0
    #when there is no probable word available
    #now for guessing the word which should exist we use quadgram
    right_word = 'apple' 
    
    for word in vocab_dict:
        prob = findprobability(sen,word,tri_dict,quad_dict)
        if prob > max_prob:
            max_prob = prob
            right_word = word
    
    print('Word Prediction is :',right_word)

#################################################################3




def main():

	#variable declaration
	tri_dict = defaultdict(int)            #for keeping count of sentences of three words
	quad_dict = defaultdict(int)           #for keeping count of sentences of three words
	vocab_dict = defaultdict(int) #for storing the different words with their frequencies    
	
	#load the corpus for the dataset
	loadCorpus('corpusfile.txt',tri_dict,quad_dict,vocab_dict)
	print("---Preprocessing Time: %s seconds ---" % (time.time() - start_time))
	
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
	
	start_time1 = time.time()
	doPrediction(sen,tri_dict,quad_dict,vocab_dict)
	print("---Time for Prediction Operation: %s seconds ---" % (time.time() - start_time1))
	

if __name__ == '__main__':
  main()

