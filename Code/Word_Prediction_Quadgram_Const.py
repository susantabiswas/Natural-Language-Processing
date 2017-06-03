#TIME COMPLEXITY :
"""
  FOR WORD PREDICTION : O(1)
  FOR WORD PREDICTION WITH 'R'TH RANK: O(R)
"""
#import the modules necessary
from nltk.util import ngrams
from collections import defaultdict
from collections import OrderedDict
import string
import time
import gc

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

#returns: string
#arg: string,dict,int
#does prediction for the the sentence
def doPrediction(sen,prob_dict,rank = 1):
  if sen in prob_dict:
    if rank <= len(prob_dict[sen]):
     return prob_dict[sen][rank-1][1]
    else:
      return prob_dict[sen][0][1]
  else:
    return "Can't predict"

####################################################################################

#returns: void
#arg: dict,dict,dict
#creates dict for storing probable words with their probabilities for a trigram sentence
def createProbableWordDict(tri_dict,quad_dict,prob_dict):
    for quad_sen in quad_dict:
      prob = 0.0
      quad_token = quad_sen.split()
      tri_sen = ' '.join(quad_token[:3])
      tri_count = tri_dict[tri_sen]

      if tri_count != 0:
        prob = quad_dict[quad_sen]/tri_count
      
      if tri_sen not in prob_dict:
        prob_dict[tri_sen] = []
        prob_dict[tri_sen].append([prob,quad_token[-1]])
      else:
        prob_dict[tri_sen].append([prob,quad_token[-1]])


    prob = None
    tri_count = None
    quad_token = None
    tri_sen = None

####################################################################################

#returns: void
#arg: dict
#for writing the probable word dict in text file
def writeProbWords(prob_dict):
  with open('probab_dict.txt','w') as file:
    for key in prob_dict:
      file.write(key+'  '+str(prob_dict[key])+'\n')

####################################################################################

#returns: void
#arg: dict
#for sorting the probable word acc. to their probabilities
def sortProbWordDict(prob_dict):
  for key in prob_dict:

    if len(prob_dict[key])>1:
        sorted(prob_dict[key],reverse = True)
      # print(prob_dict1[key])

####################################################################################


def main():

  #variable declaration
  tri_dict = defaultdict(int)            #for keeping count of sentences of three words
  quad_dict = defaultdict(int)           #for keeping count of sentences of three words
  vocab_dict = defaultdict(int) #for storing the different words with their frequencies    
  prob_dict = OrderedDict()   #for storing the probabilities of probable words for a sentence
  

  #load the corpus for the dataset
  loadCorpus('corpusfile.txt',tri_dict,quad_dict,vocab_dict)
  print("---Preprocessing Time for Corpus loading: %s seconds ---" % (time.time() - start_time))
  
  start_time1 = time.time()
  
  #creates a dictionary of probable words 
  createProbableWordDict(tri_dict,quad_dict,prob_dict)
  #sort the dictionary of probable words 
  sortProbWordDict(prob_dict)
  # writeProbWords(prob_dict)
  gc.collect()
  print("---Preprocessing Time for Creating Probable Word Dict: %s seconds ---" % (time.time() - start_time1))
  
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
  
  start_time2 = time.time()
  prediction = doPrediction(sen,prob_dict)
  print("Word Prediction:",prediction)
  print("---Time for Prediction Operation: %s seconds ---" % (time.time() - start_time2))
  
if __name__ == '__main__':
  main()

