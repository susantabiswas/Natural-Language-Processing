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
#arg: string,dict,dict,dict,dict
#loads the corpus for the dataset and makes the frequency count of quadgram and trigram strings
def loadCorpus(file_path,bi_dict,tri_dict,quad_dict,vocab_dict):

    w1 = ''    #for storing the 3rd last word to be used for next token set
    w2 = ''    #for storing the 2nd last word to be used for next token set
    w3 = ''    #for storing the last word to be used for next token set
    token = []
    word_len = 0

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
                i += 1

            #make the token list into a string    
            content = " ".join(token)
            token = content.split()
            word_len = word_len + len(token)  

            if not token:
                continue

            #add the last word from previous line
            if w3!= '':
                token.insert(0,w3)

            temp0 = list(ngrams(token,2))

            #since we are reading line by line some combinations of word might get missed for pairing
            #for trigram
            #first add the previous words
            if w2!= '':
                token.insert(0,w2)

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

            #count the frequency of the bigram sentences
            for t in temp0:
                sen = ' '.join(t)
                bi_dict[sen] += 1

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
    return word_len
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
#arg: dict,dict,dict,dict,dict,int
#creates dict for storing probable words with their probabilities for a trigram sentence
def createProbableWordDict(bi_dict,tri_dict,quad_dict,prob_dict,vocab_dict,token_len):
    i = 0
    with open('quad_dict.txt','r') as file:
    # for quad_sen in quad_dict:
        for quad_sen in file:
            prob = 0.0
            quad_token = quad_sen.split()
            tri_sen = ' '.join(quad_token[:3])
            tri_count = tri_dict[tri_sen]

            if tri_count != 0:
                prob = interpolatedProbability(quad_token,token_len, vocab_dict, bi_dict, tri_dict, quad_dict,
                                l1 = 0.25, l2 = 0.25, l3 = 0.25 , l4 = 0.25)

            if tri_sen not in prob_dict:
                prob_dict[tri_sen] = []
                prob_dict[tri_sen].append([prob,quad_token[-1]])
            else:
                prob_dict[tri_sen].append([prob,quad_token[-1]])
            i += 1
    print('i:',i)
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
    
####################################################################################
#returns: string
#arg: void
#for taking input from user
def takeInput():
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
    return sen
####################################################################################

#return:int
#arg:list,dict,dict,dict,dict
#computes the score for test data
def computeTestScore(test_sent,tri_dict,quad_dict,vocab_dict,prob_dict):
    #increment the score value if correct prediction is made else decrement its value
    score = 0
    w = open('test_result.txt','w')
    for sent in test_sent:
        sen_token = sent[:3]
        sen = " ".join(sen_token)
        correct_word = sent[3]
        
        result = doPrediction(sen,prob_dict)
        if result == correct_word:
            s = sen +" : "+result+'\n'
            w.write(s)
            score+=1

    w.close()
    return score

#######################################################################################

#return:float
#arg:list,int,dict,dict,dict,dict
#computes the score for test data
def computePerplexity(test_quadgrams,token_len,tri_dict,quad_dict,vocab_dict,prob_dict):
    
    perplexity = float(1.0)
    n = token_len

    for item in quad_dict:
        sen_token = item.split()
        sen = ' '.join(sen_token[0:3])
        prob = quad_dict[item]/tri_dict[sen]
        perplexity = perplexity * ( prob**(1./n))

    return perplexity

#######################################################################################


#returns: float
#arg: float,float,float,float,list,list,dict,dict,dict,dict
#for calculating the interpolated probablity
def interpolatedProbability(quad_token,token_len, vocab_dict, bi_dict, tri_dict, quad_dict,
                            l1 = 0.25, l2 = 0.25, l3 = 0.25 , l4 = 0.25):
    
    sen = ' '.join(quad_token)
    prob =(   
              l1*(quad_dict[sen] / tri_dict[' '.join(quad_token[0:3])]) 
            + l2*(tri_dict[' '.join(quad_token[1:4])] / bi_dict[' '.join(quad_token[1:3])]) 
            + l3*(bi_dict[' '.join(quad_token[2:4])] / vocab_dict[quad_token[2]]) 
            + l4*(vocab_dict[quad_token[3]] / token_len)
          )
    return prob

####################################################################################

#return: void
#arg:string,string,dict,dict,dict,dict,dict
#Used for testing the Language Model
def trainCorpus(train_file,test_file,bi_dict,tri_dict,quad_dict,vocab_dict,prob_dict):
      
    test_result = ''
    score = 0
    #load the training corpus for the dataset
    token_len = loadCorpus('training_corpus.txt',bi_dict,tri_dict,quad_dict,vocab_dict)
    print("---Processing Time for Corpus Loading: %s seconds ---" % (time.time() - start_time))

    start_time1 = time.time()
    writeQuads(quad_dict)
    #creates a dictionary of probable words 
    createProbableWordDict(bi_dict,tri_dict,quad_dict,prob_dict,vocab_dict,token_len)
    #sort the dictionary of probable words 
    sortProbWordDict(prob_dict)
    gc.collect()
    print(len(prob_dict))
    print("---Processing Time for Creating Probable Word Dict: %s seconds ---" % (time.time() - start_time1))

    test_data = ''
    #Now load the test corpus
    with open('testing_corpus.txt','r') as file :
        test_data = file.read()

    #remove punctuations from the test data
    test_data = removePunctuations(test_data)
    test_token = test_data.split()

    #split the test data into 4 words list
    test_token = test_data.split()
    test_quadgrams = list(ngrams(test_token,4))

    #print(len(test_token))
    start_time1 = time.time()
    score = computeTestScore(test_quadgrams,tri_dict,quad_dict,vocab_dict,prob_dict)
    print('Score:',score)
    print("---Processing Time for computing score: %s seconds ---" % (time.time() - start_time1))

    start_time2 = time.time()
    perplexity = computePerplexity(test_token,token_len,tri_dict,quad_dict,vocab_dict,prob_dict)
    print('Perplexity:',perplexity)
    print("---Processing Time for computing Perplexity: %s seconds ---" % (time.time() - start_time2))

    test_result += 'TEST RESULTS\nScore: '+str(score) + '\nPerplexity: '+str(perplexity)
    with open('test_results.txt','w') as file:
        file.write(test_result)


#############################################################################################

#returns: string
#arg: void
#for taking input from user
def takeInput():
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
    return sen
####################################################################################

#returns: void
#arg: dict
#for writing the contents of quad_dict to a text file
def writeQuads(quad_dict):
    with open('quad_dict.txt','w') as file:
        for quad in quad_dict:
            file.write(quad+'\n')
####################################################################################

def main():

    #variable declaration
    tri_dict = defaultdict(int)            #for keeping count of sentences of three words
    quad_dict = defaultdict(int)           #for keeping count of sentences of three words
    vocab_dict = defaultdict(int) #for storing the different words with their frequencies    
    prob_dict = OrderedDict()   #for storing the probabilities of probable words for a sentence
    bi_dict = defaultdict(int)

    train_file = 'training_corpus.txt'
    test_file = 'testing_corpus.txt'
    #load the corpus for the dataset
    trainCorpus(train_file,test_file,bi_dict,tri_dict,quad_dict,vocab_dict,prob_dict)

if __name__ == '__main__':
  main()

