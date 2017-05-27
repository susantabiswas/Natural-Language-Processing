#Divides the corpus in 90:10 ratio.
#One file contains 90% of data and is saved in ''
#Second file contains 10% of data is used for testing

import string
import nltk

#returns : int
#arg: string
#loads the corpus and returns the numbere of words in the corpus
def loadCorpus(file_path):

	token = []
	word_len = 0

	#open the corpus file and read it line by line
	with open(file_path,'r') as file:
	    for line in file:

	        #split the line into tokens
	        token = line.split()
	        word_len = word_len + len(token)  

	return word_len


####################################################################################

#returns : void
#arg: string,int
#loads the corpus for the training the language model.10% of corpus will be used for testing and rest 90% for training
def partitionFile(file_path,word_count):
      token = []
      word_count = int(word_count*0.9)
      
      pos = 0
      word_len = 0

      #open the corpus file and read it line by line
      file = open(file_path,'r')
      train_file = open('training_corpus.txt','w')
      test_file = open('testing_corpus.txt','w') 

      line = file.readline()

      while line:
		#split the line into tokens
            token = line.split()

            #write the line to the training file
            train_file.write(line)

            word_len = word_len + len(token)  
            
            #quit training when 90% of the corpus has been read
            if word_len >= word_count:
                  pos = file.tell()
                  break;
            
            line = file.readline()

      #Prepare the testing data
      if word_count <= word_len:
            file.seek(pos)
            test_data = file.read();
            test_file.write(test_data)
            
      file.close()
      train_file.close()
      test_file.close()

#######################################################################################

def divideCorpus(filename):
	word_count = loadCorpus(filename)
	word_count = loadCorpus(filename)
	print(word_count)

	partitionFile(filename,word_count)


def main():
	filename = 'mycorpus.txt'
	divideCorpus(filename)
	

if __name__ == '__main__':
	main()