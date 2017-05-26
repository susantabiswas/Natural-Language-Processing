#Stemming a sentance using PorterStemmer
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

#arg: list
#return: list
def doStemming(token,ps):
	i = 0
	for word in token:
		token[i] = ps.stem(word)
		i +=1
	return token

def main():
	ps = PorterStemmer()

	sen = 'This is seeker,testing currently the working of stemming and wants to know about stem and how words are stemmed.'
	token = nltk.word_tokenize(sen)
	
	print('sen:\n',sen)
	print('\nTokens before:\n',token)
	print('\nTokens after:\n',doStemming(token,ps))
	

if __name__ == '__main__' :
	main()
