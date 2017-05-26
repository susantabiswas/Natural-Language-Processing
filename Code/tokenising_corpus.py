#different ways of tokenising the corpus
import nltk
from nltk.tokenize import word_tokenize
#from nltk.tokenize import PunktWordTokenizer #tokenizes the sentance into token ,the pucntuations also tokenizes
#from nltk.tokenize import WordPunktTokenizer #tokenizes the sentance into token ,the pucntuations remains with the word
from nltk.tokenize import sent_tokenize	#uses pretrained PunktSentanceTokenizer
from nltk.tokenize import PunktSentenceTokenizer	#the Tokenizer can be trained using this


def main():
	#sample string for testing
	sen = """This is seeker's testing,testing currently the working of tokenization of text.Looks awesome!.There are different ways of Tokenizing the string"""

	print('using word_tokenize:')
	print(word_tokenize(sen))
	print()

	print('using sent_tokenize:(uses the pretrained PunktTokenizer)')
	print(sent_tokenize(sen))
	print()

	#we use our sample string for training the tokenizer
	print('using PunctTokenizer:')
	custom_tokenizer = PunktSentenceTokenizer(sen)
	print(custom_tokenizer.tokenize(sen))


if __name__ == '__main__':
	main()