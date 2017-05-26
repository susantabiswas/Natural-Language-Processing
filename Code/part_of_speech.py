#To show the part of speech associated with a word using pos_tag
import nltk
from nltk.tokenize import PunktSentenceTokenizer


def main():
	file = open('mycorpus.txt','r')
	content = file.read()

	test_data = content
	sample_data = content
	my_sent_tokenizer = PunktSentenceTokenizer(test_data)


	sen_tokens = my_sent_tokenizer.tokenize(sample_data)

	for word in sen_tokens:
		token = nltk.word_tokenize(word)
		print(nltk.pos_tag(token))

	file.close()

if __name__ == '__main__':
	main()