#how to use stopwords for filtering the tokenss

from nltk.corpus import stopwords
import nltk



#arg: list
#return: list
def remove_stopwords(token):
	stop_words = stopwords.words('english')
	#print(stop_words)
	filter_token = []
	for word in token:
		if word not in stop_words:
			filter_token.append(word)

	return filter_token

def main():
	stop_words = stopwords.words('english')
	print(stop_words)

	sen = 'This is seeker,testing currently the working of stopwords.Looks awesome !'
	token = nltk.word_tokenize(sen)
	print(sen)
	print(token)
	print(remove_stopwords(token))
	

if __name__ == '__main__' :
	main()