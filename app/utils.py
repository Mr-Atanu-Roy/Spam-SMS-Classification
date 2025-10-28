import nltk
from nltk.corpus import stopwords
import string


#func to transform text
def text_transformer(text):
    '''
    Transforms the input text by performing the following steps:
    # - Convert to lowercase
    # - Tokenization
    # - Remove special characters
    # - Remove stopwords and punctuation
    # - Stemming
    # ?Parameters:
    text (str): The input text message to be transformed.
    # ?Returns:
    str: The transformed text after preprocessing.
    '''

    #1. converting to lower case
    text = text.lower()

    # 2. Tokenize
    text = nltk.word_tokenize(text)

    #3. Remove special chars
    y = []
    for word in text:
        if word.isalnum:
            y.append(word)

    text = y[:]     #shallow copy
    y.clear()

    #4. Remove stopwords and punctuation
    for word in text:
        if word not in stopwords.words("english") and word not in string.punctuation:
            y.append(word)

    text = y[:]
    y.clear()

    #5. Stemming: bring words to root form
    ps = nltk.PorterStemmer()
    for word in text:
        y.append(ps.stem(word))

    return " ".join(y)