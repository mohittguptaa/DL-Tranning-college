import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk import pos_tag

data="""Natural language processing (NLP) is a machine learning technology that gives computers the ability to interpret, manipulate, and comprehend human-language.Natural language processing (NLP) is a machine learning technology that gives computers the ability to interpret, manipulate, and comprehend human language.Natural language processing (NLP) is a machine learning technology that gives computers the ability to interpret, manipulate, and comprehend human language."""
# print(data)

percept=nltk.download("averaged_perceptron_tagger")
punket=nltk.download('punkt')
w=word_tokenize(data)
print(w)
print("\n\n")

pos=pos_tag(w)
print(pos)
print("\n\n")

sent=sent_tokenize(data)
print(sent)

for i in sent:
    print("\n")
    print(i)

from nltk.stem import LancasterStemmer,RegexpStemmer,PorterStemmer,SnowballStemmer

lan=LancasterStemmer()
regex=RegexpStemmer('ing')
porter=PorterStemmer()
snow=SnowballStemmer('english')

# l=lan.stem("studying") #chang,study
# l=regex.stem("studying") #chang,study
# l=porter.stem("studying") #chang,studi
# l=snow.stem("studying") #chang,studi
# print(l)

from nltk.stem import WordNetLemmatizer
wordnet=nltk.download('wordnet')
print(wordnet)
wordLem=WordNetLemmatizer()
lem=wordLem.lemmatize("mice")
print("mice::",lem)