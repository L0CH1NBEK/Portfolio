import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score
import pickle



data = pd.read_csv('IMDB-Dataset.csv')
#print(data.shape)
data.sentiment.replace('positive',1, inplace=True)
data.sentiment.replace('negative',0, inplace=True)
def clean(text):
    cleaned = re.compile(r'<.*?>')
    return re.sub(cleaned, '', text)
data.review = data.review.apply(clean)

def is_special(text):
    rem = ''
    for i in text:
        if i.isalnum():
            rem = rem + i
        else:
            rem = rem + ' '
    return rem
data.review = data.review.apply(is_special)
#data.review[0]

def to_lower(text):
    return text.lower()

data.review = data.review.apply(to_lower)

def rem_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return [w for w in words if w not in stop_words]

data.review = data.review.apply(rem_stopwords)

def stem_txt(text):
    ss = SnowballStemmer('english')
    return " ".join([ss.stem(w) for w in text])

data.review = data.review.apply(stem_txt)


X = np.array(data.iloc[:,0].values)
y = np.array(data.sentiment.values)
cv = CountVectorizer(max_features = 1000) 
X = cv.fit_transform(data.review).toarray()

print("X.shape = ", X.shape)
print("y.shape = ", y.shape)

trainX, testX, trainY, testY = train_test_split(X,y, test_size = 0.2, random_state=9)

gnb, mnb, bnb = GaussianNB(), MultinomialNB(alpha=1.0, fit_prior = True), BernoulliNB(alpha=1.0, fit_prior = True)
gnb.fit(trainX, trainY)
mnb.fit(trainX, trainY)
bnb.fit(trainX, trainY)


ypg = gnb.predict(testX)
ymn = mnb.predict(testX)
ybn = gnb.predict(testX)

print("Gaussian = ", accuracy_score(testY, ypg))
print("Multinominal = ", accuracy_score(testY, ymn))
print("Bernoulli = ", accuracy_score(testY, ybn))

pickle.dump(bnb, open('model1.pkl', 'wb'))

rev = """Terrible. Complete trash. Brainless tripe. Insulting to anyone who isn't an 8 year old fan boy. Im actually pretty disgusted that this movie is making the money it is - what does it say about the people who brainlessly hand over the hard earned cash to be 'entertained' in this fashion and then come here to leave a positive 8.8 review?? Oh yes, they are morons. Its the only sensible conclusion to draw. How anyone can rate this movie amongst the pantheon of great titles is beyond me.

So trying to find something constructive to say about this title is hard...I enjoyed Iron Man? Tony Stark is an inspirational character in his own movies but here he is a pale shadow of that...About the only 'hook' this movie had into me was wondering when and if Iron Man would knock Captain America out...Oh how I wished he had :( What were these other characters anyways? Useless, bickering idiots who really couldn't organise happy times in a brewery. The film was a chaotic mish mash of action elements and failed 'set pieces'...

I found the villain to be quite amusing.

And now I give up. This movie is not robbing any more of my time but I felt I ought to contribute to restoring the obvious fake rating and reviews this movie has been getting on IMDb."""

f1 = clean(rev)
f2 = is_special(f1)
f3 = to_lower(f2)
f4 = rem_stopwords(f3)
f5 = stem_txt(f4)

bow, words = [], word_tokenize(f5)
for word in words:
    bow.append(words.count(word))
    

word_dict = cv.vocabulary_
pickle.dump(word_dict, open('bow.pkl', 'wb'))

inp = []
for i in word_dict:
    inp.append(f5.count(i[0]))
y_pred = bnb.predict(np.array(inp).reshape(1,1000))



