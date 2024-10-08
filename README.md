# Klasifikasi Sentimen Masyarakat di Twitter Menggunakan SVM Studi Kasus Karens Dinner

Klasifikasi dibuat untuk mendeteksi sentimen masyarakat berdasarkan dataset yang diambil dari Twitter API. Analisis sentimen ini dilakukan untuk mengukur sentimen masyarkat indonesia yang kental dengan budaya timur 'sopan santun' dengan hadirnya restoran Karens Dinner yang memiliki ciri khas Karens yakni 'Jutek dan Galak'. Sentimen analisis ini menggunakan Multilayer Perceptron dengan ekstraksi fitur TF-IDF (Term Frequency and Inverse Document Frequency) 

## Table of Contents:

- Crawling Data
- Cleaning Data
- Labeling
- Tokenization
- Stop Removal
- Normalization
- Data Visualization
- SVM Modelling
- Model Evaluation
- Model Testing

## Crawling Data
Menggunakan Twitter API, saya mengumpulkan 999 tweet dengan kata kunci "Karen's Dinner" untuk analisis sentimen dan tren percakapan.
```python
import tweepy 
import pandas as pd
api_key='lBYOjuTbLKvQPE0tTnND8ApbX'
api_key_secret='Ic54M23MnUWjI1rUpTDynxp1PKZPgQWpK1qg7pPGG0zHPOwVmd'
access_token= '1127055339688345600-CxJDnhJ8iUqjYPa5f73xoAOLkVezks'
access_token_secret= 'tKOtjB7ZjrO5vfpV4cPjUtfaMFT8lie7RhE2oRaNdty4D'

auth = tweepy.OAuthHandler(api_key,api_key_secret)
auth.set_access_token(access_token,access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)

search_key = "Karen's Diner"

csv_file = open(search_key+".csv","a+",newline="",encoding="utf-8")
csv_writer = csv.writer(csv_file)
t = []

for tweet in tweepy.Cursor(api.search_tweets, q=search_key, count=1000, lang="id",result_type="recent").items(1000):
    print (tweet.text)
    t.append(tweet.text.encode("utf-8"))
    tweets = [tweet.text.encode("utf-8")]
    csv_writer.writerow(tweets)
    
dicsTweets = {"teks":t}
df = pd.DataFrame(dicsTweets, columns = ["teks"])
```

## Cleaning Data
Library regex digunakan untuk membersihkan data dengan cara mendeteksi dan menghapus mention, URL, hashtag, serta karakter khusus lainnya dari teks, sehingga data menjadi lebih bersih dan siap untuk dianalisis.
```python
import re
def CleanTxt(text):
  text = re.sub(r"@(\w+)", ' ', text)       #mentions
  text = re.sub('@[^\s]+','',text)          #tags
  text = re.sub('https?:\/\/\S+', '', text) #url
  text = re.sub('RT[\s]+', '', text)        #retweet
  text = re.sub(r'#', '', text)             #tanda #
  text = re.sub(r',', '', text)             #tanda ,
  text = re.sub(r'$', '', text)             #tanda $
  text = re.sub(r'-', '', text)             #tanda -
  text = re.sub(r'[^\w]|_', ' ', text)      #punctuations
  text = text.lower()                       #lowercase
  text = re.sub('[0-9]+', '', text)
  text = re.sub(r'b', '', text)
  text = re.sub(r'xf', '', text)
  text = re.sub(r'xc', '', text)
  text = re.sub(r'x', '', text)
  text = re.sub(r'xs', '', text)
  text = re.sub(r'xe', '', text)
  return text

# Cleaning the text
df['Tweets'] = df['Tweets'].apply(CleanTxt)
df.drop_duplicates(subset ="Tweets", keep = 'first', inplace = True)
```

## Labeling
Label dibuat secara manual untuk mengkategorikan sentimen dalam tweet karena pendekatan ini dipercaya lebih akurat dalam menangkap nuansa dan konteks bahasa.

## Tokenization
Tokenization diperlukan untuk memecah teks menjadi kata-kata individual, yang memungkinkan analisis sentimen yang lebih terstruktur dan akurat serta memudahkan identifikasi pola dan pengolahan data lebih lanjut.
```python
def tokenization(text):
    text = re.split('\W+',text)
    return text

df['Tokenization'] = df['Tweets'].apply(lambda x:tokenization(x.lower()))
```

## Stop Removal
Stop removal adalah penghapusan kata-kata yang tidak diperlukan seperti kata hubung. Kata hubung bisa mengurangi akurasi model jika diikut sertakan. Hal ini sulit mengidentifkasi sentimen dalam kata hubung.
```python
stopword = nltk.corpus.stopwords.words('Indonesian')

def remove_stopwords(text):
    text = [word for word in text if word not in stopword]
    return text

df['Stop_Removal'] = df['Tokenization'].apply(lambda x: remove_stopwords(x))
```

## Normalization
Normalization merupakan proses mengubah atau memperbaiki kata yang tidak baku menjadi kata baku
```python
normalizad_word = pd.read_csv('KarensDiner2.csv')

normalizad_word_dict = {}

for index, row in normalizad_word.iterrows():
    if row[0] not in normalizad_word_dict:
        normalizad_word_dict[row[0]] = row[0] 

def normalized_term(document):
    return [normalizad_word_dict[term] if term in normalizad_word_dict else term for term in document]
df['Normalisasi'] = df['Stop_Removal'].apply(normalized_term)
```

## Data Visualization
![1](https://github.com/user-attachments/assets/1999f34f-27d1-4849-ac5d-c5f321b5b191)
![2](https://github.com/user-attachments/assets/a0ffd15f-971a-4dea-8dc9-fbca5e2262b6)

## SVM Modelling
```python
train, test = train_test_split(data, test_size=0.1, random_state=1)
X_train = train["Tweets"].values
X_test = test["Tweets"].values
y_train = train["Label"]
y_test = test["Label"]

kfolds = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.8, random_state=20)

def tokenize(text): 
    tknzr = TweetTokenizer()
    return tknzr.tokenize(text)

def stem(doc):
    return (stemmer.stem(w) for w in analyzer(doc)) 

vectorizer = CountVectorizer(
    analyzer = 'word',
    tokenizer = tokenize,
    lowercase = True,
    ngram_range=(1, 1))

np.random.seed(1)

pipeline_svm = make_pipeline(vectorizer, SVC(probability=True, kernel="linear", class_weight="balanced"))

grid_svm = GridSearchCV(pipeline_svm,param_grid = {'svc__C': [0.01, 0.1, 1]}, cv = kfolds, scoring="roc_auc", verbose=1, n_jobs=-1) 

grid_svm.fit(X_train, y_train)
grid_svm.score(X_test, y_test)
```

## Model Evaluation
```python
def get_roc_curve(model, X, y):
    pred_proba = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, pred_proba)
    return fpr, tpr

roc_svm = get_roc_curve(grid_svm.best_estimator_, X_test, y_test)
```
![modelevaluation](https://github.com/user-attachments/assets/8df360ec-993a-487b-be8f-f9f33842dd94)

## Model Testing
```python
grid_svm.predict(["Karens Diner pelayannya pemarah"])
```
Menghasilkan sentimen Negatif `array([0], dtype=int64)`

```python
grid_svm.predict(["Karens Diner saat ini banyak pengunjung"])
```
Menghasilkan sentimen Positif `array([1], dtype=int64)`

