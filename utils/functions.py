import pandas as pd
import pandas as pd
import numpy as np
import nltk
import re
import unicodedata
import pickle
from string import punctuation
from nltk.tokenize import TweetTokenizer
from nltk.stem import RSLPStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from nltk.tokenize import TweetTokenizer

tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)

##############################################################################################

def isRetweet(phrase):
    retweet = re.match(r"RT",phrase)
    if retweet:
        return True
    else:
        return False
        
def toFixPunctuation(phrase):
    phrase = re.sub(r"[\.]+", " . ", phrase)  #SUBSTITUI VÁRIOS PONTOS CONSECUTIVOS, ONDE HOUVER, POR UM ÚNICO PONTO
    phrase = re.sub(r"[!]+", " ! ", phrase)  #SUBSTITUI VÁRIOS PONTOS DE EXCLAMAÇÃO, ONDE HOUVER, POR UM ÚNICO
    phrase = re.sub(r"[?]+", " ? ", phrase)  #SUBSTITUI VÁRIOS PONTOS DE INTERROGAÇÃO, ONDE HOUVER, POR UM ÚNICO
    return(phrase)

def changePatterns(phrase):
    phrase = re.sub(r"\sc\s", " com ", phrase, flags=re.I)
    phrase = re.sub(r"\"", " ", phrase, flags=re.I)
    phrase = re.sub(r"\sñ\s", " não ", phrase, flags=re.I) 
    phrase = re.sub(r"\sq\s", " que ", phrase, flags=re.I) 
    phrase = re.sub(r"\stb\w+", " também ", phrase, flags=re.I) 
    phrase = re.sub(r"\stbm\w+", " também ", phrase, flags=re.I) 
    phrase = re.sub(r"\spq\s", " porque ", phrase, flags=re.I) 
    phrase = re.sub(r"^[k]+", "risada", phrase, flags=re.I)
    phrase = re.sub(r"\s[k]+", "risada", phrase, flags=re.I) 
    phrase = re.sub(r"\s[p]\s", " para ", phrase, flags=re.I) 
    phrase = re.sub(r"\sp/\s", " para ", phrase, flags=re.I) 
    phrase = re.sub(r"\svc\s", " você ", phrase, flags=re.I) 
    phrase = re.sub(r"\svcs\s", " vocês ", phrase, flags=re.I) 
    phrase = re.sub(r"\sgnt\s", " gente ", phrase, flags=re.I) 
    phrase = re.sub(r"\shj\s", " hoje ", phrase, flags=re.I)
    return(phrase)

def changeLink(phrase):
     phrase = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", " link ", phrase) 
     return(phrase)

def toLower(phrase): 
     return(phrase.lower())

def isNotice(phrase):
    phrase = re.sub(r"#G1", " Globo ", phrase)
    phrase = re.sub(r"#GloboNews", " Globo ", phrase)
    phrase = re.sub(r"#CentralGloboNews", " Globo ", phrase)
    phrase = re.sub(r"#OsPingosNosIs", " notícia ", phrase)
    phrase = re.sub(r"#UOL", " UOL ", phrase)
    phrase = re.sub(r"#Folha", " Folha ", phrase)
    return phrase
    
def toFixBlankSpaces(phrase):
    phrase = phrase.strip() #RETIRA ESPAÇOS NO COMEÇO E NO FINAL DA STRING
    phrase = re.sub(r"[\s]+", " ", phrase)  #SUBSTITUI VÁRIOS ESPAÇOS CONSECUTIVOS, ONDE HOUVER, POR UM ÚNICO ESPAÇO
    return(phrase)
       
def getHashtags(phrase):
    hashtags = re.findall(r"#\w+", phrase)
    if(len(hashtags)>0):
        return hashtags
    else:
        return False

def removeHashtag(phrase):
     phrase = re.sub(r"#\w+", "", phrase) 
     return(phrase)   

def removeLink(phrase):
     phrase = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", "", phrase) 
     return(phrase)             

def getMention(phrase):
    mention = re.findall(r"@\w+", phrase)
    return mention


def removeMention(phrase):
    phrase = re.sub(r"@\w+", "", phrase) 
    return(phrase)

################################################################################################

def get_hashtag(data):
    hashtags = []
    for tweet in data:
        print(tweet)
        match = re.findall(r'#\w+',tweet)
        if match:
            hashtags.append(z)
    return hashtags

def get_data(filename, header=0, names=None, skiprows=0):
    df = pd.read_excel(filename, names=names, index_col=None, header=header, skiprows=skiprows)
    return df

def get_data_generic(filename,columns_names=None, skip_rows=0):
    df = pd.read_excel(filename, names=columns_names, index_col=None, header=None, skiprows=skip_rows)
    return df

def getSubset(data, columns):
    subset = data[columns]
    return subset

def removeSpecialCharactere(word):
    nfkd = unicodedata.normalize('NFKD', word)
    withoutAccent = u"".join([c for c in nfkd if not unicodedata.combining(c)])
    return re.sub('[^a-zA-Z 0-9%$-]', '', withoutAccent)

def palavras_unicas(base):
    palavra = base.keys()
    return palavra

def get_radicals(list_words):
    stemmer = nltk.stem.RSLPStemmer()
    words_radicals = []
    for word in list_words:
        words_radicals.append(stemmer.stem(word))
    return words_radicals

def clean_tweet(tweet):
    t                  = re.sub(' q ',' que ',tweet)
    link_removed       = re.sub('https?://[A-Za-z0-9./]+','',t)
    hashtags_removed   = re.sub(r'#[a-zA-Z]+','',link_removed)
    user_removed       = re.sub(r'@[A-Za-z0-9]+','',hashtags_removed)
    punct_edited       = re.sub(r'[.!,;:?]',' . ',user_removed)
    pattern_authorized = re.sub('[^A-Za-zÀ-ú0-9$%)(\s-]', '', punct_edited)
    line_break_removed = re.sub('\n', ' ', pattern_authorized)
    clean_tweet        = line_break_removed.strip().lower()
    return clean_tweet

def remove_hifen(tweet):
    t = re.sub('\s-\s',' ',tweet)
    return t

def remove_punctuation(phrase):
    import string
    not_permited = (string.punctuation).replace("-","")
    without_punct = [caractere for caractere in phrase if caractere not in not_permited]
    without_punct = ''.join(without_punct)
    return without_punct

def remove_spaces_between_words(words):
    tokenized = [w for w in tknzr.tokenize(words)]
    words_list = []
    for word in tokenized:
        words_list.append(word.strip())
    return " ".join(words_list)
        
def pre_process_data(list_to_append, data_to_process):
# for index, row in df_selected.head().iterrows():
# print(removeSpecialCharactere(clean_tweet(row[0])),index)
    for row in data_to_process:
        phrase_processed = removeSpecialCharactere(clean_tweet(row))
        phrase_tokenized = [t for t in tknzr.tokenize(phrase_processed) if t.isalpha()]
        list_to_append.append(" ".join(phrase_tokenized))

def get_radicals_stop_words(stop_words):
    list_stop_words_no_accent = []
    for stop_word in stop_words:
        stopword_no_accent = removeSpecialCharactere(stop_word)
        list_stop_words_no_accent.append(stopword_no_accent)
    list_stop_words_radical = get_radicals(list_stop_words_no_accent)
    stop_words_dist_freq    = nltk.FreqDist(list_stop_words_radical)
    list_stop_words_final   = palavras_unicas(stop_words_dist_freq)
    return list_stop_words_final

def extrator_palavras(document):
    doc = set(document)
    caracteristicas = {}
    for palavras in lista_palavras_unicas_tweets_final:
        caracteristicas['%s' % palavras] = palavras in doc
    return caracteristicas

def tokenized_tweets(data):
    tweets_tokenizeds = []
    for tweet in data:
        tokenized = [w for w in tknzr.tokenize(tweet)]
        tweets_tokenizeds.append(tokenized)
    return tweets_tokenizeds

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()