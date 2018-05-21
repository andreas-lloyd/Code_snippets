'''
Two wrappers for TFIDF and TDM just with the default features that I like
'''
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def get_tdm(train_text, which = 'tfidf'):
    '''
    A wrapper to train our vectoriser - this isn't really that necessary but Like it to keep everything in one place
    These are some default features that work relatively well
    '''
    if which == 'tfidf':
        print('Getting TFIDF')
        vectorizer = TfidfVectorizer(min_df = 3, strip_accents = 'unicode', token_pattern = r'\w{1,}', ngram_range = (1, 2), stop_words = 'english')
    else:
        print('Getting CountVectorizer')
        vectorizer = CountVectorizer(min_df = 3, strip_accents = 'unicode', token_pattern = r'\w{1,}', ngram_range = (1, 2), stop_words = 'english')

    # Fit on our word list
    vectorizer.fit(text_list)

    # Return the vectorizer so we can transform later
    return vectorizer