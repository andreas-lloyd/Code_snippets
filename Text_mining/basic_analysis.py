from collections import Counter
import nltk

def cloud_n_count(text_list, most_common = 10, n_gram = None, most_common_ng = 10):
    '''
    Wrapper to get word cloud and word counts for basic analysis - can also give n_gram argument to check for bigrams etc
    '''
    
    # Split list of words into separate words and get counter
    word_list = [word for text in text_list for word in nltk.tokenize.wordpunct_tokenize(text)]
    word_count = Counter(word_list)
    
    print('The {} most common words are the following:'.format(most_common))
    print(word_count.most_common(most_common))
    
    # Then for ngrams as well
    if n_gram:
        # For each text, split it up into list of words, get those ngrams, then put into a long list
        text_ngrams = [text_ngram for text in text_list for text_ngram in list(nltk.ngrams(nltk.tokenize.wordpunct_tokenize(text), n_gram))]
        ngram_count = Counter(text_ngrams)
        
        print('The {} most common ngrams are the following:'.format(most_common_ng))
        print(ngram_count.most_common(most_common_ng))
    
    # And finally get the wordcloud
    get_wordcloud(text_list)
    
    if n_gram:
        return word_count, ngram_count
    else:
        return word_count