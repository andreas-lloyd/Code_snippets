from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pyLDAvis
import pyLDAvis.sklearn
import pandas as pd

def get_tdm(text_list, which = 'tfidf'):
    '''
    A wrapper to train our vectoriser - this isn't really that necessary but Like it to keep everything in one place
    These are some default features that work relatively well
    '''
    if which == 'tfidf':
        print('Getting TFIDF')
        vectorizer = TfidfVectorizer(min_df = 3, strip_accents = 'unicode', token_pattern = r'\w{1,}', ngram_range = (1, 2))
    else:
        print('Getting CountVectorizer')
        vectorizer = CountVectorizer(min_df = 3, strip_accents = 'unicode', token_pattern = r'\w{1,}', ngram_range = (1, 2))

    # Fit on our word list
    vectorizer.fit(text_list)

    # Return the vectorizer so we can transform later
    return vectorizer

def get_topics(text_list, n_topics, mode = 'LDA', vectorizer = 'tfidf', print_top = False, visualise = False, lda_name = None, assign_topics = False):
    '''
    Wrapper for topic modelling where only have to feed in the term matrix - can choose from LDA or NMF
    Add printing of top documents??
    '''
    
    # Get the vectorizer
    vectorizer = get_tdm(text_list, which = vectorizer)
    
    # Declare the model depending on the mode
    if mode == 'LDA':
        model = LatentDirichletAllocation(n_components = n_topics, max_iter = 50, learning_method = 'batch', evaluate_every = 1, random_state = 1, verbose = 1)
    else:
        model = NMF(n_components = n_topics, alpha = 0.1, random_state = 1, verbose = 1)
    
    # Get the matrix
    term_matrix = vectorizer.transform(text_list)
    
    # Then fit using the matrix
    model.fit(term_matrix)
    
    # Print top sentences
    model_transform = pd.DataFrame(model.transform(term_matrix))
    
    if print_top:
        for topic_n in range(n_topics):
            print('Getting top documents for topic {}'.format(topic_n + 1))
            top_5 = model_transform[i].sort_values().index[0:5]
            for doc_id in top_5:
                print(text_list[doc_id])
    
    
    # Get the LDAvis
    if visualise:
        prepared_viz = pyLDAvis.sklearn.prepare(model, term_matrix, vectorizer)
        
        if lda_name:
            pyLDAvis.save_html(prepared_viz, str(home_path / lda_name))
        else:
            pyLDAvis.show(prepared_viz)
    
    # If we want to will return the assigned topics for the input
    if assign_topics:
        print(len(text_list))
        print(len(model_transform.idxmax(axis = 1)))
        return model, vectorizer, model_transform.idxmax(axis = 1)
    
    return model, vectorizer

def display_topics(model, features, N):
    '''
    Wrapper to display the top words in a topic model
    '''
    for topic_n, topic in enumerate(model.components_):
        indices = topic.argsort()[::-1]
        
        print('Topic {}:'.format(topic_n + 1))
        print(' '.join([features[i] + ' (' + format(topic[i], '.2f') + ')' for i in indices[:N]]))
        print('\n')

def get_wordcloud_freq(frequency_dictionary):
    wc = wordcloud.WordCloud(stopwords = stopword_list,
                            width = 400, height = 200).generate_from_frequencies(frequency_dictionary)
    
    # Make bigrams appear with an underscore for clarity
    wc.layout_ = [((layout[0][0].replace(' ', '_'), layout[0][1]), layout[1], layout[2], layout[3], layout[4]) for layout in wc.layout_]

    plt.figure(figsize = (200, 200))
    plt.imshow(wc)
    plt.axis('off')
    plt.show()

def topic_wordcloud_weights(model, features):
    '''
    Give in a trained model and its get_feature_names
    '''
    for topic in model.components_:
        freq_dic = {word : freq for word, freq in zip(features, topic)}
        get_wordcloud_freq(freq_dic)