import wordcloud
import matplotlib.pyplot as plt

def get_wordcloud(documents, stopwords = [], width = 800, height = 400):
    '''
    Wrapper to get the word cloud - note that this AUTOMATICALLY considers bigrams
    Note that will also balance the absolute rank of a word and the number of times it appears
    
    Expecting list of texts as documents
    '''
    wc = wordcloud.WordCloud(stopwords = stopwords,
                            width = width, height = height).generate(' '.join(documents))
    
    # Make bigrams appear with an underscore for clarity
    wc.layout_ = [((layout[0][0].replace(' ', '_'), layout[0][1]), layout[1], layout[2], layout[3], layout[4]) for layout in wc.layout_]
    
    plt.figure(figsize = (200, 200))
    plt.imshow(wc)
    plt.axis('off')
    plt.show()