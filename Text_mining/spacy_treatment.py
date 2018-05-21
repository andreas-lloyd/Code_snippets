import spacy, pandas as pd
from collections import Counter
nlp = spacy.load('en_core_web_lg')

def basic_treatment(text_list, pos_remove = [], special_words = [], remove_words = [], min_length = 0, check_removed = False, vocab_name = None):
    '''
    Function that does some basic treatment of the data according to spacy stuff
    
    Note that have added a "vocab_name" that will allow the printing of the entire vocabulary and word counts to an Excel
    '''
    
    #print(special_words)
    
    treated_data = []
    i = 0
    
    # Initially will remove full stops and dashes because they seem to cause problems
    text_list_treated = [text.replace('.', ' ').replace('-', ' ').replace("'", ' ').replace(';', ' ') for text in text_list]
    
    
    # We will declare a dictionary that will count how many times a word is removed from a text (note only 1 per text)
    if check_removed:
        removal_count = {}
    
    # Instead of looping over the texts, we use pipe - could increase batch size I guess
    for doc in nlp.pipe(text_list_treated, batch_size = 500):
        if i % 500 == 0:
            print('Have processed {} texts'.format(i))

        # Will remove certain things and potentially also get a list of the things we have removed
        token_list = [token.lemma_.lower() for token in doc if (token.is_alpha and not token.is_stop and token.pos_ not in pos_remove and len(token.lemma_) > min_length and token.lemma_.lower() not in remove_words and token.text.lower() not in remove_words) or (token.text.lower() in special_words or token.lemma_.lower() in special_words)]
        treated_data.append(' '.join(token_list))
        
        # If we want to - get the words that were removed most times
        if check_removed:
            all_tokens = [token.lemma_.lower() for token in doc]
            removed_words = list(set(all_tokens) - set(token_list))
            
            #if 'trabajar' in removed_words:
            #    print([(token, token in special_words) for token in all_tokens])
            
            for word in removed_words:
                if word in removal_count:
                    removal_count[word] += 1
                else:
                    removal_count[word] = 1
                
        i += 1

    print('Treated data - shape is {}'.format(len(treated_data)))   
    
    # Print the vocabulary to file
    if vocab_name:
        word_list = [word for doc in treated_data for word in doc.split()]
        word_count = Counter(word_list)
        
        count_frame = pd.DataFrame(word_count.most_common(len(word_count)), columns = ['word', 'count'])
        
        count_frame.to_excel(vocab_name, index = False)
        
    
    if check_removed:
        return treated_data, removal_count
    else:
        return treated_data