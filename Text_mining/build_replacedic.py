import pandas as pd

def build_replacedic(excel_path):
    '''
    Give an Excel path and build a dictionary of form "pattern" : "replacement"
    If find a NULL - then replace with ''
    '''
    
    replace_excel = pd.read_excel(excel_path)
    
    # Get nulls out
    replace_excel['Replace'][replace_excel['Replace'].isnull()] = ''
    
    # Get dic of replacements - will remove up to the end of the word
    replacements = replace_excel.apply(lambda x: {'\\b' + x['Find'] + '\\b' if '*' not in x['Find'] else x['Find'] : x['Replace']}, axis = 1)
    
    # Then turn into dic 
    return {find: replace for replace_pair in replacements.values.tolist() for find, replace in replace_pair.items()}