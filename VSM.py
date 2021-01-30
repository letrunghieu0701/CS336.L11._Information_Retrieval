# Các thư viện cần thiết

import math
import operator
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WhitespaceTokenizer 

import os
import re
import time
import string
import itertools

# Định nghĩa các biến

stop_words = stopwords.words('english')

white_space_tokenizer = WhitespaceTokenizer()

porter_stemmer = PorterStemmer()

wordnet_lemmatizer = WordNetLemmatizer()

ID_OF_DOC_FOR_QUERY = 0

# Các hàm thành phần

def getFilePathList(path):
    list_file_path = list()
    
    for file_name in os.listdir(path):
        full_file_path = path + "\\" + file_name
        
        if os.path.isfile(full_file_path):
            list_file_path.append(full_file_path)
            
    return list_file_path

def indexingDocument(list_file_path):
    id_path_of_files = dict()
    
    for index, file_path in enumerate(list_file_path):
        id_path_of_files[index] = file_path
    
    return id_path_of_files

def preprocessString(string_data, mode_of_preprocessing):
    # Lower the text
    preprocess_data = string_data.lower()
    
    # Remove Unicode characters
    preprocess_data = preprocess_data.encode('ascii', 'ignore').decode()
    
    # One letter in a word should not be present more than twice in continuation, ex: "I misssss youuu" -> "I miss youu"
    preprocess_data = ''.join(''.join(s)[:2] for _, s in itertools.groupby(preprocess_data)) 
    
    # Remove punctuations, each punctuation = space, ex: ""information @#$retrieval" -> "information    retrieval"
    preprocess_data = re.sub('[%s]' % re.escape(string.punctuation), ' ', preprocess_data)   
        
    # Tokenize word by white space
    preprocess_data = white_space_tokenizer.tokenize(preprocess_data)
    
############################################################################################   
    
    
    # Remove stop words
#     preprocess_data = [word for word in preprocess_data if word not in stop_words]
        
    # Stem word
#     preprocess_data = [porter_stemmer.stem(word) for word in preprocess_data]
  
    # Lemmatize word
#     preprocess_data = [wordnet_lemmatizer.lemmatize(word) for word in preprocess_data]

    if mode_of_preprocessing == 'stopWord-lemmatize':
        # Remove stop words
        preprocess_data = [word for word in preprocess_data if word not in stop_words]
        # Lemmatize word
        preprocess_data = [wordnet_lemmatizer.lemmatize(word) for word in preprocess_data]
        
    elif mode_of_preprocessing == 'stopWord-stem':
        # Remove stop words
        preprocess_data = [word for word in preprocess_data if word not in stop_words]
        # Stem word
        preprocess_data = [porter_stemmer.stem(word) for word in preprocess_data]
        
    elif mode_of_preprocessing == 'stopWord-noStem':
        # Remove stop words
        preprocess_data = [word for word in preprocess_data if word not in stop_words]
        
    elif mode_of_preprocessing == 'noStopWord-lemmatize':
        # Lemmatize word
        preprocess_data = [wordnet_lemmatizer.lemmatize(word) for word in preprocess_data]
        
    elif mode_of_preprocessing == 'noStopWord-stem':
        # Stem word
        preprocess_data = [porter_stemmer.stem(word) for word in preprocess_data]
        
    elif mode_of_preprocessing == 'noStopWord-noStem':
        pass
        

############################################################################################
        
    return preprocess_data

# test_a = "What have I done? And why? Who's this"
# test_a = preprocessString(test_a, 'noStopWord-noStem')

# test_b = "What have I done"
# test_b = preprocessString(test_b, 'stopWord-noStem')
# test_b

def create_termID_forQuery(dictionary_of_docs, query, mode_of_preprocessing):
    term_id = list()
    
    query = preprocessString(query, mode_of_preprocessing)
    
    for term in query:
        if term in dictionary_of_docs.keys():
            term_id.append([term, ID_OF_DOC_FOR_QUERY])
    
    return term_id

def create_termID_forDocument(id_path_of_files, mode_of_preprocessing):
    term_id = list()
    
    for index, file_path in id_path_of_files.items():
        with open(file_path, "r") as f:
            content = f.readlines()
            f.close()
            
            lyric = str()
            for i in range(2, len(content)):
                lyric = lyric + ' ' + content[i]           
            lyric = preprocessString(lyric, mode_of_preprocessing)
            
            for term in lyric:
                term_id.append([term, index])
    
    return term_id

def createDictionaryAndVectorDoc(term_id):
    dictionary = dict()
    vector_docs = dict()
        
    for term, id_of_doc in term_id:
        
        if id_of_doc not in vector_docs.keys():
            vector_docs[id_of_doc] = {term}
            
        elif term not in vector_docs[id_of_doc]:
            vector_docs[id_of_doc].add(term)
        
        
        # Nếu term chưa có trong dictionary thì thêm term, ndoc, id_tf vào
        if term not in dictionary.keys():
            dictionary[term] = {'ndoc': 1,
                                'id_tf': {id_of_doc: 1}}
            
        # Nếu term đã có trong dictionary rồi thì sẽ cập nhật các chỉ số ndoc, id_tf nếu thỏa điều kiện
        else:
            
            # Nếu term này đã xuất hiện trong id_tf thì chỉ cập nhật mỗi id_tf
            # (tức là cập nhật tần số xuất hiện của term trong document này)
            if id_of_doc in dictionary[term]['id_tf'].keys():
                dictionary[term]['id_tf'][id_of_doc] += 1
                
            # Nếu term này chưa xuất hiện trong id_tf thì phải cập nhật thêm ndoc lên 1 đơn vị
            # và cập nhật thêm một cặp id_tf mới cho term
            else:
                dictionary[term]['ndoc'] += 1
                dictionary[term]['id_tf'][id_of_doc] = 1
            
    return [dictionary, vector_docs]


# Thay giá trị tf bằng weight tf: 1 + math.log(tf, 10)

def calculateTF(dictionary):
    for term in dictionary.keys():        
        for doc_id in dictionary[term]['id_tf'].keys():
            tf = dictionary[term]['id_tf'][doc_id]
            dictionary[term]['id_tf'][doc_id] = 1 + math.log(tf, 10)

# Thay giá trị ndoc bằng IDF = log(number_of_docs / ndoc, 10)

def calculateIDF(dictionary, number_of_docs):
    for term in dictionary.keys():
        ndoc = dictionary[term]['ndoc']
        dictionary[term]['ndoc'] = math.log(number_of_docs / ndoc, 10)

def calculateIDF_forQuery(dictionary_of_docs, dictionary_of_query):
    for term in dictionary_of_query.keys():
        ndoc = dictionary_of_docs[term]['ndoc']
        dictionary_of_query[term]['ndoc'] = ndoc

# Thay giá trị id_tf bằng tf-idf = tf * idf

def calculate_TF_IDF(dictionary):
    for term in dictionary.keys():
        idf_of_term = dictionary[term]['ndoc']
        
        for id_of_doc in dictionary[term]['id_tf'].keys():
            tf_of_term_in_doc = dictionary[term]['id_tf'][id_of_doc]
                      
            dictionary[term]['id_tf'][id_of_doc] = tf_of_term_in_doc * idf_of_term

# vector là list

def calculateDenominatorOfVector(vector):
    denominator = 0
    
    for value in vector:
        denominator += math.pow(value, 2)
        
    sqrt_denominator = math.sqrt(denominator)
    
    if sqrt_denominator > 0:
        return sqrt_denominator
    else:
        return 1


def normalizeDictionary(dictionary, vector_docs):
    
    for id_of_doc in vector_docs.keys():
        vector_weight_of_document = list()
        
        for term in vector_docs[id_of_doc]:
            tf_idf_of_term = dictionary[term]['id_tf'][id_of_doc]
            vector_weight_of_document.append(tf_idf_of_term)
            
        denominator_of_vector = calculateDenominatorOfVector(vector_weight_of_document)
        
        for term in vector_docs[id_of_doc]:
            tf_idf_of_term = dictionary[term]['id_tf'][id_of_doc]
            dictionary[term]['id_tf'][id_of_doc] = tf_idf_of_term / denominator_of_vector
            
def Union(set_of_doc, set_of_query):    
    union_set = set().union(set_of_doc, set_of_query)
    return union_set

def processDocumentsFromFoler(path, mode_of_preprocessing):
    
    list_file_path = getFilePathList(path)

    id_path_of_files = indexingDocument(list_file_path)

    term_id = create_termID_forDocument(id_path_of_files, mode_of_preprocessing)

    dictionary, vector_docs = createDictionaryAndVectorDoc(term_id)

    calculateTF(dictionary)

    calculateIDF(dictionary, len(id_path_of_files))

    calculate_TF_IDF(dictionary)

    normalizeDictionary(dictionary, vector_docs)
    
    return [dictionary, vector_docs, id_path_of_files]

def processQuery(dictionary_of_docs, query, mode_of_preprocessing):
    
    term_id = create_termID_forQuery(dictionary_of_docs, query, mode_of_preprocessing)
    
    dictionary_of_query, vector_docs = createDictionaryAndVectorDoc(term_id)

    calculateTF(dictionary_of_query)

    calculateIDF_forQuery(dictionary_of_docs, dictionary_of_query)

    calculate_TF_IDF(dictionary_of_query)

    normalizeDictionary(dictionary_of_query, vector_docs)
    
    return dictionary_of_query

def calculateCosineSimilarity(dictionary_of_docs, dictionary_of_query):
    
    weight_of_documents = dict()
    
    for term in dictionary_of_query.keys():
        weight_of_term_in_query = dictionary_of_query[term]['id_tf'][ID_OF_DOC_FOR_QUERY]      
        
        for id_of_doc in dictionary_of_docs[term]['id_tf'].keys():
            
            weight_of_term_in_document =  dictionary_of_docs[term]['id_tf'][id_of_doc]
            
            if id_of_doc not in weight_of_documents.keys():
                weight_of_documents[id_of_doc] = weight_of_term_in_document * weight_of_term_in_query
            else:
                weight_of_documents[id_of_doc] += weight_of_term_in_document * weight_of_term_in_query
          
    
    sorted_weight_of_documents = sorted(weight_of_documents.items(),
                                        key = operator.itemgetter(1),
                                        reverse = True)
    return sorted_weight_of_documents

def calculateEuclidSimilarity(dictionary_of_docs, dictionary_of_query, vector_docs):
    
    weight_of_documents = dict()
    
    set_of_query = set(dictionary_of_query.keys())
    
    for id_of_doc in vector_docs.keys():
        union_term = Union(vector_docs[id_of_doc], set_of_query)
        
        for term in union_term:
            
            # weight_of_term_in_document
            if term in vector_docs[id_of_doc]:
                weight_of_term_in_document = dictionary_of_docs[term]['id_tf'][id_of_doc]
            else:
                weight_of_term_in_document = 0
            
            # weight_of_term_in_query
            if term in dictionary_of_query.keys():
                weight_of_term_in_query = dictionary_of_query[term]['id_tf'][ID_OF_DOC_FOR_QUERY]
            else:
                weight_of_term_in_query = 0
                
            if id_of_doc not in weight_of_documents.keys():
                weight_of_documents[id_of_doc] = math.pow(weight_of_term_in_document - weight_of_term_in_query, 2)
            else:
                weight_of_documents[id_of_doc] += math.pow(weight_of_term_in_document - weight_of_term_in_query, 2)
    
    for id_of_doc in weight_of_documents.keys():
        value = weight_of_documents[id_of_doc]
        weight_of_documents[id_of_doc] = math.sqrt(value)
    
    sorted_weight_of_documents = sorted(weight_of_documents.items(),
                                        key=operator.itemgetter(1),
                                        reverse = False)
    return sorted_weight_of_documents

def print_topK_result(sorted_weight_of_documents,
                      id_path_of_files,
                      top_k_results_to_return,
                      similarity_measure):
    
    length_of_docs_returned = len(sorted_weight_of_documents)
    
    # Đối với code này:
    # Euclid sẽ không bao giờ xảy ra trường hợp không có kết quả trả về, những kết quả top-k có thể có những giá trị sqrt(2)
    # Còn Cosine thì vẫn có trường hợp không có kết quả trả về, tại vì xét trên những term trong query
    if length_of_docs_returned == 0:
        print("\nNo such song relate to query!!!\n")
        return -1
    
    if similarity_measure == 'Cosine':
        print("\n--------------- Higher Score Is Better ---------------\n")
    elif similarity_measure == 'Euclid':
        print("\n--------------- Lower Score Is Better ---------------\n")
    
    if top_k_results_to_return > length_of_docs_returned:
        top_k_results_to_return = length_of_docs_returned
        print("Only " + str(top_k_results_to_return) + " relate to query\n")
        
    for index in range(top_k_results_to_return):
        score = sorted_weight_of_documents[index][1]
        print('Top', index + 1, ':', score)
        
    print('\n')
    
    for index in range(top_k_results_to_return):
        file_id = sorted_weight_of_documents[index][0]
        file_path = id_path_of_files[file_id]
        score = sorted_weight_of_documents[index][1]
        
        with open(file_path, 'r') as file:
            content = file.read().strip().split('\n')
            file.close()
            print('Position ' + str(index + 1) + ':')
            print('Song:', content[0])
            print('Artist:', content[1])
            print('Lyric:', content[2])
            print('\n')      

def init_Vector_Space_Model(path, mode_of_preprocessing):
    return processDocumentsFromFoler(path, mode_of_preprocessing)


def searchDocumentWithQuery(dictionary_of_docs,
                            vector_docs,
                            id_path_of_files,
                            query,
                            top_k_results_to_return,
                            mode_of_preprocessing,
                            similarity_measure):
    
    dictionary_of_query = processQuery(dictionary_of_docs, query, mode_of_preprocessing)
    
    sorted_weight_of_documents = list()
    
    start_time = time.process_time()
    
    if similarity_measure == 'Cosine':
        sorted_weight_of_documents = calculateCosineSimilarity(dictionary_of_docs, dictionary_of_query)
    elif similarity_measure == 'Euclid':
        sorted_weight_of_documents = calculateEuclidSimilarity(dictionary_of_docs, dictionary_of_query, vector_docs)
        
    end_time = time.process_time()
    
    print('\nTime To Search: ', end_time - start_time, " seconds")
        
    print_topK_result(sorted_weight_of_documents, id_path_of_files, top_k_results_to_return, similarity_measure)

# Đường dẫn tới thư mục chứa file txt

file_path_of_docs = 'D:\\A_Truy_Van_Thong_Tin_Da_Phuong_Tien\\song_spotify\\8982_txt'

print('\n* Stopword removal options:')
print('\t 1. Stopword removal using NLTK stopword list')
print('\t 2. No stopword removal')
stopword_removal_option = int(input('Please choose stopword removal option: '))

while(stopword_removal_option not in [1, 2]):
    stopword_removal_option = int(input('Please choose stopword removal option again: '))
    
mode = str()
if stopword_removal_option == 1:
    mode += 'stopWord-'
elif stopword_removal_option == 2:
    mode += 'noStopWord-'
    
print('\n* Word stemming options:')
print('\t 1. WordNet Lemmatizer')
print('\t 2. Porter Stemmer')
print('\t 3. No word stemming')
word_stemming_option = int(input('Please choose word stemming option: '))

while(word_stemming_option not in [1, 2, 3]):
    word_stemming_option = int(input('Please choose word stemming option again: '))

if word_stemming_option == 1:
    mode += 'lemmatize'
elif word_stemming_option == 2:
    mode += 'stem'
elif word_stemming_option == 3:
    mode += 'noStem'
  


print('\nMethod applied:')
# Liệt kê các thông tin của phương pháp được áp dụng

if stopword_removal_option == 1:   
    print('\t - Stopword removal: Stopword removal using NLTK stopword list')
elif stopword_removal_option == 2:
    print('\t - No stopword removal')
    
if word_stemming_option == 1:
    print('\t - Word stemming: WordNet Lemmatizer')
elif word_stemming_option == 2:
    print('\t - Word stemming: Porter Stemmer')
elif word_stemming_option == 3:
    print('\t - No word stemming')

print('\t - Term weighting: TF-IDF')

    
print('\n\n--------------------------- Initializing Vector Space Model ---------------------------\n')

start_time = time.process_time()
dict_of_docs, vector_of_docs, id_files = init_Vector_Space_Model(file_path_of_docs, mode)
end_time = time.process_time()

print('Initializing Finished')
print('Time To Build Model: ', end_time - start_time, " seconds")
print('Ready To Search')

# In ra những term nằm trong tất cả document

# for term in dict_of_docs.keys():
#     if dict_of_docs[term]['ndoc'] == 0:
#         print(term)


print('\n* Similarity measure options:')
print('\t 1. Cosine Similarity')
print('\t 2. Eulidean Distance')
ranking_function_option = int(input('Please choose Similarity Measure option: '))
simi_measure = str()

while(ranking_function_option not in [1, 2]):
    ranking_function_option = int(input('Please choose Similarity measure option again: '))

if ranking_function_option == 1:
    print('\t - Similarity Measure: Cosine Similarity')
    simi_measure = 'Cosine'
elif ranking_function_option == 2:
    print('\t - Similarity Measuren: Eclidean Distance')
    simi_measure = 'Euclid'


query_to_search = input('\nPlease type query to search: ')
top_k = int(input('\nPlease choose top k results to return: '))
searchDocumentWithQuery(dict_of_docs, vector_of_docs, id_files, query_to_search, top_k, mode, simi_measure)
