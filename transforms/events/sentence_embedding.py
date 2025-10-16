from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import public_function as pf
import numpy as np
from tqdm import tqdm
# tfidf * word embedding


def read_text(path):
    text = []
    f = open(path, 'r')
    line = f.readline()
#     text.append(line[:-12])
    text.append(line.split('\t')[0])
    while line:
        line = f.readline()
#         text.append(line[:-12])
        text.append(line.split('\t')[0])
    f.close()
    # 去最后的空串
    return text[:-1]


def sentence_embedding(file_dict, train_path, test_path, save_path, service_num):
    data_dict = pf.load(file_dict)

    train_text = read_text(train_path)
    test_text = read_text(test_path)
    vectorizer = CountVectorizer(lowercase=False, token_pattern=r'(?u)\b\S\S+')  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
    # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    vec_train = vectorizer.fit_transform(train_text)
    tfidf_train = transformer.fit_transform(vec_train)
    # 预测
    vec_test = vectorizer.transform(test_text)
    tfidf_test = transformer.transform(vec_test)

    weight_train = tfidf_train.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    weight_test = tfidf_test.toarray()
#     weight_test = tfidf_test.toarray()[-len(test_text): ]

    word = vectorizer.get_feature_names_out()  # 获取词袋模型中的所有词语
    word_dict = {word[i]: i for i in range(len(word))}

    
#     print('dict(vectorizer words) - dict(fasttext words) = ', set(word_dict.keys()-set(data_dict.keys())))
#     assert len(word_dict) + 1 == len(data_dict)
    print('len vectorizer words:', len(word_dict))
    print('len fasttext words:', len(data_dict))
    print('dict(fasttext words) - dict(vectorizer words) = ', set(data_dict.keys()-set(word_dict.keys())))
    print('dict(vectorizer words) - dict(fasttext words) = ', set(word_dict.keys()-set(data_dict.keys())))

    train_embedding = tfidf_word_embedding(weight_train, data_dict, train_text, word_dict, service_num)
    test_embedding = tfidf_word_embedding(weight_test, data_dict, test_text, word_dict, service_num)

    train_embedding.extend(test_embedding)

    print('sentence_embedding shape:', f'{len(train_embedding)} * {len(train_embedding[0])} * {len(train_embedding[0][0])}')
    pf.save(save_path, train_embedding)


# --- REPLACE this entire function ---
def tfidf_word_embedding(tfidf_matrix, data_dict, texts, word_dict, service_num):
    length = len(data_dict[list(data_dict.keys())[0]])
    
    case_embedding = []
    sentence_embedding = []
    
    # NEW: Wrap the main loop with tqdm for a real-time progress bar
    # This will show the progress as it processes each line of your text files.
    for count, text in enumerate(tqdm(texts, desc="Generating Embeddings")):
        temp = np.array([0] * length, 'float32')
        
        if text != '':
            words = list(set(text.split(' ')))
            sparse_row = tfidf_matrix.getrow(count)
            
            for word in words:
                if word in word_dict:
                    word_index = word_dict[word]
                    weight = sparse_row[0, word_index]
                    temp = temp + weight * np.array(data_dict[word])

        case_embedding.append(temp)
        if (count + 1) % service_num == 0:
            sentence_embedding.append(case_embedding)
            case_embedding = []
            
    return sentence_embedding
# --- REPLACE this entire function as well ---
def sentence_embedding(file_dict, train_path, test_path, save_path, service_num):
    data_dict = pf.load(file_dict)

    train_text = read_text(train_path)
    test_text = read_text(test_path)
    vectorizer = CountVectorizer(lowercase=False, token_pattern=r'(?u)\b\S\S+')
    transformer = TfidfTransformer()
    
    vec_train = vectorizer.fit_transform(train_text)
    tfidf_train = transformer.fit_transform(vec_train) # This is a sparse matrix

    vec_test = vectorizer.transform(test_text)
    tfidf_test = transformer.transform(vec_test) # This is also a sparse matrix

    # THE FIX: We have REMOVED the .toarray() calls that caused the crash.
    # weight_train = tfidf_train.toarray() <--- REMOVED
    # weight_test = tfidf_test.toarray() <--- REMOVED

    word = vectorizer.get_feature_names_out()
    word_dict = {word[i]: i for i in range(len(word))}
    
    print('len vectorizer words:', len(word_dict))
    print('len fasttext words:', len(data_dict))
    print('dict(fasttext words) - dict(vectorizer words) = ', set(data_dict.keys()-set(word_dict.keys())))
    print('dict(vectorizer words) - dict(fasttext words) = ', set(word_dict.keys()-set(data_dict.keys())))

    # We now pass the efficient sparse matrices directly to the new function
    train_embedding = tfidf_word_embedding(tfidf_train, data_dict, train_text, word_dict, service_num)
    test_embedding = tfidf_word_embedding(tfidf_test, data_dict, test_text, word_dict, service_num)

    # MEMORY FIX: Use chunked saving instead of extending in memory
    print('train_embedding shape:', f'{len(train_embedding)} * {len(train_embedding[0]) if train_embedding else 0} * {len(train_embedding[0][0]) if train_embedding and train_embedding[0] else 0}')
    print('test_embedding shape:', f'{len(test_embedding)} * {len(test_embedding[0]) if test_embedding else 0} * {len(test_embedding[0][0]) if test_embedding and test_embedding[0] else 0}')
    
    # Combine embeddings list without extending (more memory efficient)
    all_embeddings = train_embedding + test_embedding
    total_shape = f'{len(all_embeddings)} * {len(all_embeddings[0]) if all_embeddings else 0} * {len(all_embeddings[0][0]) if all_embeddings and all_embeddings[0] else 0}'
    print('combined_embedding shape:', total_shape)
    
    # Use chunked save to avoid memory error
    pf.save_chunked(save_path, all_embeddings, chunk_size=500)
    
    # Clear memory immediately
    del all_embeddings, train_embedding, test_embedding


