import argparse

import pandas as pd
from gensim.models import KeyedVectors, Word2Vec
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

pd.set_option("display.max_colwidth", 200)
import re
from tqdm import tqdm
import string
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
import tensorflow_hub as hub
import tensorflow as tf
from textblob import TextBlob
from statistics import mean
import numpy as np
import pickle

en_stops = set(stopwords.words('english'))

import spacy
nlp = spacy.load('en_core_web_lg')

import neuralcoref
coref = neuralcoref.NeuralCoref(nlp.vocab)
nlp.add_pipe(coref, name='neuralcoref')

model = None
model_domain = None
sbert_model = None
elmo = None
model_cyber = None

def main(params):
    print(params)

    input_name = params.dataDir + params.input
    output_name = params.dataDir + params.output

    print("### Start Prepare Dataset")
    print("Input: " + input_name)
    print("Output: " + output_name + "\n")

    df = pd.read_excel(input_name, engine="openpyxl", sheet_name="Sheet", header=0)
    input_rows = df.shape[0]
    print("Read input file. Rows: " + str(input_rows))


    # pre-process
    tqdm.pandas(desc="Pre-processing topic")
    df['topic_processed'] = df['topic'].progress_map(preprocess_data)

    tqdm.pandas(desc="Pre-processing content")
    df['content_processed'] = df['content'].progress_map(preprocess_data)

    # generate words count column
    nltk.download('punkt')
    df['topic_words_count'] = df['topic'].str.split().str.len()
    df['content_words_count'] = df['content'].str.split().str.len()
    df['total_words_count'] = df['topic_words_count'] + df['content_words_count']

    df['total_sentence_count'] = df.progress_apply(cal_sentence_count, axis=1)

    # Sentiment Score
    tqdm.pandas(desc="Generate sentiment for topic")
    df['sentiment_topic_pd'] = df['topic_processed'].progress_map(cal_sentiment)

    tqdm.pandas(desc="Generate sentiment for content")
    df['sentiment_content_pd'] = df['content_processed'].progress_map(cal_sentiment)

    # Jaccard Coefficient
    tqdm.pandas(desc="Generate Jaccard Coefficient")
    df['jaccard_coef_pd'] = df.progress_apply(cal_jaccard_coef, axis=1)

    # Cosine Similarity using scikit-learn on Raw Data
    tqdm.pandas(desc="Generate Cosine Similarity using sklearn")
    df['cosine_similarity_sklearn'] = df.progress_apply(cal_cosine_similarity, axis=1)

    # Cosine Similarity using scikit-learn on Processed Data
    tqdm.pandas(desc="Generate Cosine Similarity using sklearn on processed data")
    df['cosine_similarity_sklearn_pd'] = df.progress_apply(cal_cosine_similarity_on_processed_data, axis=1)

    # Cosine Similarity using spaCy on Raw Data
    tqdm.pandas(desc="Generate Cosine Similarity using spaCy")
    df['cosine_similarity_spacy'] = df.progress_apply(cal_cosine_similarity_spacy, axis=1)

    # Cosine Similarity using spaCy on Processed Data
    tqdm.pandas(desc="Generate Cosine Similarity using spaCy on processed data")
    df['cosine_similarity_spacy_pd'] = df.progress_apply(cal_cosine_similarity_spacy_on_processed_data, axis=1)

    # word mover's distance on Processed Data using GoogleNews-vectors-negative300
    print("Loading GoogleNews-vectors-negative300.bin.gz ...")
    global model
    model = KeyedVectors.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin.gz', binary=True)

    tqdm.pandas(desc="Generate Word Mover's Distance on processed data using GoogleNews-vectors-negative300")
    df['wmd_google_nonsplit_pd'] = df.progress_apply(cal_wmd, axis=1)

    tqdm.pandas(desc="Generate Word Mover's Distance on processed data using GoogleNews-vectors-negative300")
    df['wmd_google_pd'] = df.progress_apply(cal_wmd_split, axis=1)

    # word mover's distance on Processed Data using Domain-word2vec
    print("Loading Domain-Word2vec.model.gz ...")
    global model_domain
    model_domain = Word2Vec.load('./data/CASIE/embeddings/Domain-Word2vec.model.gz')
    tqdm.pandas(desc="Generate Word Mover's Distance on processed data using Domain-Word2vec")
    df['wmd_domain_pd'] = df.progress_apply(cal_wmd_domain, axis=1)

    # word mover's distance on Processed Data using Cyber-word2vec
    print("Loading Cyber-Word2vec/1million.word2vec.model ...")
    global model_cyber
    model_cyber = Word2Vec.load('./model/1million.word2vec.model')
    tqdm.pandas(desc="Generate Word Mover's Distance on processed data using Cyber-Word2vec")
    df['wmd_cyber_nonsplit_pd'] = df.progress_apply(cal_wmd_cyber, axis=1)

    tqdm.pandas(desc="Generate Word Mover's Distance on processed data using Cyber-Word2vec")
    df['wmd_cyber_pd'] = df.progress_apply(cal_wmd_cyber_split, axis=1)

    # ELMO
    tf.compat.v1.disable_eager_execution()
    global elmo
    elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)

    list_df = [df[i:i + 2] for i in range(0, df.shape[0], 2)]
    elmo_topic = [elmo_vectors_word_level(x['topic']) for x in list_df]
    elmo_content = [elmo_vectors_word_level(x['content']) for x in list_df]
    elmo_topic_arr = np.concatenate(elmo_topic, axis=0)
    elmo_content_arr = np.concatenate(elmo_content, axis=0)

    arr_elmo = []
    for i in tqdm(range(df.shape[0])):
        topic = elmo_topic_arr[i]
        content = elmo_content_arr[i]

        data = [topic, content]

        cosine_similarity_matrix = cosine_similarity(data)

        arr_elmo.append(cosine_similarity_matrix[0][1])

    df['elmo_cosine_similarity_sklearn'] = arr_elmo

    # Sentence BERT
    print("Loading punkt & sentence-transformers/bert-base-nli-mean-tokens ...")
    nltk.download('punkt')
    # Full list https://github.com/UKPLab/sentence-transformers/blob/master/docs/pretrained-models/sts-models.md
    global sbert_model
    sbert_model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')
    tqdm.pandas(desc="Generate Cosine Similarity using SentenceBERT on raw data")
    df['sentence_avg_cosine_similarity_bert'] = df.progress_apply(cal_average_cosine_similarity_sentencebert, axis=1)

    # ELMO
    tqdm.pandas(desc="Generate Cosine Similarity using ELMO on raw data")
    df['sentence_avg_cosine_similarity_elmo'] = df.progress_apply(cal_average_cosine_similarity_sentence_elmo, axis=1)


    # Save
    save(df, output_name)
    print("\n### Finish generate features.")
    print("Input: " + input_name)
    print("Output: " + output_name)
    print("Columns: " + str(df.columns.tolist()))


def save(df, output_name):
    wb = Workbook()
    ws = wb.active
    for row in dataframe_to_rows(df, index=False, header=True):
        ws.append(row)

    wb.save(output_name)
    print("Saved to: " + output_name)

def remove_stopwords(Input):
    output_list = []
    for word in Input.lower().split():
        if word not in en_stops:
            output_list.append(word)
    output = " ".join(output_list)
    return output


# Function to replace pronouns
def replace_pronouns(text):
    doc = nlp(text)
    return doc._.coref_resolved


def preprocess_data(Input):
    proc = Input.replace("\n", " ")
    proc = proc.replace("\r", "")

    # replacing pronouns
    proc = replace_pronouns(proc)

    # remove stopwords
    proc = remove_stopwords(proc)

    # remove punctuation
    output = proc.translate(str.maketrans('', '', string.punctuation))

    return output


def isValiableString(str1):
    pattern = re.compile(r'[A-Za-z]')
    res = re.findall(pattern, str(str1))
    if len(res):
        return True
    else:
        return False


def cal_sentence_count(Input):
    topic = Input['topic']
    content = Input['content']

    proc = topic + content

    proc = content.replace("\n", " ")
    proc = proc.replace("\r", "")
    proc = TextBlob(proc)

    count = 0
    for sentence in proc.sentences:
        if not sentence.strip() or not isValiableString(sentence.strip()):
            continue
        count = count + 1

    return count


def cal_sentiment(Input):
    my_blob = TextBlob(str(Input))
    sentiment = str(my_blob.sentiment.polarity)

    return sentiment


def cal_jaccard_coef(Input):
    topic = Input['topic_processed']
    content = Input['content_processed']

    # List the unique words in a document
    words_topic = set(topic.lower().split())
    words_content = set(content.lower().split())

    intersection = words_topic.intersection(words_content)

    union = words_topic.union(words_content)

    jaccard_coefficient = float(len(intersection))/len(union)

    return jaccard_coefficient


def cal_cosine_similarity(Input):
    topic = Input['topic']
    content = Input['content']

    data = [topic, content]
    count_vectorizer = CountVectorizer()
    vector_matrix = count_vectorizer.fit_transform(data)

    cosine_similarity_matrix = cosine_similarity(vector_matrix)

    return cosine_similarity_matrix[0][1]


def cal_cosine_similarity_on_processed_data(Input):
    topic = Input['topic_processed']
    content = Input['content_processed']

    data = [topic, content]
    count_vectorizer = CountVectorizer()
    vector_matrix = count_vectorizer.fit_transform(data)

    cosine_similarity_matrix = cosine_similarity(vector_matrix)

    return cosine_similarity_matrix[0][1]


def cal_cosine_similarity_spacy(Input):
    topic = nlp(Input['topic'])
    content = nlp(Input['content'])

    return (content.similarity(topic))


def cal_cosine_similarity_spacy_on_processed_data(Input):
    topic = Input['topic_processed']
    content = Input['content_processed']

    topic = nlp(topic)
    content = nlp(content)

    return (content.similarity(topic))


def cal_wmd(Input):
    topic = Input['topic_processed']
    content = Input['content_processed']

    global model
    wmd = model.wmdistance(topic, content)

    return wmd


def cal_wmd_split(Input):
    topic = Input['topic_processed']
    content = Input['content_processed']

    wmd = model.wmdistance(topic.split(), content.split())

    return wmd

def cal_wmd_domain(Input):
    topic = Input['topic_processed']
    content = Input['content_processed']

    global model_domain
    wmd = model_domain.wv.wmdistance(topic.split(), content.split())

    return wmd


def cal_wmd_cyber(Input):
    topic = Input['topic_processed']
    content = Input['content_processed']

    global model_cyber
    wmd = model_cyber.wv.wmdistance(topic, content)

    return wmd

def cal_wmd_cyber_split(Input):
    topic = Input['topic_processed']
    content = Input['content_processed']

    global model_cyber
    wmd = model_cyber.wv.wmdistance(topic.split(), content.split())

    return wmd

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


def isValiableString(str1):
    pattern = re.compile(r'[A-Za-z]')
    res = re.findall(pattern, str(str1))
    if len(res):
        return True
    else:
        return False


def cal_average_cosine_similarity_sentencebert(Input):
    topic = Input['topic']
    content = Input['content']

    global sbert_model
    topic_vec = sbert_model.encode([topic])[0]

    proc = content.replace("\n", " ")
    proc = proc.replace("\r", "")
    proc = TextBlob(proc)

    cosine_similarity_list = []
    for sentence in proc.sentences:
        if not sentence.strip() or not isValiableString(sentence.strip()):
            continue
        sim = cosine(topic_vec, sbert_model.encode([sentence])[0])
        cosine_similarity_list.append(sim)

    output = mean(cosine_similarity_list)
    return output


def elmo_vectors(x):
  embeddings = elmo(x.split(), signature="default", as_dict=True)["elmo"]
  with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    sess.run(tf.compat.v1.tables_initializer())

    return sess.run(tf.reduce_mean(embeddings,1))


def cal_average_cosine_similarity_sentence_elmo(Input):
    topic = Input['topic']
    content = Input['content']

    topic_vec = elmo_vectors(topic)[0]

    proc = content.replace("\n", " ")
    proc = proc.replace("\r", "")
    proc = TextBlob(proc)

    cosine_similarity_list = []
    for sentence in proc.sentences:
        if not sentence.strip() or not isValiableString(sentence.strip()):
            continue
        sim = cosine(topic_vec, elmo_vectors(sentence)[0])
        cosine_similarity_list.append(sim)

    output = mean(cosine_similarity_list)
    return output

def elmo_vectors_word_level(x):
  embeddings = elmo(x.tolist(), signature="default", as_dict=True)["elmo"]
  with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    sess.run(tf.compat.v1.tables_initializer())

    return sess.run(tf.reduce_mean(embeddings, 1))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ####################################################################
    # Parse command line
    ####################################################################

    # Default Dirs
    parser.add_argument('--dataDir', type=str, default='./data/', help='intput Corpus folder')

    # input & output
    parser.add_argument('--input', type=str, default='dataset_long.xlsx', help='input file name')
    parser.add_argument('--output', type=str, default='dataset_long_with_feature.xlsx', help='output file name')

    m_args = parser.parse_args()
    main(m_args)
