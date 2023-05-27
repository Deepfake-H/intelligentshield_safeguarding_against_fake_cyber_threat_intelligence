import argparse

import pandas as pd
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

pd.set_option("display.max_colwidth", 200)
from tqdm import tqdm

import matplotlib.pyplot as plt

from transformers import RobertaForSequenceClassification, RobertaTokenizer
import tensorflow_hub as hub
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np

import spacy
import html
import emoji
from transformers import pipeline
import re
import torch
torch.cuda.is_available()

from transformers import GPT2LMHeadModel, GPT2Tokenizer, BertTokenizer, BertForMaskedLM

import itertools
plt.rc("font", size=14)
plt.figure(figsize=(5,4), dpi=80)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
roberta_model = None
roberta_tokenizer = None
lm_gpt_2 = None
dl_pipe = None
elmo_model = None

class AbstractLanguageChecker():
    """
    Abstract Class that defines the Backend API of GLTR.

    To extend the GLTR interface, you need to inherit this and
    fill in the defined functions.
    """

    def __init__(self):
        '''
        In the subclass, you need to load all necessary components
        for the other functions.
        Typically, this will comprise a tokenizer and a model.
        '''
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def check_probabilities(self, in_text, topk=40):
        '''
        Function that GLTR interacts with to check the probabilities of words

        Params:
        - in_text: str -- The text that you want to check
        - topk: int -- Your desired truncation of the head of the distribution

        Output:
        - payload: dict -- The wrapper for results in this function, described below

        Payload values
        ==============
        bpe_strings: list of str -- Each individual token in the text
        real_topk: list of tuples -- (ranking, prob) of each token
        pred_topk: list of list of tuple -- (word, prob) for all topk
        '''
        raise NotImplementedError

    def postprocess(self, token):
        """
        clean up the tokens from any special chars and encode
        leading space by UTF-8 code '\u0120', linebreak with UTF-8 code 266 '\u010A'
        :param token:  str -- raw token text
        :return: str -- cleaned and re-encoded token text
        """
        raise NotImplementedError

class LM(AbstractLanguageChecker):
    def __init__(self, model_name_or_path="gpt2"):
        super(LM, self).__init__()
        self.enc = GPT2Tokenizer.from_pretrained(model_name_or_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
        self.model.to(self.device)
        self.model.eval()
        self.start_token = '<|endoftext|>'
        print("Loaded GPT-2 model!")

    def check_probabilities(self, in_text, topk=40):
        # Process input
        start_t = torch.full((1, 1),
                             self.enc.encoder[self.start_token],
                             device=self.device,
                             dtype=torch.long)

        context = self.enc.encode(in_text)

        # index out of range
        context = context[:min(1022, len(context))]

        context = torch.tensor(context,
                               device=self.device,
                               dtype=torch.long).unsqueeze(0)
        context = torch.cat([start_t, context], dim=1)
        # Forward through the model
        logits, _ = self.model(context)

        # construct target and pred
        yhat = torch.softmax(logits[0, :-1], dim=-1)
        y = context[0, 1:]
        # Sort the predictions for each timestep
        sorted_preds = np.argsort(-yhat.data.cpu().numpy())
        # [(pos, prob), ...]
        real_topk_pos = list(
            [int(np.where(sorted_preds[i] == y[i].item())[0][0])
             for i in range(y.shape[0])])
        real_topk_probs = yhat[np.arange(
            0, y.shape[0], 1), y].data.cpu().numpy().tolist()
        real_topk_probs = list(map(lambda x: round(x, 5), real_topk_probs))

        real_topk = list(zip(real_topk_pos, real_topk_probs))
        # [str, str, ...]
        bpe_strings = [self.enc.decoder[s.item()] for s in context[0]]

        bpe_strings = [self.postprocess(s) for s in bpe_strings]

        # [[(pos, prob), ...], [(pos, prob), ..], ...]
        pred_topk = [
            list(zip([self.enc.decoder[p] for p in sorted_preds[i][:topk]],
                     list(map(lambda x: round(x, 5),
                              yhat[i][sorted_preds[i][
                                      :topk]].data.cpu().numpy().tolist()))))
            for i in range(y.shape[0])]

        pred_topk = [[(self.postprocess(t[0]), t[1]) for t in pred] for pred in pred_topk]
        payload = {'bpe_strings': bpe_strings,
                   'real_topk': real_topk,
                   'pred_topk': pred_topk}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return payload

    def sample_unconditional(self, length=100, topk=5, temperature=1.0):
        context = torch.full((1, 1),
                             self.enc.encoder[self.start_token],
                             device=self.device,
                             dtype=torch.long)
        prev = context
        output = context
        past = None
        # Forward through the model
        with torch.no_grad():
            for i in range(length):
                logits, past = self.model(prev, past=past)
                logits = logits[:, -1, :] / temperature
                # Filter predictions to topk and softmax
                probs = torch.softmax(top_k_logits(logits, k=topk),
                                      dim=-1)
                # Sample
                prev = torch.multinomial(probs, num_samples=1)
                # Construct output
                output = torch.cat((output, prev), dim=1)

        output_text = self.enc.decode(output[0].tolist())
        return output_text

    def postprocess(self, token):
        with_space = False
        with_break = False
        if token.startswith('Ġ'):
            with_space = True
            token = token[1:]
            # print(token)
        elif token.startswith('â'):
            token = ' '
        elif token.startswith('Ċ'):
            token = ' '
            with_break = True

        token = '-' if token.startswith('â') else token
        token = '“' if token.startswith('ľ') else token
        token = '”' if token.startswith('Ŀ') else token
        token = "'" if token.startswith('Ļ') else token

        if with_space:
            token = '\u0120' + token
        if with_break:
            token = '\u010A' + token

        return token


class BERTLM(AbstractLanguageChecker):
    def __init__(self, model_name_or_path="bert-base-cased"):
        super(BERTLM, self).__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(
            model_name_or_path,
            do_lower_case=False)
        self.model = BertForMaskedLM.from_pretrained(
            model_name_or_path)
        self.model.to(self.device)
        self.model.eval()
        # BERT-specific symbols
        self.mask_tok = self.tokenizer.convert_tokens_to_ids(["[MASK]"])[0]
        self.pad = self.tokenizer.convert_tokens_to_ids(["[PAD]"])[0]
        print("Loaded BERT model!")

    def check_probabilities(self, in_text, topk=40, max_context=20,
                            batch_size=20):
        '''
        Same behavior as GPT-2
        Extra param: max_context controls how many words should be
        fed in left and right
        Speeds up inference since BERT requires prediction word by word
        '''
        in_text = "[CLS] " + in_text + " [SEP]"
        tokenized_text = self.tokenizer.tokenize(in_text)
        # Construct target
        y_toks = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        # Only use sentence A embedding here since we have non-separable seq's
        segments_ids = [0] * len(y_toks)
        y = torch.tensor([y_toks]).to(self.device)
        segments_tensor = torch.tensor([segments_ids]).to(self.device)

        # TODO batching...
        # Create batches of (x,y)
        input_batches = []
        target_batches = []
        for min_ix in range(0, len(y_toks), batch_size):
            max_ix = min(min_ix + batch_size, len(y_toks) - 1)
            cur_input_batch = []
            cur_target_batch = []
            # Construct each batch
            for running_ix in range(max_ix - min_ix):
                tokens_tensor = y.clone()
                mask_index = min_ix + running_ix
                tokens_tensor[0, mask_index + 1] = self.mask_tok

                # Reduce computational complexity by subsetting
                min_index = max(0, mask_index - max_context)
                max_index = min(tokens_tensor.shape[1] - 1,
                                mask_index + max_context + 1)

                tokens_tensor = tokens_tensor[:, min_index:max_index]
                # Add padding
                needed_padding = max_context * 2 + 1 - tokens_tensor.shape[1]
                if min_index == 0 and max_index == y.shape[1] - 1:
                    # Only when input is shorter than max_context
                    left_needed = (max_context) - mask_index
                    right_needed = needed_padding - left_needed
                    p = torch.nn.ConstantPad1d((left_needed, right_needed),
                                               self.pad)
                    tokens_tensor = p(tokens_tensor)
                elif min_index == 0:
                    p = torch.nn.ConstantPad1d((needed_padding, 0), self.pad)
                    tokens_tensor = p(tokens_tensor)
                elif max_index == y.shape[1] - 1:
                    p = torch.nn.ConstantPad1d((0, needed_padding), self.pad)
                    tokens_tensor = p(tokens_tensor)

                cur_input_batch.append(tokens_tensor)
                cur_target_batch.append(y[:, mask_index + 1])
                # new_segments = segments_tensor[:, min_index:max_index]
            cur_input_batch = torch.cat(cur_input_batch, dim=0)
            cur_target_batch = torch.cat(cur_target_batch, dim=0)
            input_batches.append(cur_input_batch)
            target_batches.append(cur_target_batch)

        real_topk = []
        pred_topk = []

        with torch.no_grad():
            for src, tgt in zip(input_batches, target_batches):
                # Compute one batch of inputs
                # By construction, MASK is always the middle
                logits = self.model(src, torch.zeros_like(src))[:,
                         max_context + 1]
                yhat = torch.softmax(logits, dim=-1)

                sorted_preds = np.argsort(-yhat.data.cpu().numpy())
                # TODO: compare with batch of tgt

                # [(pos, prob), ...]
                real_topk_pos = list(
                    [int(np.where(sorted_preds[i] == tgt[i].item())[0][0])
                     for i in range(yhat.shape[0])])
                real_topk_probs = yhat[np.arange(
                    0, yhat.shape[0], 1), tgt].data.cpu().numpy().tolist()
                real_topk.extend(list(zip(real_topk_pos, real_topk_probs)))

                # # [[(pos, prob), ...], [(pos, prob), ..], ...]
                pred_topk.extend([list(zip(self.tokenizer.convert_ids_to_tokens(
                    sorted_preds[i][:topk]),
                    yhat[i][sorted_preds[i][
                            :topk]].data.cpu().numpy().tolist()))
                    for i in range(yhat.shape[0])])

        bpe_strings = [self.postprocess(s) for s in tokenized_text]
        pred_topk = [[(self.postprocess(t[0]), t[1]) for t in pred] for pred in pred_topk]
        payload = {'bpe_strings': bpe_strings,
                   'real_topk': real_topk,
                   'pred_topk': pred_topk}
        return payload

    def postprocess(self, token):

        with_space = True
        with_break = token == '[SEP]'
        if token.startswith('##'):
            with_space = False
            token = token[2:]

        if with_space:
            token = '\u0120' + token
        if with_break:
            token = '\u010A' + token
        #
        # # print ('....', token)
        return token

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
    print("Columns: " + str(df.columns.tolist()))


    df['text'] = df['topic_processed'] + ' ' + df['content_processed']
    df_proc = df[['text', 'label']]

    if params.function == "analyse":
        text_analyse(df_proc)
    elif params.function == "passive_aggressive":
        passive_aggressive(df_proc)
    elif params.function == "logic_regression":
        logic_regression(df_proc)
    elif params.function == "decision_tree":
        decision_tree(df_proc)
    elif params.function == "random_forest":
        random_forest(df_proc)
    elif params.function == "roberta":
        roberta(df)
    elif params.function == "gltr":
        gltr(df, output_name)
    elif params.function == "deep_learning":
        deep_learning(df, output_name)
    elif params.function == "elmo":
        elmo(df_proc)


def text_analyse(df):
    showCloud(df, 'Real')
    showCloud(df, 'Fake')

def passive_aggressive(df):
    x_train, x_test, y_train, y_test = prepareData(df)

    tfidf_vectorizer = TfidfVectorizer(stop_words = 'english', max_df = 0.7)
    tfidf_train = tfidf_vectorizer.fit_transform(x_train)
    tfidf_test = tfidf_vectorizer.transform(x_test)

    model = PassiveAggressiveClassifier(max_iter = 50)

    model.fit(tfidf_train, y_train)
    predicted = model.predict(tfidf_test)
    dispReport('Passive Aggressive Classifier', 'Test', y_test, predicted)


def logic_regression(df):
    x_train, x_test, y_train, y_test = prepareData(df)
    # Vectorising and applying TF-IDF
    pipe = Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('model', LogisticRegression())])

    # Fitting the model
    model = pipe.fit(x_train, y_train)

    # Accuracy
    predicted = model.predict(x_test)
    dispReport('Logistic regression', 'Test', y_test, predicted)

def decision_tree(df):
    x_train, x_test, y_train, y_test = prepareData(df)
    # Vectorising and applying TF-IDF
    pipe = Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('model', DecisionTreeClassifier(criterion= 'entropy',
                                                                    max_depth=20,
                                                                    splitter='best',
                                                                    random_state=42))])

    model = pipe.fit(x_train, y_train)
    predicted = model.predict(x_test)
    dispReport('Decision Tree Classifier', 'Test', y_test, predicted)

def random_forest(df):
    x_train, x_test, y_train, y_test = prepareData(df)
    # Vectorising and applying TF-IDF
    pipe = Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('model', RandomForestClassifier(n_estimators=50,
                                                                    criterion='entropy'))])

    model = pipe.fit(x_train, y_train)
    predicted = model.predict(x_test)
    dispReport('Random Forest Classifier', 'Test', y_test, predicted)

def roberta(df):
    global roberta_model, roberta_tokenizer
    checkpoint = './model/detector-large.pt'

    print(f'Loading checkpoint from {checkpoint}')
    data = torch.load(checkpoint, map_location='cpu')

    model_name = 'roberta-large' if data['args']['large'] else 'roberta-base'
    roberta_model = RobertaForSequenceClassification.from_pretrained(model_name)
    roberta_tokenizer = RobertaTokenizer.from_pretrained(model_name)

    roberta_model.load_state_dict(data['model_state_dict'])
    roberta_model.eval()
    roberta_model = roberta_model.to(device)

    tqdm.pandas(desc="Analyse text using RoBERTa Model")
    df[['roberta_predict', 'roberta_real', 'roberta_fake']] = df.progress_apply(cal_roberta, axis=1, result_type='expand')

    dispReport('RoBERTa Classifier', 'Full', df['label'], df['roberta_predict'])

def cal_roberta(input_data):
    text = input_data['content']

    global roberta_model, roberta_tokenizer
    tokens = roberta_tokenizer.encode(text)
    all_tokens = len(tokens)
    tokens = tokens[:min(510, roberta_tokenizer.max_len - 2)]
    used_tokens = len(tokens)

    tokens = torch.tensor([roberta_tokenizer.bos_token_id] + tokens + [roberta_tokenizer.eos_token_id]).unsqueeze(0)
    mask = torch.ones_like(tokens)

    with torch.no_grad():
        logits = roberta_model(tokens.to(device), attention_mask=mask.to(device))[0]
        probs = logits.softmax(dim=-1)

    fake, real = probs.detach().cpu().flatten().numpy().tolist()

    predict = 'Real' if real > 0.5 else 'Fake'

    return predict, real, fake

def gltr(df, output_name):
    global lm_gpt_2
    lm_gpt_2 = LM()

    tqdm.pandas(desc="Analyse text using GLTR(GPT-2) Model")
    df[['gltr_gpt2_count_less_than_10', 'gltr_gpt2_count_10_to_100', 'gltr_gpt2_count_100_to_1000',
        'gltr_gpt2_count_more_than_1000', 'gltr_gpt2_count_total']] = df.progress_apply(cal_gltr_gpt2, axis=1,
                                                                                        result_type='expand')

    df['gltr_gpt2_count_less_than_10_percentage'] = df['gltr_gpt2_count_less_than_10'] / df['gltr_gpt2_count_total']
    df['gltr_gpt2_count_10_to_100_percentage'] = df['gltr_gpt2_count_10_to_100'] / df['gltr_gpt2_count_total']
    df['gltr_gpt2_count_100_to_1000_percentage'] = df['gltr_gpt2_count_100_to_1000'] / df['gltr_gpt2_count_total']
    df['gltr_gpt2_count_more_than_1000_percentage'] = df['gltr_gpt2_count_more_than_1000'] / df['gltr_gpt2_count_total']

    save(df, output_name)

def save(df, output_name):
    wb = Workbook()
    ws = wb.active
    for row in dataframe_to_rows(df, index=False, header=True):
        ws.append(row)

    wb.save(output_name)

    print("Columns: " + str(df.columns.tolist()))
    print("*** Saved to: " + output_name)

def deep_learning(df, output_name):
    global dl_pipe
    dl_pipe = pipeline("sentiment-analysis", model='bvrau/covid-twitter-bert-v2-struth')

    tqdm.pandas(desc="Analyse text using DeepLearning Model")
    df[['DL_ctb_predict', 'DL_ctb_score']] = df.progress_apply(cal_dl, axis=1, result_type='expand')

    save(df, output_name)

    print('\n*** DeepLearning Model ***')
    accuracy(df['label'], df['DL_ctb_predict'])
    print(classification_report(df['label'], df['DL_ctb_predict']))
    dispConfusionMatrix(df['label'], df['DL_ctb_predict'])

def DeepLearning(input_data):
    global dl_pipe
    result = dl_pipe(input_data)
    resultdict = result[0]
    label = resultdict['label']
    score = resultdict['score']

    return str(label).title(), score

def cal_dl(input_data):
    text = input_data['content']

    dl_preproccessed_input = preprocmain(text)
    label, score = DeepLearning(dl_preproccessed_input)

    return label, score


def elmo(df):
    x_train, x_test, y_train, y_test = prepareData(df)
    global elmo_model
    elmo_model = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)

    elmo_steps = 100
    list_train = [x_train[i:i + elmo_steps] for i in range(0, x_train.shape[0], elmo_steps)]
    list_test = [x_test[i:i + elmo_steps] for i in range(0, x_test.shape[0], elmo_steps)]

    elmo_train = [elmo_vectors(x) for x in list_train]
    elmo_test = [elmo_vectors(x) for x in list_test]

    elmo_train_new = np.concatenate(elmo_train, axis=0)
    elmo_test_new = np.concatenate(elmo_test, axis=0)

    xtrain, xvalid, ytrain, yvalid = train_test_split(elmo_train_new, y_train, test_size=0.2, random_state=42)

    lreg = LogisticRegression()
    lreg.fit(xtrain, ytrain)

    preds_valid = lreg.predict(xvalid)
    dispReport('Logistic Regression', 'Validation', yvalid, preds_valid)

    preds_test = lreg.predict(elmo_test_new)
    dispReport('Logistic Regression', 'Test', y_test, preds_test)


def elmo_vectors(x):
    global elmo_model
    embeddings = elmo_model(x.tolist(), signature="default", as_dict=True)["elmo"]
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.tables_initializer())
        return sess.run(tf.reduce_mean(embeddings,1))

def cal_gltr_gpt2(input_data):
    text = input_data['content']
    global lm_gpt_2
    payload = lm_gpt_2.check_probabilities(text, topk=5)

    count_1 = sum(1 for i in payload['real_topk'] if i[0] <= 10)
    count_2 = sum(1 for i in payload['real_topk'] if (i[0] > 10 and i[0] <= 100))
    count_3 = sum(1 for i in payload['real_topk'] if (i[0] > 100 and i[0] <= 1000))
    count_4 = sum(1 for i in payload['real_topk'] if (i[0] > 1000))
    total = len(payload['real_topk'])

    return count_1, count_2, count_3, count_4, total
def showCloud(dataset, label):
    label_data = dataset[dataset["label"] == label]
    all_words = ' '.join([text for text in label_data.text])

    wordcloud = WordCloud(width=800,
                          height=500,
                          max_font_size=110,
                          collocations=False).generate(all_words)

    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Word Cloud for (label:" + label + ")")
    plt.show()


def accuracy(y_test, predicted):
    score = accuracy_score(y_test, predicted)
    print("Accuracy: ", round(score * 100, 2), "%")


###  Confustion matrix  ###
def plotConfusionMatrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalised confusion matrix')
    else:
        print('Confusion matrix, without normalisation')

    thresh = cm.max() / 1.1
    # print(str(thresh + 'thresh')
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


###   Display Confustion Matrix  ###
def dispConfusionMatrix(y_test, predicted):
    cm = metrics.confusion_matrix(y_test, predicted)
    print('Confusion Matrix:')
    print('TP: ' + str(cm[0][0]) + '\t\tFP: ' + str(cm[0][1]))
    print('FN: ' + str(cm[1][0]) + '\t\tTN: ' + str(cm[1][1]))
    plotConfusionMatrix(cm, classes=['Fake', 'Real'])

def dispReport(ModelName, DatasetName, y_true, y_pred):
    print('\n*** ' + ModelName + ' Model - ' + DatasetName + ' Dataset ***')
    accuracy(y_true, y_pred)
    print(classification_report(y_true, y_pred))
    dispConfusionMatrix(y_true, y_pred)

def prepareData(dataset):
    # Divide data for training and testing (currently 80:20 - train:test)
    x_train, x_test, y_train, y_test = train_test_split(dataset['text'], dataset['label'], test_size=0.2,
                                                        random_state=42)

    return x_train, x_test, y_train, y_test

def top_k_logits(logits, k):
    '''
    Filters logits to only the top k choices
    from https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_gpt2.py
    '''
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1]
    return torch.where(logits < min_values,
                       torch.ones_like(logits, dtype=logits.dtype) * -1e10,
                       logits)


def twitter_cleaning(input):
    """ This function prepares the input for preprocessing by removing Twitter specific
        data such as:
            -URLs
            -Twitter handles
            -Hashtags
            -Twitter URL, User and Retweet tags
            -Repeated characters

        This function also converts emojis into text, for example:
            -😂 becomes :face_with_tears_of_joy:
        ** Parameters **
        input: a str containing the body of a Tweet

        ** Returns **
        remqmarks: A string object containing the original Tweet processed
    """
    ## Removing URLs
    remurl = re.sub('http://\S+|https://\S+', '', input)

    ## Removing Twitter Handles
    remhand = re.sub('@[^\s]+', '', remurl)

    ## Removing Hashtags (covid)
    remhash1 = remhand.replace('#covid', 'covid')

    ## Removing Hashtags (covid19)
    remhash2 = remhash1.replace('#covid19', 'covid19')

    ## Removing Hashtags (coronavirus)
    remhash3 = remhash2.replace('#coronavirus', 'coronavirus')

    ## Removing Hashtags (general)
    remhash4 = re.sub('#[^\s]+', '', remhash3)

    ## Removing twitterurl tags
    remtwitterurl = remhash4.replace('twitterurl', '')

    ## Removing twitteruser tags
    remtwitteruser = remtwitterurl.replace('twitteruser', '')

    ## Removing rt tags
    remrt = remtwitteruser.replace('rt', '')

    ## Switching Emojis to their descriptions
    rememoji = emoji.demojize(remrt)

    ## Removing '???' occurences
    remqmarks = rememoji.replace('???', '')
    remqmarks = remqmarks.replace('??', '')

    return remqmarks


def general_cleanup(input):
    """ This function prepares the input for preprocessing by tidying general
        data such as:
            -removing non-ASCII characters
            -removing HTML entities
            -removing additional spaces created during preprocessing preparation
        ** Parameters **
        input: a str containing the body of a Tweet
        ** Returns **
        A string object containing the original Tweet processed
    """
    ## Removing non-ASCII characters
    nonascii = input.encode("ascii", "ignore")
    remnonascii = nonascii.decode()

    # Removing HTML entities
    remhtml = html.unescape(remnonascii)

    # Cleaning up double spaces created in removal
    remspc = re.sub(' +', ' ', remhtml)

    return remspc


def spacy_preproc(input):
    """ This function enables the preprocessing of the Tweet using the spaCy libraries.
        First, the spaCy language model is loaded into the program, before tokenization occurs.
        This exchanges the string values into tokens which can be used again by spaCy.
        Excess spaces are then removed and colons removed. The colons appear from the conversion
        of emojis to text representation. Finally, the tokens are output as str objects to a new list.
        ** Parameters **
        input: a str containing the body of a Tweet
        ** Returns **
        A string object of the processed original Tweet
    """
    # loading the basic English library for preprocessing tasks
    nlp = spacy.load('en_core_web_sm')
    stopword_list = nlp.Defaults.stop_words

    # sets the text being input for preprocessing
    text_test = nlp(input)

    ## Tokenisation
    # creates a list to hold the results of the tokenization
    token_list = []

    # appends the processed tokens to the list, removing any spaces or colons
    for token in text_test:
        if str(token) != ' ':
            token_list.append(token)

    token_list = token_list[:min(350, len(token_list) - 2)]
    ## removes any colons which enter the token list
    for tokens in token_list:
        if str(tokens) == ':':
            token_list.remove(tokens)

    # # list for the removed stopwords
    # stopwords_rem = []

    # # removes stopwords from the text
    # for token in token_list:
    #     if str(token) not in stopword_list:
    #         stopwords_rem.append(token)

    # for token in stopwords_rem:
    #     if str(token) == '&':
    #         stopwords_rem.remove(token)

    # # creating a list of lemmatised tokens from the sentence
    # lemma_text = []

    # # adds lemmatised version of all words (if applicable) into a final list, ready for the model
    # for word in token_list:
    #     lemma_text.append(word.lemma_)

    # # creating a list of lemmatised tokens from the sentence
    final_text = []

    # adds coverts tokens back into str, then adds into a final list, ready for the model
    for word in token_list:
        final_text.append(str(word))

    preproc_str = ' '.join(final_text)

    return preproc_str


def preprocmain(input_text):
    """The main function for this program. Controls the I/O and flow of program execution

        ** Parameters **
        input_text: a str containing the body of a Tweet
        ** Returns **
        spacy_cleaned: A string object of the preprocessed original Tweet
    """
    twitter_cleaned = twitter_cleaning(input_text)
    general_cleaned = general_cleanup(twitter_cleaned)
    spacy_cleaned = spacy_preproc(general_cleaned)
    return spacy_cleaned

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ####################################################################
    # Parse command line
    ####################################################################

    # Function
    parser.add_argument('--function', type=str, default='analyse', help='Select function from: analyse, '
                                                                        'passive_aggressive, logic_regression, '
                                                                        'decision_tree, random_forest, roberta, gltr, '
                                                                        'deep_learning')

    # Default Dirs
    parser.add_argument('--dataDir', type=str, default='./data/', help='intput folder')

    # input & output
    parser.add_argument('--input', type=str, default='dataset_long.xlsx', help='input file name')
    parser.add_argument('--output', type=str, default='dataset_long_with_feature.xlsx', help='output file name')

    m_args = parser.parse_args()
    main(m_args)






