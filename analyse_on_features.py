import argparse

import pandas as pd
pd.set_option("display.max_colwidth", 200)
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

import matplotlib.pyplot as plt
import itertools
from minepy import MINE

import seaborn as sns




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

    df_proc = df.drop(['topic', 'content', 'topic_processed', 'content_processed'], axis=1)
    df_proc['label'] = df_proc['label'].map({'Real': 1, 'Fake': 0})

    if params.function == "analyse":
        feature_analyse(df_proc)
    elif params.function == "feature_selection_k_best":
        feature_selection_k_best(df_proc, params.features, params.k,)
    elif params.function == "feature_selection_mic":
        feature_selection_mic(df_proc, params.features)
    elif params.function == "logistic_regression":
        logistic_regression(df_proc, params.features)
    elif params.function == "random_forest":
        random_forest(df_proc, params.features)


def feature_analyse(df):
    print(df.isnull().sum())
    # Data Distribution
    plt.rc("font", size=14)
    sns.set(style="white")
    sns.set(style="whitegrid", color_codes=True)
    columns_list = []
    columns_list.append(['content_words_count', 'topic_words_count', 'total_words_count', 'total_sentence_count'])
    columns_list.append(
        ['sentiment_topic_pd', 'sentiment_content_pd', 'jaccard_coef_pd', 'sentence_avg_cosine_similarity_bert'])
    columns_list.append(['cosine_similarity_sklearn', 'cosine_similarity_sklearn_pd', 'cosine_similarity_spacy',
                         'cosine_similarity_spacy_pd'])
    columns_list.append(
        ['wmd_google_nonsplit_pd', 'wmd_google_pd', 'wmd_domain_pd', 'wmd_cyber_nonsplit_pd', 'wmd_cyber_pd'])
    columns_list.append(
        ['elmo_cosine_similarity_sklearn', 'elmo_cosine_similarity_sklearn', 'elmo_cosine_similarity_sklearn',
         'elmo_cosine_similarity_sklearn'])

    print('Data Distribution')
    for mcols in columns_list:
        range_plot(mcols, df, m_plot)

    print('Density Plot for Real and Fake Data')
    for mcols in columns_list:
        range_plot(mcols, df, m_plot_by_label)

def feature_selection_k_best(df, features, k):
    pd.set_option('display.float_format', lambda x: '%.6f' % x)

    all_feature_cols = ['sentiment_content_pd', 'jaccard_coef_pd', 'cosine_similarity_sklearn_pd',
                        'cosine_similarity_spacy_pd', 'wmd_google_pd', 'wmd_domain_pd', 'wmd_cyber_pd',
                        'sentence_avg_cosine_similarity_bert']

    feature_cols = all_feature_cols

    if features != "":
        feature_cols = features

    X = df[feature_cols]
    y = df['label']

    selectBest = SelectKBest(f_regression, k=k)
    fit = selectBest.fit(X, y)

    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    # concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']
    print('\n*** Feature selection - K Best ***')
    print('k = ', str(k))
    print(featureScores.nlargest(13, 'Score'))

def feature_selection_mic(df, features):
    pd.set_option('display.float_format', lambda x: '%.6f' % x)

    all_feature_cols = ['sentiment_content_pd', 'jaccard_coef_pd', 'cosine_similarity_sklearn_pd',
                        'cosine_similarity_spacy_pd', 'wmd_google_pd', 'wmd_domain_pd', 'wmd_cyber_pd',
                        'sentence_avg_cosine_similarity_bert']

    feature_cols = all_feature_cols

    if features != "":
        feature_cols = features

    X = df[feature_cols]
    y = df['label']

    m = MINE()

    scores = []
    for col in X.columns.values.tolist():
        m.compute_score(X[col], y)
        scores.append(m.mic())
        # print(col + "\t\t  : %2.3f" % m.mic())

    dfscores = pd.DataFrame(scores)
    dfcolumns = pd.DataFrame(X.columns)
    # concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']
    print('\n*** Feature selection - Mutual information and maximal information coefficient (MIC) ***')
    print(featureScores.nlargest(13, 'Score'))

def logistic_regression(df, features):
    sns.set(style="white")

    all_feature_cols = ['sentiment_content_pd', 'jaccard_coef_pd', 'cosine_similarity_sklearn_pd',
                        'cosine_similarity_spacy_pd', 'wmd_google_pd', 'wmd_domain_pd', 'wmd_cyber_pd',
                        'sentence_avg_cosine_similarity_bert']

    feature_cols = all_feature_cols

    if features != "":
        feature_cols = features

    X = df[feature_cols]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

    logreg = LogisticRegression(max_iter=10000)
    logreg.fit(X_train, y_train.values.reshape(-1))

    y_pred = logreg.predict(X_test)
    # Calculate accuracy of model over testing data
    print('\n*** Logistic Regression Classifier Model ***')
    print('Number of features: ' + str(len(feature_cols)))
    print('features list: ' + str(feature_cols))
    print('[Fake:0  Real:1]')

    accuracy(y_test, y_pred)
    print(classification_report(y_test, y_pred))
    dispConfusionMatrix(y_test, y_pred)

def random_forest(df, features):
    sns.set(style="white")

    all_feature_cols = ['sentiment_content_pd', 'jaccard_coef_pd', 'cosine_similarity_sklearn_pd',
                        'cosine_similarity_spacy_pd', 'wmd_google_pd', 'wmd_domain_pd', 'wmd_cyber_pd',
                        'sentence_avg_cosine_similarity_bert']

    feature_cols = all_feature_cols

    if features != "":
        feature_cols = features

    X = df[feature_cols]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

    model = RandomForestClassifier(n_estimators=50, criterion='entropy')
    model.fit(X_train, y_train.values.reshape(-1))

    y_pred = model.predict(X_test)
    # Calculate accuracy of model over testing data
    print('\n*** Random Forest Classifier Model ***')
    print('Number of features: ' + str(len(feature_cols)))
    print('features list: ' + str(feature_cols))
    print('[Fake:0  Real:1]')

    accuracy(y_test, y_pred)
    print(classification_report(y_test, y_pred))
    dispConfusionMatrix(y_test, y_pred)

def m_plot(col_name, input_df, i_row, i_col, i):
    x = 100 * i_row + 10 * i_col + i
    ax = plt.subplot(x)
    ax = input_df[col_name].hist(bins=15, color='teal', alpha=0.6)
    ax.set(xlabel=col_name)

def range_plot(input_cols, input_df, fun_plot):
    num_of_row = 1
    num_of_col = len(input_cols)
    plt.figure(figsize=(20,4), dpi=80)
    i = 1
    for col in input_cols:
        fun_plot(col, input_df, num_of_row, num_of_col, i)
        i+=1
    plt.show()

def m_plot_by_label(col_name, input_df, i_row, i_col, i):
    ax = plt.subplot(100 * i_row + 10 * i_col + i)
    ax = sns.kdeplot(input_df[col_name][input_df.label == 1], color="darkturquoise", shade=True)
    sns.kdeplot(input_df[col_name][input_df.label == 0], color="lightcoral", shade=True)
    plt.legend(['Real', 'Fake'])
    ax.set(xlabel=col_name)


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
    # plt.xticks(tick_marks, classes, rotation=45)
    plt.xticks(tick_marks, classes)
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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ####################################################################
    # Parse command line
    ####################################################################

    # Function
    parser.add_argument('--function', type=str, default='analyse', help='Select function from: analyse, '
                                                                        'feature_selection_k_best, '
                                                                        'feature_selection_mic, '
                                                                        'logistic_regression, '
                                                                        'random_forest')
    parser.add_argument('--k', type=int, default=2, help='For K Best model')

    # selected feature
    parser.add_argument('--features', type=str, default='', nargs='+', help='Selected feature')

    # Default Dirs
    parser.add_argument('--dataDir', type=str, default='./data/', help='intput folder')

    # input & output
    parser.add_argument('--input', type=str, default='dataset_long.xlsx', help='input file name')
    parser.add_argument('--output', type=str, default='dataset_long_with_feature.xlsx', help='output file name')

    m_args = parser.parse_args()
    main(m_args)










