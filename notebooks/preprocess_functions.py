import re
from nltk.corpus import stopwords
import string
import pandas as pd
from bs4 import BeautifulSoup
import spacy


def lemmetize(doc):
    nlp = spacy.load('en')
    sents = nlp(doc)
    doc_new = []
    # https://spacy.io/usage/linguistic-features
    for token in sents:
        if token.pos_ not in ['AUX', 'DET', 'NUM', 'PUNC', 'PRON', 'SYM']:
            if token.ent_type_ not in ['PERSON', 'GPE', 'ORG', 'NORP']:
                doc_new.append(token.lemma_)

    return " ".join(doc_new)


def lower_case(df):
    df["text"] = [text.lower() for text in df["text"]]
    return df


def remove_special_chars(df, col='text'):
    """

    :param df: input df with 'text' column
    :return: df with special chars removed from text column
    """
    # Remove url
    df[col] = [re.sub(r"http\S+", " ", text) for text in df[col]]

    # Remove special chars and numbers
    df[col] = df[col].str.replace("[^a-zA-Z#]", " ")

    df[col] = [re.sub(r"\b[A-Z\.]{2,}s?\b", "", text) for text in df[col]]
    df[col] = [re.sub('\S*@\S*\s?', ' ', text) for text in df[col]]
    # df["len"] = df["text"].str.len()
    return df


def remove_stop_words(col):
    """

    :param col: The pd.series containing text
    :param stop_words: the stopwords that need to be removed
    :return:
    """
    stop_words = set(stopwords.words('english'))
    punctuation = list(string.punctuation)
    stop_words.update(punctuation)
    stop_words.extend(['and', 'here', 'there', 'to', 'at'])
    col_new = []
    for text in col:
        list_of_words = text.split()
        new_text = " ".join([i for i in list_of_words if i not in stop_words])
        col_new.append(new_text)
    return col_new


# Removing the html strips
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


# Removing the square brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)


# Removing the noisy text
def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text

##############################################################


dataset = "bank"

# bank
if dataset == "bank":
    original_path = "BankReviews.xlsx"
    path = "../datasets/bank_dataset.csv"
    df = pd.read_excel(original_path, encoding='windows-1252', sheet_name='BankReviews')
    print(df.shape, df.label.value_counts())
    df.label.replace(1, 0, inplace=True)
    df.label.replace(5, 1, inplace=True)
    df.dropna(inplace=True)
    df['processed'] = df['text'].apply(denoise_text)
    df.drop_duplicates(subset=['processed'], inplace=True, keep='last')
    print(df.shape, df['label'].value_counts())
    df = remove_special_chars(df, col='processed')
    df['processed'] = df["processed"].apply(lemmetize)
    # Remove words of length 0-2
    df["processed"] = df['processed'].apply(lambda x: " ".join([word for word in x.split()]))
    df.to_csv(path)

# IMDB 50K
if dataset == "50k":
    original_path = "../datasets/IMDB_50k_kaggle.csv"
    path = "../datasets/imdb50k_dataset.csv"
    df = pd.read_csv(original_path)
    df.columns = ['text', 'label']
    if dataset == "50k":
        df.label.replace("positive", 1, inplace=True)
        df.label.replace("negative", 0, inplace=True)
    df = lower_case(df)
    df['text'] = df['text'].apply(denoise_text)
    df = remove_special_chars(df)
    df['processed'] = df["text"].apply(lemmetize)
    # Remove words of length 0-2
    df["processed"] = df['processed'].apply(lambda x: " ".join([word for word in x.split()]))
    df.to_csv(path)


##############################################################
# IMDB YELP AMAZON 1K
# https://www.kaggle.com/marklvl/sentiment-labelled-sentences-data-set
if dataset == "yelp":

    filepath_dict = {'yelp':   '../datasets/sentiment/yelp_labelled.txt',
                     #'amazon': '../datasets/sentiment/amazon_cells_labelled.txt',
                     # 'imdb':   '../datasets/imdb_labelled.txt'}
                     }

    df_list = []
    for source, filepath in filepath_dict.items():
        df = pd.read_csv(filepath, names=['text', 'label'], sep='\t')
        df = lower_case(df)
        df['text'] = df['text'].apply(denoise_text)
        df = remove_special_chars(df)
        print("lemmetize", source)
        df['processed'] = df["text"].apply(lemmetize)
        # Remove words of length 0-2
        df["processed"] = df['processed'].apply(lambda x: " ".join([word for word in x.split()
                                                                    if len(word) > 3]))
        print(df.head())
        df.to_csv(f"../datasets/{source}_dataset.csv")


#############################################################

# Davidson dataset
if dataset == "davidson":
    pth = '../datasets/davidson_dataset.csv'
    df = pd.read_csv(pth)
    print(df['label'].value_counts())
    df_non_hate = df[df["label"] == 0]
    df_non_hate = df_non_hate.sample(1400)
    df_hate = df[df["label"] == 1]
    df_out = df_non_hate.append(df_hate, ignore_index=True)
    df_out.reset_index(drop=True, inplace=True)
    print(df_non_hate.head)
    df_out.drop("index", axis=1, inplace=True)
    df_out.to_csv('bdavidson_dataset.csv')
    df = lower_case(df)
    df = remove_special_chars(df)
    df['processed'] = df['text'].apply(denoise_text)

    # df['processed'] = df["text"].apply(lemmetize)
    # Remove words of length 0-2
    # df["processed"] = df['processed'].apply(lambda x: " ".join([word for word in x.split()
    #                                                             if len(word) > 3]))
    print(df.head())
    df.to_csv(pth)

#########################################
# white supremecy dataset
if dataset == "white":
    # metadata = pd.read_csv('../hate-speech-dataset/annotations_metadata.csv')
    # all_files = '../hate-speech-dataset/all_files/'
    # text = []
    # metadata["label"].replace('hate', 1, inplace=True)
    # metadata["label"].replace("noHate", 0, inplace=True)
    # metadata = metadata[(metadata["label"] == 0) | (metadata["label"] == 1)]
    # print(metadata.head())
    # for id, row in metadata.iterrows():
    #     file = row['file_id']
    #     string = open(all_files + file + '.txt', 'r',  encoding="utf8").read()
    #     text.append(string)
    #
    # df = pd.DataFrame()
    # df["text"] = text
    # df["label"] = metadata["label"]
    # df.to_csv('supremacy_dataset.csv')
    df = pd.read_csv('supremacy_dataset.csv')
    # df["label"].replace('hate', 1, inplace=True)
    # df["label"].replace("noHate", 0, inplace=True)
    # print(df.head())
    # df = lower_case(df)
    # df = remove_special_chars(df)
    # df['processed'] = df['text'].apply(denoise_text)
    # df["processed"] = df['processed'].apply(lambda x: " ".join([word for word in x.split()
    #                                                             if len(word) > 3]))
    # pth = '../datasets/supremacy_dataset.csv'
    # df.to_csv(pth)
