import re
from nltk.corpus import stopwords
import string
import pandas as pd
from bs4 import BeautifulSoup
import spacy


def lemmetize(doc):
    print(doc)
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


def remove_special_chars(df):
    """

    :param df: input df with 'text' column
    :return: df with special chars removed from text column
    """
    # Remove url
    df['text'] = [re.sub(r"http\S+", " ", text) for text in df["text"]]

    # Remove special chars and numbers
    df['text'] = df['text'].str.replace("[^a-zA-Z#]", " ")

    df["text"] = [re.sub(r"\b[A-Z\.]{2,}s?\b", "", text) for text in df["text"]]
    df["text"] = [re.sub('\S*@\S*\s?', ' ', text) for text in df["text"]]
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
# IMDB 50K
# dataset = "50k"
# original_path = "../datasets/IMDB_50k_kaggle.csv"
# path = "../datasets/imdb50k_dataset.csv"
# df = pd.read_csv(original_path)
# df.columns = ['text', 'label']
# if dataset == "50k":
#     df.label.replace("positive", 1, inplace=True)
#     df.label.replace("negative", 0, inplace=True)
# df = lower_case(df)
# df['text'] = df['text'].apply(denoise_text)
# df = remove_special_chars(df)
# df['processed'] = df["text"].apply(lemmetize)
# # Remove words of length 0-2
# df["processed"] = df['processed'].apply(lambda x: " ".join([word for word in x.split()]))
# df.to_csv(path)


##############################################################
# IMDB YELP AMAZON 1K
# https://www.kaggle.com/marklvl/sentiment-labelled-sentences-data-set
# import pandas as pd
#
# filepath_dict = {'yelp':   '../datasets/yelp_labelled.txt',}
#                  # 'amazon': '../datasets/amazon_cells_labelled.txt',
#                  # 'imdb':   '../datasets/imdb_labelled.txt'}
#
# df_list = []
# for source, filepath in filepath_dict.items():
#     df = pd.read_csv(filepath, names=['text', 'label'], sep='\t')
#     df = lower_case(df)
#     df['text'] = df['text'].apply(denoise_text)
#     df = remove_special_chars(df)
#     print("lemmetize", source)
#     df['processed'] = df["text"].apply(lemmetize)
#     # Remove words of length 0-2
#     # df["processed"] = df['processed'].apply(lambda x: " ".join([word for word in x.split()
#     #                                                             if len(word) > 3]))
#     print(df.head())
#     df.to_csv(f"../datasets/{source}_dataset.csv")
#
#
##############################################################

# Davidson dataset
# pth = '../datasets/davidson_dataset.csv'
# df = pd.read_csv(pth)
# df = lower_case(df)
# df['text'] = df['text'].apply(denoise_text)
# df = remove_special_chars(df)
# df['processed'] = df["text"].apply(lemmetize)
# # Remove words of length 0-2
# # df["processed"] = df['processed'].apply(lambda x: " ".join([word for word in x.split()
# #                                                             if len(word) > 3]))
# print(df.head())
# df.to_csv(pth)

#########################################
# YELP HUGE
pth = '../datasets/huge_yelp.csv'
df = pd.read_csv(pth)
df = df[['text', 'stars']]


df_good = df[(df["stars"] == 5)]

df_good = df_good.sample(750)
df_bad = df[(df["stars"] == 1)]

df_final = df_good.append(df_bad)

df_final.stars.replace(5, '1', inplace=True)
df_final.stars.replace(1, 0, inplace=True)
df_final.stars.replace('1', 1, inplace=True)
df_final.columns = ["text", "label"]
df_final.reset_index(drop=True, inplace=True)
print(df_final.columns)
print(df_final.head())
print(df_final["label"].value_counts())
df = lower_case(df_final)
df['text'] = df['text'].apply(denoise_text)
df = remove_special_chars(df)
print("lemmetize")
df['processed'] = df["text"].apply(lemmetize)
# Remove words of length 0-2
df["processed"] = df['processed'].apply(lambda x: " ".join([word for word in x.split()
                                                            if len(word) > 2]))
print(df.head())
df_yelp = pd.read_csv('../datasets/yelp_dataset.csv')
df_combined = df_yelp.append(df, ignore_index=True)
df_combined.drop_duplicates(inplace=True)
print(df.shape, df_yelp.shape, df_combined.shape)
df.to_csv('../datasets/combined_yelp_dataset.csv')