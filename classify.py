import requests
import copy
import operator
import re
import json
import numpy as np
from nltk.tokenize import word_tokenize
from collections import defaultdict, Counter
from numpy import dot
from numpy.linalg import norm

from image_prediction import predict_image_label
from image_train import save_image_training_data


def clean_descriptions(data):
    clean_re = re.compile('<.*?>|\n')
    for row in data:
        row['description'] = re.sub(clean_re, '', row['description']).lower()


def cos_sim_vec(a,b):
    return dot(a, b)/(norm(a)*norm(b))


def cos_sim_word(w1, w2, embedding_dict):
    if w1 not in embedding_dict:
        print(w1 + " not in dictionary")
        return
    if w2 not in embedding_dict:
        print(w2 + " not in dictionary")
        return
    return cos_sim_vec(embedding_dict[w1], embedding_dict[w2])


def add_human_label(data):
    labels = [""] * 30
    labels[0] = 'bag'
    labels[1] = 'pant'
    labels[2] = 'tops'
    labels[3] = 'shoe'
    labels[4] = 'lingerie'
    labels[5] = 'shoe'
    labels[6] = 'others'
    labels[7] = 'jumpsuit'
    labels[8] = 'tops'
    labels[9] = 'others'
    labels[10] = 'tops'
    labels[11] = 'pant'
    labels[12] = 'tops'
    labels[13] = 'pant'
    labels[14] = 'others'
    labels[15] = 'bag'
    labels[16] = 'shoe'
    labels[17] = 'tops'
    labels[18] = 'bag'
    labels[19] = 'pant'
    labels[20] = 'bag'
    labels[21] = 'tops'
    labels[22] = 'bag'
    labels[23] = 'skirt'
    labels[24] = 'bag'
    labels[25] = 'shoe'
    labels[26] = 'pant'
    labels[27] = 'tops'
    labels[28] = 'tops'
    labels[29] = 'tops'

    for i, row in enumerate(data):
        if i >= len(labels) or not labels[i]:
            break
        if 'human_label' not in row:
            row['human_label'] = labels[i]

    return labels


def get_ngram_label(data):
    ngram_label = defaultdict(set)
    for index, row in enumerate(data):
        if 'human_label' not in row:
            break
        label = row['human_label']
        if label:
            for ngram in data[index]['description'].split(','):
                ngram = ngram.strip()
                words = list(filter(len, ngram.split()))
                if len(words) <= 3:
                    ngram_label[ngram].add(label)
            for word in word_tokenize(data[index]['description']):
                ngram_label[word].add(label)
    strong_ngram_label = {}
    for key, value in ngram_label.items():
        if len(value) == 1:
            strong_ngram_label[key] = next(iter(value))
    return strong_ngram_label


def rule_based_classifier(ngram_label_dict, data):
    for row in data:
        if 'rule_label' not in row:
            description = row['description']
            predict_labels = Counter()
            for ngram in description.split(','):
                ngram = ngram.strip()
                if ngram in ngram_label_dict:
                    predict_labels[ngram_label_dict[ngram]] += 1
            for word in word_tokenize(description):
                if word in ngram_label_dict:
                    predict_labels[ngram_label_dict[word]] += 1
            if predict_labels:
                predict_label = max(predict_labels.items(), key=operator.itemgetter(1))[0]
                row['rule_label'] = predict_label


def high_confidence_rule_based_classifier(ngram_label_dict, data):
    for row in data:
        if 'h_rule_label' not in row:
            description = row['description']
            predict_labels = Counter()
            for ngram in description.split(','):
                ngram = ngram.strip()
                if ngram in ngram_label_dict:
                    predict_labels[ngram_label_dict[ngram]] += 1
            for word in word_tokenize(description):
                if word in ngram_label_dict:
                    predict_labels[ngram_label_dict[word]] += 1
            if predict_labels and len(predict_labels) == 1:
                predict_label = predict_labels.most_common(1)[0][0]
                row['h_rule_label'] = predict_label


def word_embedding_classifier(embedding_dict, data, k=3):
    for row in data:
        if 'embedding_label' not in row:
            description = row['description']
            category_scores = defaultdict(list)
            for word in word_tokenize(description):
                if word in embedding_dict:
                    for category in categories:
                        category_scores[category].append(cos_sim_word(word, category, embedding_dict))
            for value in category_scores.values():
                value.sort(reverse=True)
            if not category_scores:
                print("empty description for row")
                print(row)
                continue
            num_sel = min(k, len(list(category_scores.values())[0]))
            label_scores = {}
            for key, value in category_scores.items():
                score = sum(value[:num_sel])
                label_scores[key] = score
            embedding_label = max(label_scores.items(), key=operator.itemgetter(1))[0]
            row['embedding_label'] = embedding_label


def get_labeled_percentage(data):
    embedding_label_per = sum('embedding_label' in row for row in data) / len(data)
    rule_label_per = sum('rule_label' in row for row in data) / len(data)
    h_label_per = sum('h_rule_label' in row for row in data) / len(data)
    return (rule_label_per, h_label_per, embedding_label_per)


def assign_category(data, p_data):
    for i, p_row in enumerate(p_data):
        if 'human_label' in p_row:
            data[i]['category'] = p_row['human_label'].upper()
            continue
        labels = Counter()
        if 'h_rule_label' in p_row:
            labels[p_row['h_rule_label']] += 1
        elif 'rule_label' in p_row:
            labels[p_row['rule_label']] += 1
        if 'embedding_label' in p_row:
            labels[p_row['embedding_label']] += 1
        if 'image_label' in p_row:
            labels[p_row['image_label']] += 1
        label = labels.most_common(1)[0][0]
        data[i]['category'] = label.upper()


if __name__ == '__main__':
    # load data
    data_address = "https://raw.githubusercontent.com/chenlh0/product-classification-challenge/master/product_data.json"
    categories_address = "https://raw.githubusercontent.com/chenlh0/product-classification-challenge/master/product_categories.txt"
    data = requests.get(data_address).json()
    categories_response = requests.get(categories_address).text
    categories = [category.lower() for category in filter(len, categories_response.split('\n'))]

    # make copy of original data
    o_data = copy.deepcopy(data)

    # clean up data
    clean_descriptions(data)
    labels = add_human_label(data)

    # text based classifier
    embedding_dict = {}
    with open('embedding_dict.json') as f:
        embedding_dict = json.load(f)
    strong_ngram_label = get_ngram_label(data)
    rule_based_classifier(strong_ngram_label, data)
    high_confidence_rule_based_classifier(strong_ngram_label, data)
    word_embedding_classifier(embedding_dict, data)
    rule_label_per, h_label_per, embedding_label_per = get_labeled_percentage(data)

    # image training takes time
    image_training = False
    if image_training:
        save_image_training_data(data, categories, labels)
        predict_image_label()
    y_image_pred = np.load('y_image_pred.npy').astype(int)
    for i in range(y_image_pred.shape[0]):
        data[len(labels) + i]['image_label'] = categories[y_image_pred[i]]

    # dump result to disk
    dump_debug_prediction = False
    if dump_debug_prediction:
        prediction_data = json.dumps(data)
        f = open("prediction_data.json", "w")
        f.write(prediction_data)
        f.close()

    assign_category(o_data, data)
    with open("product_data_with_category.json", "w") as f:
        f.write(json.dumps(o_data))
