from __future__ import absolute_import, division, print_function

import argparse
import csv
import json
import logging
import os
import random
import sys

import tensorflow as tf
import numpy as np
import torch
import torch.nn.functional as F
from transformers import (WEIGHTS_NAME, AdamW, BertConfig,
                          BertForTokenClassification, BertTokenizer, BertPreTrainedModel, BertModel,
                          get_linear_schedule_with_warmup)

from torch import nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from seqeval.metrics import classification_report

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    filename='log_1_6_hier_final_seed_42_dropout_0.5.txt',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

BertLayerNorm = torch.nn.LayerNorm

label_file = os.path.join("data", "hier_label.txt")

def make_onehot_vector(file):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nodes = []
    num_label = 0
    all_label_dim =242#1~3階層の全ラベル数
    with open(file, 'r') as f:
        for i, line in enumerate(f):

            num_label += 1
            label = line.rstrip('\n')
            if label != 'NULL' and label != 'other':
                nodes.append(label)
    onehot_vector = torch.zeros(all_label_dim, all_label_dim, dtype=torch.float32, device=device)
    with open(file, 'r') as f:
        for id, line in enumerate(f):
            if id == 0 or id == 1:
                code = []
                for i in range(242):
                    if i == id:
                        code.append(1)
                    else:
                        code.append(0)
                onehot_vector[id, :] = torch.tensor(code, dtype=torch.float32, device=device)

            else:
                label = line.rstrip('\n')
                temp_ = label.split("/")[1:]

                #temp = ["/" + "/".join(temp_[:q + 1]) for q in range(1, len(temp_))]#labelを階層ごとに分離

                temp = ["/" + "/".join(temp_[:q + 1]) for q in range(len(temp_))]
                code = []

                code.append(0)
                code.append(0)

                for i, node in enumerate(nodes):
                    if node in temp:
                        code.append(1)
                    else:
                        code.append(0)
                onehot_vector[id, :] = torch.tensor(code, dtype=torch.float32, device=device)
    return onehot_vector

def extract_bottom_embedding(vector, hidden_dim, type_label_dim):#type_label_dim:b,iを除いたtypeを表すラベルの種類数
    device = "cuda" if torch.cuda.is_available() else "cpu"
    matrix = torch.zeros(hidden_dim, type_label_dim, dtype=torch.float32, device=device)
    j = 0
    all_label_dim =242
    for i in range(all_label_dim):
        if i == 0 or i == 1:
            matrix[:,j] = vector[:,i]
            j += 1
        elif i >= 42:# indexが2~41までは第1,2階層ラベル．第3階層ラベルのみを取り出す
            matrix[:,j] = vector[:,i]
            j += 1
    return matrix

def broadcast(vector, span_vector, hidden_dim, label_dim):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    matrix = torch.zeros(hidden_dim, label_dim, dtype=torch.float32, device=device)
    matrix[:, 0] = vector[:, 0]
    matrix[:, 1] = vector[:, 1]
    j = 2
    for i in range(2, 202):
        label_vector = vector[:, i]
        matrix[:, j] =  label_vector + span_vector[0]  # +B_embedding
        matrix[:, j + 1] = label_vector + span_vector[1]# +I_embedding
        j += 2
    return matrix


class New_BertForTokenClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(New_BertForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        #self.label_embedding_weight = nn.Linear(768, 402, bias=False).weight
        #self.label_embedding = nn.Embedding.from_pretrained(self.label_embedding_weight, freeze=False)
        self.onehot_vector = make_onehot_vector(label_file)
        # self.label_embedding_weight = nn.Linear(data.HP_hidden_dim, data.label_alphabet_size, bias=False).weight
        self.label_embedding_weight = nn.Linear(768, 242, bias=False).weight
        self.label_embedding = nn.Embedding.from_pretrained(self.label_embedding_weight, freeze=False)
        # self.label_tensor = torch.LongTensor([i for i in range(data.label_alphabet_size)])
        self.span_embedding_weight = nn.Linear(768, 2, bias=False).weight
        self.span_embedding = nn.Embedding.from_pretrained(self.span_embedding_weight, freeze=False)
        self.init_weights()

class Ner(New_BertForTokenClassification):

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, valid_ids=None,
                attention_mask_label=None, label_embedding_id=None, span_embedding_id=None):
        sequence_output = self.bert(input_ids, token_type_ids, attention_mask, head_mask=None)[0]  # 文ベクトル
        batch_size, max_len, feat_dim = sequence_output.shape  # 行：単語数，列：768次元数(隠れ層の次元)
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32, device='cuda')

        for i in range(batch_size):
            jj = -1
            subword_lis = []
            for j in range(max_len):
                if valid_ids[i][j].item() == 1:
                    if subword_lis:
                        m = nn.MaxPool1d(len(subword_lis))
                        tensor_cat = torch.cat(subword_lis)  # subwordのvectorを連結
                        tensor_cat = torch.transpose(tensor_cat, 0, 1)
                        tensor_cat = tensor_cat.view(1, 768, len(subword_lis))
                        tensor_pooled = m(tensor_cat)
                        subword_vector = tensor_pooled.view(1, 768)[0]
                        valid_output[i][jj] = subword_vector
                        subword_lis = []
                        # counter += 1
                    jj += 1
                    valid_output[i][jj] = sequence_output[i][j]
                else:
                    if not subword_lis:
                        subword_lis.append(sequence_output[i][j - 1].view(1, 768))
                    subword_lis.append(sequence_output[i][j].view(1, 768))
        # print(type_output)
        sequence_output = self.dropout(valid_output)
        label_tensor = torch.tensor(label_embedding_id, device='cuda')
        span_tensor = torch.tensor(span_embedding_id, device='cuda')
        S = self.onehot_vector
        V = self.label_embedding(label_tensor)
        W_label = torch.matmul(S, V)
        W_label = torch.transpose(W_label, 0, 1)
        W_label = extract_bottom_embedding(W_label, 768, 202)
        W_span = self.span_embedding(span_tensor)
        # print(W.size())
        # print(self.label_dim)
        W = broadcast(W_label, W_span, 768, 402)
        # print(W.size())
        logits = torch.matmul(sequence_output, W)
        #W = self.label_embedding(label_tensor)
        #W = torch.transpose(W, 0, 1)
        # output = torch.matmul(sequence_output,S)
        #logits = torch.matmul(sequence_output, W)
        # type_logits = self.type_classifier(sequence_output) # B,I,Oの三種類の次元に線形変換
        # print(type_logits)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            # Only keep active parts of the loss
            attention_mask_label = None

            if attention_mask_label is not None:# 基本的にlossの計算はこっちには飛ばない
                active_loss = attention_mask_label.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            # type_loss = loss_fct(type_logits.view(-1, 4), types.view(-1))
            # print(label_loss)
            # print(label_loss.shape)
            # print(type_loss)
            # print(type_loss.shape)

            return loss
        else:
            return logits



class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, type=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.type = type


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, valid_ids=None, label_mask=None, type_ids=None,
                 type_mask=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask
        self.type_ids = type_ids
        self.type_mask = type_mask



def readfile(filename):
    '''
    read file
    '''
    f = open(filename)
    data = []
    sentence = []
    label = []
    for line in f:
        if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
            if len(sentence) > 0:
                data.append((sentence, label))
                sentence = []
                label = []
            continue
        splits = line.split("\t")
        sentence.append(splits[0])
        label.append(splits[-1][:-1])

    if len(sentence) > 0:
        data.append((sentence, label))
        sentence = []
        label = []
    return data


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    # def get_type(self):
    # raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return readfile(input_file)


"""
    @classmethod
    def _read_tsv_train(cls, input_file, quotechar=None):
        return readfile_train(input_file)
"""


class NerProcessor(DataProcessor):
    """Processor for the CoNLL-2003 data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "valid.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.txt")), "test")

    # def get_labels(self):
    # return ["O", "B-MISC", "I-MISC",  "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "[CLS]", "[SEP]"]

    # def get_labels(self):
    # return ['O', 'Name_Other', 'Person', 'God', 'Organization','Organization_Other', 'International_Organization', 'Show_Organization', 'Family', 'Ethnic_Group', 'Ethnic_Group_Other', 'Nationality', 'Sports_Organization', 'Sports_Organization_Other', 'Pro_Sports_Organization', 'Sports_League', 'Corporation', 'Corporation_Other', 'Company', 'Company_Group', 'Political_Organization', 'Political_Organization_Other', 'Government', 'Political_Party', 'Cabinet', 'Military', 'Location', 'Location_Other', 'Spa', 'GPE', 'GPE_Other', 'City', 'County', 'Province', 'Country', 'Region', 'Region_Other', 'Continental_Region', 'Domestic_Region', 'Geological_Region', 'Geological_Region_Other', 'Mountain', 'Island', 'River', 'Lake', 'Sea', 'Bay', 'Astral_Body', 'Astral_Body_Other', 'Star', 'Planet', 'Constellation', 'Address', 'Address_Other', 'Postal_Address', 'Phone_Number', 'Email', 'URL', 'Facility', 'Facility_Other', 'Facility_Part', 'Archaeological_Place', 'Archaeological_Place_Other', 'Tumulus', 'GOE', 'GOE_Other', 'Public_Institution', 'School', 'Research_Institute', 'Market', 'Park', 'Sports_Facility', 'Museum', 'Zoo', 'Amusement_Park', 'Theater', 'Worship_Place', 'Car_Stop', 'Station', 'Airport', 'Port', 'Line', 'Line_Other', 'Railroad', 'Road', 'Canal', 'Water_Route', 'Tunnel', 'Bridge', 'Product', 'Product_Other', 'Material', 'Clothing', 'Money_Form', 'Drug', 'Weapon', 'Stock', 'Award', 'Decoration', 'Offense', 'Service', 'Class', 'Character', 'ID_Number', 'Vehicle', 'Vehicle_Other', 'Car', 'Train', 'Aircraft', 'Spaceship', 'Ship', 'Food_Other', 'Dish', 'Art', 'Art_Other', 'Picture', 'Broadcast_Program', 'Movie', 'Show', 'Music', 'Book', 'Printing', 'Printing_Other', 'Newspaper', 'Magazine', 'Doctrine_Method', 'Doctrine_Method_Other', 'Culture', 'Religion', 'Academic', 'Sport', 'Style', 'Movement', 'Theory', 'Plan', 'Rule', 'Rule_Other', 'Treaty', 'Law', 'Title', 'Title_Other', 'Position_Vocation', 'Language', 'Language_Other', 'National_Language', 'Unit', 'Unit_Other', 'Currency', 'Event', 'Event_Other', 'Occasion', 'Occasion_Other', 'Religious_Festival', 'Game', 'Conference', 'Incident', 'Incident_Other', 'War', 'Natural_Phenomenon', 'Natural_Phenomenon_Other', 'Natural_Disaster', 'Earthquake', 'Natural_Object', 'Natural_Object_Other', 'Element', 'Compound', 'Mineral', 'Living_Thing', 'Living_Thing_Other', 'Fungus', 'Mollusc_Arthropod', 'Insect', 'Fish', 'Amphibia', 'Reptile', 'Bird', 'Mammal', 'Flora', 'Living_Thing_Part', 'Living_Thing_Part_Other', 'Animal_Part', 'Flora_Part', 'Disease', 'Disease_Other', 'Animal_Disease', 'Color', 'Color_Other', 'Nature_Color', 'Time_Top_Other', 'Timex', 'Timex_Other', 'Time', 'Date', 'Day_Of_Week', 'Era', 'Periodx', 'Periodx_Other', 'Period_Time', 'Period_Day', 'Period_Week', 'Period_Month', 'Period_Year', 'Numex_Other', 'Money', 'Stock_Index', 'Point', 'Percent', 'Multiplication', 'Frequency', 'Age', 'School_Age', 'Ordinal_Number', 'Rank', 'Latitude_Longtitude', 'Measurement', 'Measurement_Other', 'Physical_Extent', 'Space', 'Volume', 'Weight', 'Speed', 'Intensity', 'Temperature', 'Calorie', 'Seismic_Intensity', 'Seismic_Magnitude', 'Countx', 'Countx_Other', 'N_Person', 'N_Organization', 'N_Location', 'N_Location_Other', 'N_Country', 'N_Facility', 'N_Product', 'N_Event', 'N_Natural_Object', 'N_Natural_Object_Other', 'N_Animal', 'N_Flora']

    # def get_labels(self):
    # return ['O', 'Name_Other', 'Person', 'God', 'Organization_Other', 'International_Organization', 'Show_Organization', 'Family', 'Ethnic_Group_Other', 'Nationality', 'Sports_Organization_Other', 'Pro_Sports_Organization', 'Sports_League', 'Corporation_Other', 'Company', 'Company_Group', 'Political_Organization_Other', 'Government', 'Political_Party', 'Cabinet', 'Military', 'Location_Other', 'Spa', 'GPE_Other', 'City', 'County', 'Province', 'Country', 'Region_Other', 'Continental_Region', 'Domestic_Region', 'Geological_Region_Other', 'Mountain', 'Island', 'River', 'Lake', 'Sea', 'Bay', 'Astral_Body_Other', 'Star', 'Planet', 'Constellation', 'Address_Other', 'Postal_Address', 'Phone_Number', 'Email', 'URL', 'Facility_Other', 'Facility_Part', 'Archaeological_Place_Other', 'Tumulus', 'GOE_Other', 'Public_Institution', 'School', 'Research_Institute', 'Market', 'Park', 'Sports_Facility', 'Museum', 'Zoo', 'Amusement_Park', 'Theater', 'Worship_Place', 'Car_Stop', 'Station', 'Airport', 'Port', 'Line_Other', 'Railroad', 'Road', 'Canal', 'Water_Route', 'Tunnel', 'Bridge', 'Product_Other', 'Material', 'Clothing', 'Money_Form', 'Drug', 'Weapon', 'Stock', 'Award', 'Decoration', 'Offense', 'Service', 'Class', 'Character', 'ID_Number', 'Vehicle_Other', 'Car', 'Train', 'Aircraft', 'Spaceship', 'Ship', 'Food_Other', 'Dish', 'Art_Other', 'Picture', 'Broadcast_Program', 'Movie', 'Show', 'Music', 'Book', 'Printing_Other', 'Newspaper', 'Magazine', 'Doctrine_Method_Other', 'Culture', 'Religion', 'Academic', 'Sport', 'Style', 'Movement', 'Theory', 'Plan', 'Rule_Other', 'Treaty', 'Law', 'Title_Other', 'Position_Vocation', 'Language_Other', 'National_Language', 'Unit_Other', 'Currency', 'Event_Other', 'Occasion_Other', 'Religious_Festival', 'Game', 'Conference', 'Incident_Other', 'War', 'Natural_Phenomenon_Other', 'Natural_Disaster', 'Earthquake', 'Natural_Object_Other', 'Element', 'Compound', 'Mineral', 'Living_Thing_Other', 'Fungus', 'Mollusc_Arthropod', 'Insect', 'Fish', 'Amphibia', 'Reptile', 'Bird', 'Mammal', 'Flora', 'Living_Thing_Part_Other', 'Animal_Part', 'Flora_Part', 'Disease_Other', 'Animal_Disease', 'Color_Other', 'Nature_Color', 'Time_Top_Other', 'Timex_Other', 'Time', 'Date', 'Day_Of_Week', 'Era', 'Periodx_Other', 'Period_Time', 'Period_Day', 'Period_Week', 'Period_Month', 'Period_Year', 'Numex_Other', 'Money', 'Stock_Index', 'Point', 'Percent', 'Multiplication', 'Frequency', 'Age', 'School_Age', 'Ordinal_Number', 'Rank', 'Latitude_Longtitude', 'Measurement_Other', 'Physical_Extent', 'Space', 'Volume', 'Weight', 'Speed', 'Intensity', 'Temperature', 'Calorie', 'Seismic_Intensity', 'Seismic_Magnitude', 'Countx_Other', 'N_Person', 'N_Organization', 'N_Location_Other', 'N_Country', 'N_Facility', 'N_Product', 'N_Event', 'N_Natural_Object_Other', 'N_Animal', 'N_Flora']

    def get_labels(self):
        return ['O', 'B-NAME_OTHER', 'I-NAME_OTHER', 'B-PERSON', 'I-PERSON', 'B-GOD', 'I-GOD', 'B-ORGANIZATION_OTHER', 'I-ORGANIZATION_OTHER', 'B-INTERNATIONAL_ORGANIZATION', 'I-INTERNATIONAL_ORGANIZATION', 'B-SHOW_ORGANIZATION', 'I-SHOW_ORGANIZATION', 'B-FAMILY', 'I-FAMILY', 'B-ETHNIC_GROUP_OTHER', 'I-ETHNIC_GROUP_OTHER', 'B-NATIONALITY', 'I-NATIONALITY', 'B-SPORTS_ORGANIZATION_OTHER', 'I-SPORTS_ORGANIZATION_OTHER', 'B-PRO_SPORTS_ORGANIZATION', 'I-PRO_SPORTS_ORGANIZATION', 'B-SPORTS_LEAGUE', 'I-SPORTS_LEAGUE', 'B-CORPORATION_OTHER', 'I-CORPORATION_OTHER', 'B-COMPANY', 'I-COMPANY', 'B-COMPANY_GROUP', 'I-COMPANY_GROUP', 'B-POLITICAL_ORGANIZATION_OTHER', 'I-POLITICAL_ORGANIZATION_OTHER', 'B-GOVERNMENT', 'I-GOVERNMENT', 'B-POLITICAL_PARTY', 'I-POLITICAL_PARTY', 'B-CABINET', 'I-CABINET', 'B-MILITARY', 'I-MILITARY', 'B-LOCATION_OTHER', 'I-LOCATION_OTHER', 'B-SPA', 'I-SPA', 'B-GPE_OTHER', 'I-GPE_OTHER', 'B-CITY', 'I-CITY', 'B-COUNTY', 'I-COUNTY', 'B-PROVINCE', 'I-PROVINCE', 'B-COUNTRY', 'I-COUNTRY', 'B-REGION_OTHER', 'I-REGION_OTHER', 'B-CONTINENTAL_REGION', 'I-CONTINENTAL_REGION', 'B-DOMESTIC_REGION', 'I-DOMESTIC_REGION', 'B-GEOLOGICAL_REGION_OTHER', 'I-GEOLOGICAL_REGION_OTHER', 'B-MOUNTAIN', 'I-MOUNTAIN', 'B-ISLAND', 'I-ISLAND', 'B-RIVER', 'I-RIVER', 'B-LAKE', 'I-LAKE', 'B-SEA', 'I-SEA', 'B-BAY', 'I-BAY', 'B-ASTRAL_BODY_OTHER', 'I-ASTRAL_BODY_OTHER', 'B-STAR', 'I-STAR', 'B-PLANET', 'I-PLANET', 'B-CONSTELLATION', 'I-CONSTELLATION', 'B-ADDRESS_OTHER', 'I-ADDRESS_OTHER', 'B-POSTAL_ADDRESS', 'I-POSTAL_ADDRESS', 'B-PHONE_NUMBER', 'I-PHONE_NUMBER', 'B-EMAIL', 'I-EMAIL', 'B-URL', 'I-URL', 'B-FACILITY_OTHER', 'I-FACILITY_OTHER', 'B-FACILITY_PART', 'I-FACILITY_PART', 'B-ARCHAEOLOGICAL_PLACE_OTHER', 'I-ARCHAEOLOGICAL_PLACE_OTHER', 'B-TUMULUS', 'I-TUMULUS', 'B-GOE_OTHER', 'I-GOE_OTHER', 'B-PUBLIC_INSTITUTION', 'I-PUBLIC_INSTITUTION', 'B-SCHOOL', 'I-SCHOOL', 'B-RESEARCH_INSTITUTE', 'I-RESEARCH_INSTITUTE', 'B-MARKET', 'I-MARKET', 'B-PARK', 'I-PARK', 'B-SPORTS_FACILITY', 'I-SPORTS_FACILITY', 'B-MUSEUM', 'I-MUSEUM', 'B-ZOO', 'I-ZOO', 'B-AMUSEMENT_PARK', 'I-AMUSEMENT_PARK', 'B-THEATER', 'I-THEATER', 'B-WORSHIP_PLACE', 'I-WORSHIP_PLACE', 'B-CAR_STOP', 'I-CAR_STOP', 'B-STATION', 'I-STATION', 'B-AIRPORT', 'I-AIRPORT', 'B-PORT', 'I-PORT', 'B-LINE_OTHER', 'I-LINE_OTHER', 'B-RAILROAD', 'I-RAILROAD', 'B-ROAD', 'I-ROAD', 'B-CANAL', 'I-CANAL', 'B-WATER_ROUTE', 'I-WATER_ROUTE', 'B-TUNNEL', 'I-TUNNEL', 'B-BRIDGE', 'I-BRIDGE', 'B-PRODUCT_OTHER', 'I-PRODUCT_OTHER', 'B-MATERIAL', 'I-MATERIAL', 'B-CLOTHING', 'I-CLOTHING', 'B-MONEY_FORM', 'I-MONEY_FORM', 'B-DRUG', 'I-DRUG', 'B-WEAPON', 'I-WEAPON', 'B-STOCK', 'I-STOCK', 'B-AWARD', 'I-AWARD', 'B-DECORATION', 'I-DECORATION', 'B-OFFENCE', 'I-OFFENCE', 'B-SERVICE', 'I-SERVICE', 'B-CLASS', 'I-CLASS', 'B-CHARACTER', 'I-CHARACTER', 'B-ID_NUMBER', 'I-ID_NUMBER', 'B-VEHICLE_OTHER', 'I-VEHICLE_OTHER', 'B-CAR', 'I-CAR', 'B-TRAIN', 'I-TRAIN', 'B-AIRCRAFT', 'I-AIRCRAFT', 'B-SPACESHIP', 'I-SPACESHIP', 'B-SHIP', 'I-SHIP', 'B-FOOD_OTHER', 'I-FOOD_OTHER', 'B-DISH', 'I-DISH', 'B-ART_OTHER', 'I-ART_OTHER', 'B-PICTURE', 'I-PICTURE', 'B-BROADCAST_PROGRAM', 'I-BROADCAST_PROGRAM', 'B-MOVIE', 'I-MOVIE', 'B-SHOW', 'I-SHOW', 'B-MUSIC', 'I-MUSIC', 'B-BOOK', 'I-BOOK', 'B-PRINTING_OTHER', 'I-PRINTING_OTHER', 'B-NEWSPAPER', 'I-NEWSPAPER', 'B-MAGAZINE', 'I-MAGAZINE', 'B-DOCTRINE_METHOD_OTHER', 'I-DOCTRINE_METHOD_OTHER', 'B-CULTURE', 'I-CULTURE', 'B-RELIGION', 'I-RELIGION', 'B-ACADEMIC', 'I-ACADEMIC', 'B-SPORT', 'I-SPORT', 'B-STYLE', 'I-STYLE', 'B-MOVEMENT', 'I-MOVEMENT', 'B-THEORY', 'I-THEORY', 'B-PLAN', 'I-PLAN', 'B-RULE_OTHER', 'I-RULE_OTHER', 'B-TREATY', 'I-TREATY', 'B-LAW', 'I-LAW', 'B-TITLE_OTHER', 'I-TITLE_OTHER', 'B-POSITION_VOCATION', 'I-POSITION_VOCATION', 'B-LANGUAGE_OTHER', 'I-LANGUAGE_OTHER', 'B-NATIONAL_LANGUAGE', 'I-NATIONAL_LANGUAGE', 'B-UNIT_OTHER', 'I-UNIT_OTHER', 'B-CURRENCY', 'I-CURRENCY', 'B-EVENT_OTHER', 'I-EVENT_OTHER', 'B-OCCASION_OTHER', 'I-OCCASION_OTHER', 'B-RELIGIOUS_FESTIVAL', 'I-RELIGIOUS_FESTIVAL', 'B-GAME', 'I-GAME', 'B-CONFERENCE', 'I-CONFERENCE', 'B-INCIDENT_OTHER', 'I-INCIDENT_OTHER', 'B-WAR', 'I-WAR', 'B-NATURAL_PHENOMENON_OTHER', 'I-NATURAL_PHENOMENON_OTHER', 'B-NATURAL_DISASTER', 'I-NATURAL_DISASTER', 'B-EARTHQUAKE', 'I-EARTHQUAKE', 'B-NATURAL_OBJECT_OTHER', 'I-NATURAL_OBJECT_OTHER', 'B-ELEMENT', 'I-ELEMENT', 'B-COMPOUND', 'I-COMPOUND', 'B-MINERAL', 'I-MINERAL', 'B-LIVING_THING_OTHER', 'I-LIVING_THING_OTHER', 'B-FUNGUS', 'I-FUNGUS', 'B-MOLLUSK_ARTHROPOD', 'I-MOLLUSK_ARTHROPOD', 'B-INSECT', 'I-INSECT', 'B-FISH', 'I-FISH', 'B-AMPHIBIA', 'I-AMPHIBIA', 'B-REPTILE', 'I-REPTILE', 'B-BIRD', 'I-BIRD', 'B-MAMMAL', 'I-MAMMAL', 'B-FLORA', 'I-FLORA', 'B-LIVING_THING_PART_OTHER', 'I-LIVING_THING_PART_OTHER', 'B-ANIMAL_PART', 'I-ANIMAL_PART', 'B-FLORA_PART', 'I-FLORA_PART', 'B-DISEASE_OTHER', 'I-DISEASE_OTHER', 'B-ANIMAL_DISEASE', 'I-ANIMAL_DISEASE', 'B-COLOR_OTHER', 'I-COLOR_OTHER', 'B-NATURE_COLOR', 'I-NATURE_COLOR', 'B-TIME_TOP_OTHER', 'I-TIME_TOP_OTHER', 'B-TIMEX_OTHER', 'I-TIMEX_OTHER', 'B-TIME', 'I-TIME', 'B-DATE', 'I-DATE', 'B-DAY_OF_WEEK', 'I-DAY_OF_WEEK', 'B-ERA', 'I-ERA', 'B-PERIODX_OTHER', 'I-PERIODX_OTHER', 'B-PERIOD_TIME', 'I-PERIOD_TIME', 'B-PERIOD_DAY', 'I-PERIOD_DAY', 'B-PERIOD_WEEK', 'I-PERIOD_WEEK', 'B-PERIOD_MONTH', 'I-PERIOD_MONTH', 'B-PERIOD_YEAR', 'I-PERIOD_YEAR', 'B-NUMEX_OTHER', 'I-NUMEX_OTHER', 'B-MONEY', 'I-MONEY', 'B-STOCK_INDEX', 'I-STOCK_INDEX', 'B-POINT', 'I-POINT', 'B-PERCENT', 'I-PERCENT', 'B-MULTIPLICATION', 'I-MULTIPLICATION', 'B-FREQUENCY', 'I-FREQUENCY', 'B-AGE', 'I-AGE', 'B-SCHOOL_AGE', 'I-SCHOOL_AGE', 'B-ORDINAL_NUMBER', 'I-ORDINAL_NUMBER', 'B-RANK', 'I-RANK', 'B-LATITUDE_LONGITUDE', 'I-LATITUDE_LONGITUDE', 'B-MEASUREMENT_OTHER', 'I-MEASUREMENT_OTHER', 'B-PHYSICAL_EXTENT', 'I-PHYSICAL_EXTENT', 'B-SPACE', 'I-SPACE', 'B-VOLUME', 'I-VOLUME', 'B-WEIGHT', 'I-WEIGHT', 'B-SPEED', 'I-SPEED', 'B-INTENSITY', 'I-INTENSITY', 'B-TEMPERATURE', 'I-TEMPERATURE', 'B-CALORIE', 'I-CALORIE', 'B-SEISMIC_INTENSITY', 'I-SEISMIC_INTENSITY', 'B-SEISMIC_MAGNITUDE', 'I-SEISMIC_MAGNITUDE', 'B-COUNTX_OTHER', 'I-COUNTX_OTHER', 'B-N_PERSON', 'I-N_PERSON', 'B-N_ORGANIZATION', 'I-N_ORGANIZATION', 'B-N_LOCATION_OTHER', 'I-N_LOCATION_OTHER', 'B-N_COUNTRY', 'I-N_COUNTRY', 'B-N_FACILITY', 'I-N_FACILITY', 'B-N_PRODUCT', 'I-N_PRODUCT', 'B-N_EVENT', 'I-N_EVENT', 'B-N_NATURAL_OBJECT_OTHER', 'I-N_NATURAL_OBJECT_OTHER', 'B-N_ANIMAL', 'I-N_ANIMAL', 'B-N_FLORA', 'I-N_FLORA']

    def _create_examples(self, lines, set_type):
        examples = []
        for i, (sentence, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            label = label
            # type = type
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list, 1)}  # {label:0~200の数値}の辞書(key:0~200の数値，value：ラベル)
    # type_map = {type: i for i, type in enumerate(type_list,1)}
    # label_map[0] = 'None'
    # type_map[0] = 'None'

    features = []
    for (ex_index, example) in enumerate(examples):
        textlist = example.text_a.split(" ")  # 各sentenceのtokenが入っている
        labellist = example.label  # 各tokenの正解ラベルが入っている
        typelist = example.type
        tokens = []
        labels = []
        valid = []
        label_mask = []
        # types = []
        # type_mask = []

        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)  # 単語をsubword化
            tokens.extend(token)
            label_1 = labellist[i]  # 単語の正解ラベル
            # type_1 = typelist[i]

            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                    # types.append(type_1)
                    valid.append(1)
                    label_mask.append(1)
                    # type_mask.append(1)
                else:
                    valid.append(0)

        if len(tokens) >= max_seq_length - 1:  # sequenceの長さがmax_seq_lengthを超えたらその分だけ足切り
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
            valid = valid[0:(max_seq_length - 2)]
            label_mask = label_mask[0:(max_seq_length - 2)]
            # type_mask = type_mask[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        type_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        valid.insert(0, 1)
        label_mask.insert(0, 1)
        # label_ids.append(label_map["[CLS]"])  # sentenceの最初は'[CLS]'
        label_ids.append(0)
        # type_mask.insert(0, 1)
        # type_ids.append(0)
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            if len(labels) > i:
                label_ids.append(label_map[labels[i]])  # 正解ラベルのラベルidをマッピング
                # idを1000とかにすると，class数を超えるのでだめ．0以上class数未満にする必要あり
            # if len(types) > i:
            # type_ids.append(type_map[types[i]])
        ntokens.append("[SEP]")
        segment_ids.append(0)
        valid.append(1)
        label_mask.append(1)
        # label_ids.append(label_map["[SEP]"])  # sentenceの終わりは'[SEP]'
        label_ids.append(0)
        # type_mask.append(1)
        # type_ids.append(0)
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)  # tokenをidに変換
        input_mask = [1] * len(input_ids)
        label_mask = [1] * len(label_ids)
        # type_mask = [1] * len(type_ids)
        while len(input_ids) < max_seq_length:  # padding
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            valid.append(1)
            label_mask.append(0)
            # type_ids.append(0)
            # type_mask.append(0)
        while len(label_ids) < max_seq_length:  # padding
            label_ids.append(0)
            label_mask.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(valid) == max_seq_length
        assert len(label_mask) == max_seq_length
        # assert len(type_ids) == max_seq_length
        # assert len(type_mask) == max_seq_length

        if ex_index < 5:  # 例を取り出している
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("valid_ids: %s" % " ".join([str(x) for x in valid]))
            logger.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
            # logger.info("type_ids: %s" % " ".join([str(x) for x in type_ids]))
            # logger.info("label: %s (id = %d)" % (example.label, label_ids))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_ids,
                          valid_ids=valid,
                          label_mask=label_mask))

    return features


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    processors = {"ner": NerProcessor}

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels()
    # type_list = processor.get_type()
    num_labels = len(label_list) + 1
    # num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = 0
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        # num_train_optimization_steps = int(
        # len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * 10
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    # Prepare model
    config = BertConfig.from_pretrained(args.bert_model, num_labels=num_labels, finetuning_task=args.task_name)
    model = Ner.from_pretrained(args.bert_model,
                                from_tf=False,
                                config=config)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(device)

    param_optimizer = list(model.named_parameters())

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    warmup_steps = int(args.warmup_proportion * num_train_optimization_steps)
    # warmup_steps = 0
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    #scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=num_train_optimization_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_optimization_steps)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    label_map = {i: label for i, label in enumerate(label_list, 1)}
    # break_counter = 0

    label_embedding_id = [i for i in range(242)]
    span_embedding_id = [0,1]

    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        all_valid_ids = torch.tensor([f.valid_ids for f in train_features], dtype=torch.long)
        all_lmask_ids = torch.tensor([f.label_mask for f in train_features], dtype=torch.long)
        # all_type_ids = torch.tensor([f.type_ids for f in train_features], dtype=torch.long)
        # all_tmask_ids = torch.tensor([f.type_mask for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_valid_ids,
                                   all_lmask_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        #model.train()
        best_f = -100
        epoch = 0
        tr_loss_list = []
        tr_dev_loss_list = []
        dev_f_list = []
        dev_precision_list = []
        dev_recall_list = []
        train_f_list = []
        train_precision_list = []
        train_recall_list = []

        # label_list_2 =  processor.get_test_labels()
        eval_examples = processor.get_dev_examples(args.data_dir)
        # eval_examples = processor.get_dev_examples(args.data_dir)
        eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length,
                                                     tokenizer)
        # eval_features = convert_examples_to_features(eval_examples, label_list_2, args.max_seq_length, tokenizer)
        all_input_dev_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_dev_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_dev_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_dev_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_valid_dev_ids = torch.tensor([f.valid_ids for f in eval_features], dtype=torch.long)
        all_lmask_dev_ids = torch.tensor([f.label_mask for f in eval_features], dtype=torch.long)
        # all_type_dev_ids = torch.tensor([f.type_ids for f in eval_features], dtype=torch.long)
        # all_tmask_dev_ids = torch.tensor([f.type_mask for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(all_input_dev_ids, all_input_dev_mask, all_segment_dev_ids, all_label_dev_ids,
                                  all_valid_dev_ids, all_lmask_dev_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
        # break_counter = 0

        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            model.train()
            epoch += 1
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, valid_ids, l_mask = batch

                loss = model(input_ids, segment_ids, input_mask, label_ids, valid_ids, l_mask, label_embedding_id=label_embedding_id, span_embedding_id=span_embedding_id)
                # print(label_loss)
                # print(type_loss)
                # loss = label_loss + type_loss

                """
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                """
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

            tr_loss_list.append(tr_loss)  # epochごとのtotallossを格納

            logger.info("***** Running dev_evaluation *****")
            logger.info("  Num examples = %d", len(eval_examples))
            logger.info("  Batch size = %d", args.eval_batch_size)

            model.eval()
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            y_true_label = []
            y_pred_label = []
            y_true_type = []
            y_pred_type = []

            label_map = {i: label for i, label in enumerate(label_list, 1)}
            # type_map = {i: type for i, type in enumerate(type_list, 1)}
            label_map[0] = 'None'
            # type_map[0] = 'None'
            tr_dev_loss = 0
            y_true = []
            y_pred = []

            for input_ids, input_mask, segment_ids, label_ids, valid_ids, l_mask in tqdm(
                    eval_dataloader,
                    desc="Evaluating"):
                # print(type_ids)
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                valid_ids = valid_ids.to(device)
                label_ids = label_ids.to(device)
                l_mask = l_mask.to(device)
                # type_ids = type_ids.to(device)
                # t_mask = t_mask.to(device)
                # print(type_ids)

                # dev_loss = model(input_ids, segment_ids, input_mask, label_ids, valid_ids, l_mask)
                # tr_dev_loss += dev_loss.item()
                with torch.no_grad():
                    logits = model(input_ids, segment_ids, input_mask, valid_ids=valid_ids,
                                   attention_mask_label=l_mask, label_embedding_id=label_embedding_id, span_embedding_id=span_embedding_id)

                logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
                # type_logits = torch.argmax(F.log_softmax(type_logits, dim=2), dim=2)
                logits = logits.detach().cpu().numpy()
                # type_logits = type_logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                # type_ids = type_ids.to('cpu').numpy()
                input_mask = input_mask.to('cpu').numpy()

                for i, label in enumerate(label_ids):
                    temp_1 = []
                    temp_2 = []
                    # print(label) ok
                    for j, m in enumerate(label):
                        if j == 0:
                            continue
                        elif label_ids[i][j] == 0:  # [SEP]の場合
                            y_true.append(temp_1)
                            y_pred.append(temp_2)
                            # print(temp_1) ok
                            break
                        else:
                            temp_1.append(label_map[label_ids[i][j]])
                            temp_2.append(label_map[logits[i][j]])

            output_file = os.path.join(args.output_dir, "dev_predict_third_label" + str(epoch) + ".txt")
            with open(output_file, "w") as writer:
                for i, example in enumerate(eval_examples):
                    sentence = example.text_a.split(' ')
                    for word, true_label, pred_label in zip(sentence, y_true[i], y_pred[i]):
                        line = word + ' ' + true_label + ' ' + pred_label + '\n'
                        writer.write(line)
                    writer.write('\n')

            input_file = os.path.join(args.output_dir, "dev_predict_third_label" + str(epoch) + ".txt")
            with open(input_file, "r") as fi:
                true_labels = []
                pred_labels = []
                for line in fi:
                    if line != '\n':
                        line = line.rstrip('\n').split(' ')
                        true_label = line[1]
                        pred_label = line[2]
                        # if true_label[2:] in label_set or pred_label[2:] in label_set:
                        true_labels.append(true_label)
                        pred_labels.append(pred_label)
            report = classification_report(true_labels, pred_labels, digits=4)
            logger.info("\n%s", report)

            # tr_dev_loss_list.append(tr_dev_loss)
            #report = classification_report(y_true, y_pred, digits=4)
            # report_type = classification_report(y_true_type, y_pred_type, digits=4)
            report_lis = report.split('\n')
            precision = float(report_lis[-3].split()[-4])
            recall = float(report_lis[-3].split()[-3])
            dev_precision_list.append(precision)
            dev_recall_list.append(recall)
            current_f = float(report_lis[-3].split()[-2])  # devのf1
            dev_f_list.append(current_f)
            if current_f > best_f:
                # break_counter = 0
                # output_eval_file_dev = os.path.join(args.output_dir, "dev_eval_results.txt")
                # with open(output_eval_file_dev, "w") as writer:
                # logger.info("***** best_dev_Eval results *****")
                # writer.write('best_epoch:'+str(epoch)+'\n')
                # writer.write(report)
                best_f = current_f
                #best_dev_epoch = epoch
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                model_to_save.save_pretrained(args.output_dir)
                tokenizer.save_pretrained(args.output_dir)
                label_map = {i: label for i, label in enumerate(label_list, 1)}
                # type_map = {i: label for i, label in enumerate(type_list, 1)}
                model_config = {"bert_model": args.bert_model, "do_lower": args.do_lower_case,
                                "max_seq_length": args.max_seq_length, "num_labels": len(label_list) + 1,
                                "label_map": label_map}
                json.dump(model_config, open(os.path.join(args.output_dir, "model_config.json"), "w"))
                #logger.info("\n%s", report)


                # report = classification_report(y_true, y_pred, digits=4)
                #logger.info("\n%s", report)
                output_eval_file_test = os.path.join(args.output_dir,
                                                     "dev_eval_third_results" + str(epoch) + ".txt")
                with open(output_eval_file_test, "w") as writer:
                    # logger.info("***** dev_Eval results *****")
                    # logger.info("\n%s", report)
                    writer.write('best_epoch:' + str(epoch) + '\n')
                    writer.write(report)

        # Save a trained model and the associated configuration
        # model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        # model_to_save.save_pretrained(args.output_dir)
        # tokenizer.save_pretrained(args.output_dir)
        # label_map = {i : label for i, label in enumerate(label_list,1)}
        # model_config = {"bert_model":args.bert_model,"do_lower":args.do_lower_case,"max_seq_length":args.max_seq_length,"num_labels":len(label_list)+1,"label_map":label_map}
        # json.dump(model_config,open(os.path.join(args.output_dir,"model_config.json"),"w"))
        # Load a trained model and config that you have fine-tuned
    else:
        # Load a trained model and vocabulary that you have fine-tuned
        model = Ner.from_pretrained(args.output_dir)
        tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)

    model.to(device)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = processor.get_train_examples(args.data_dir)
        # eval_examples = processor.get_dev_examples(args.data_dir)
        eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length,
                                                     tokenizer)
        logger.info("***** Running train_evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_valid_ids = torch.tensor([f.valid_ids for f in eval_features], dtype=torch.long)
        all_lmask_ids = torch.tensor([f.label_mask for f in eval_features], dtype=torch.long)
        # all_type_ids = torch.tensor([f.type_ids for f in eval_features], dtype=torch.long)
        # all_tmask_ids = torch.tensor([f.type_mask for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_valid_ids,
                                  all_lmask_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        label_map = {i: label for i, label in enumerate(label_list, 1)}
        # type_map = {i: type for i, type in enumerate(type_list, 1)}
        label_map[0] = 'None'
        # type_map[0] = 'None'
        tr_dev_loss = 0
        y_true = []
        y_pred = []

        for input_ids, input_mask, segment_ids, label_ids, valid_ids, l_mask in tqdm(
                eval_dataloader,
                desc="Evaluating"):
            # print(type_ids)
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            valid_ids = valid_ids.to(device)
            label_ids = label_ids.to(device)
            l_mask = l_mask.to(device)
            # type_ids = type_ids.to(device)
            # t_mask = t_mask.to(device)
            # print(type_ids)

            # dev_loss = model(input_ids, segment_ids, input_mask, label_ids, valid_ids, l_mask)
            # tr_dev_loss += dev_loss.item()
            with torch.no_grad():
                label_logits = model(input_ids, segment_ids, input_mask, valid_ids=valid_ids,
                                     attention_mask_label=l_mask, label_embedding_id=label_embedding_id,
                                     span_embedding_id=span_embedding_id)

            logits = torch.argmax(F.log_softmax(label_logits, dim=2), dim=2)
            # type_logits = torch.argmax(F.log_softmax(type_logits, dim=2), dim=2)
            logits = logits.detach().cpu().numpy()
            # type_logits = type_logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            # type_ids = type_ids.to('cpu').numpy()
            input_mask = input_mask.to('cpu').numpy()

            for i, label in enumerate(label_ids):
                temp_1 = []
                temp_2 = []
                # print(label) ok
                for j, m in enumerate(label):
                    if j == 0:
                        continue
                    elif label_ids[i][j] == 0:  # [SEP]の場合
                        y_true.append(temp_1)
                        y_pred.append(temp_2)
                        # print(temp_1) ok
                        break
                    else:
                        temp_1.append(label_map[label_ids[i][j]])
                        temp_2.append(label_map[logits[i][j]])

        output_file = os.path.join(args.output_dir, "train_predict_third_label.txt")
        with open(output_file, "w") as writer:
            for i, example in enumerate(eval_examples):
                sentence = example.text_a.split(' ')
                for word, true_label, pred_label in zip(sentence, y_true[i], y_pred[i]):
                    line = word + ' ' + true_label + ' ' + pred_label + '\n'
                    writer.write(line)
                writer.write('\n')

        input_file = os.path.join(args.output_dir, "train_predict_third_label.txt")
        with open(input_file, "r") as fi:
            true_labels = []
            pred_labels = []
            for line in fi:
                if line != '\n':
                    line = line.rstrip('\n').split(' ')
                    true_label = line[1]
                    pred_label = line[2]
                    # if true_label[2:] in label_set or pred_label[2:] in label_set:
                    true_labels.append(true_label)
                    pred_labels.append(pred_label)
        report = classification_report(true_labels, pred_labels, digits=4)
        logger.info("\n%s", report)

        # report = classification_report(y_true, y_pred, digits=4)
        # logger.info("\n%s", report)
        output_eval_file_test = os.path.join(args.output_dir, "train_eval_third_results.txt")
        with open(output_eval_file_test, "w") as writer:
            logger.info("***** train_Eval results *****")
            logger.info("\n%s", report)
            writer.write(report)

        eval_examples = processor.get_test_examples(args.data_dir)
        # eval_examples = processor.get_dev_examples(args.data_dir)
        eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length,
                                                     tokenizer)
        logger.info("***** Running test_evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_valid_ids = torch.tensor([f.valid_ids for f in eval_features], dtype=torch.long)
        all_lmask_ids = torch.tensor([f.label_mask for f in eval_features], dtype=torch.long)
        # all_type_ids = torch.tensor([f.type_ids for f in eval_features], dtype=torch.long)
        # all_tmask_ids = torch.tensor([f.type_mask for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_valid_ids,
                                  all_lmask_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        label_map = {i: label for i, label in enumerate(label_list, 1)}
        # type_map = {i: type for i, type in enumerate(type_list, 1)}
        label_map[0] = 'None'
        # type_map[0] = 'None'
        tr_dev_loss = 0
        y_true = []
        y_pred = []

        for input_ids, input_mask, segment_ids, label_ids, valid_ids, l_mask in tqdm(
                eval_dataloader,
                desc="Evaluating"):
            # print(type_ids)
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            valid_ids = valid_ids.to(device)
            label_ids = label_ids.to(device)
            l_mask = l_mask.to(device)
            # type_ids = type_ids.to(device)
            # t_mask = t_mask.to(device)
            # print(type_ids)

            # dev_loss = model(input_ids, segment_ids, input_mask, label_ids, valid_ids, l_mask)
            # tr_dev_loss += dev_loss.item()
            with torch.no_grad():
                label_logits = model(input_ids, segment_ids, input_mask, valid_ids=valid_ids,
                                     attention_mask_label=l_mask, label_embedding_id=label_embedding_id,
                                     span_embedding_id=span_embedding_id)

            logits = torch.argmax(F.log_softmax(label_logits, dim=2), dim=2)
            # type_logits = torch.argmax(F.log_softmax(type_logits, dim=2), dim=2)
            logits = logits.detach().cpu().numpy()
            # type_logits = type_logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            # type_ids = type_ids.to('cpu').numpy()
            input_mask = input_mask.to('cpu').numpy()

            for i, label in enumerate(label_ids):
                temp_1 = []
                temp_2 = []
                # print(label) ok
                for j, m in enumerate(label):
                    if j == 0:
                        continue
                    elif label_ids[i][j] == 0:  # [SEP]の場合
                        y_true.append(temp_1)
                        y_pred.append(temp_2)
                        # print(temp_1) ok
                        break
                    else:
                        temp_1.append(label_map[label_ids[i][j]])
                        temp_2.append(label_map[logits[i][j]])

        output_file = os.path.join(args.output_dir, "test_predict_third_label.txt")
        with open(output_file, "w") as writer:
            for i, example in enumerate(eval_examples):
                sentence = example.text_a.split(' ')
                for word, true_label, pred_label in zip(sentence, y_true[i], y_pred[i]):
                    line = word + ' ' + true_label + ' ' + pred_label + '\n'
                    writer.write(line)
                writer.write('\n')

        input_file = os.path.join(args.output_dir, "test_predict_third_label.txt")
        with open(input_file, "r") as fi:
            true_labels = []
            pred_labels = []
            for line in fi:
                if line != '\n':
                    line = line.rstrip('\n').split(' ')
                    true_label = line[1]
                    pred_label = line[2]
                    # if true_label[2:] in label_set or pred_label[2:] in label_set:
                    true_labels.append(true_label)
                    pred_labels.append(pred_label)
        report = classification_report(true_labels, pred_labels, digits=4)
        logger.info("\n%s", report)

