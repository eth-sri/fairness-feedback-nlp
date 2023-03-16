from torch.utils.data import Dataset,DataLoader
from torch.utils.data.dataloader import _BaseDataLoaderIter,_DatasetKind
from Transfer.Datasets.Perturbations import replace_identity,bound,IDENTITY_TERMS
import pandas as pd
import numpy as np
import torch
import re
from Transfer.Models.Models import RobertaClassifierMultiHead
from Transfer.Datasets.Saliency_utils import check_word_list,mask_attribution,gradient_x_saliency,prepend_labels
from transformers import RobertaTokenizer,RobertaForMaskedLM,BartTokenizer,BartForConditionalGeneration
from tqdm import tqdm
import json
import html
import re

import os
dirname = os.path.dirname(__file__)

class ArrayDataset(Dataset):
    # Modified version of torch.utils.data.TensorDataset that takes numpy arrays instead
    def __init__(self, *arrays) -> None:
        assert all(arrays[0].shape[0] == array.shape[0] for array in arrays), "Size mismatch between tensors"
        self.arrays = arrays

    def __getitem__(self, index):
        return tuple(array[index] for array in self.arrays)

    def __len__(self):
        return self.arrays[0].shape[0]

class Mod_SingleProcessDataLoaderIter(_BaseDataLoaderIter):
    def __init__(self, loader):
        super(Mod_SingleProcessDataLoaderIter, self).__init__(loader)
        assert self._timeout == 0
        assert self._num_workers == 0

        self._dataset_fetcher = _DatasetKind.create_fetcher(
            self._dataset_kind, self._dataset, False, lambda x:x, self._drop_last)
        #Set autoollate to false to allow for list-based indexing. Set collate function to the identity

    def _next_data(self):
        index = self._next_index()  # may raise StopIteration
        data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
        if self._pin_memory:
            raise Exception("pin_memory not supported")
        return data

class BatchDataLoader(DataLoader):
    def __init__(self,train_data,batch_size, shuffle):
        super().__init__(train_data,batch_size=batch_size, shuffle=shuffle)
    def _get_iterator(self):
        return Mod_SingleProcessDataLoaderIter(self)

class Kaggle_Toxicity(Dataset):
    # Modified version of torch.utils.data.TensorDataset that takes numpy arrays instead
    def __init__(self,tokenizer=lambda x:x,label_subset = None,dropna=True,max_length=None) -> None:
        try:
            data_train = pd.read_csv(
             dirname + "/../../Datasets/Kaggle_Toxicity/train_preprocessed.csv")
            del data_train["parent_id"]
        except:
            data_train = pd.read_csv(dirname + "/../../Datasets/Kaggle_Toxicity/train.csv")
            data_train["comment_text"] = data_train["comment_text"].str.replace("\n", " ").str.replace("\xa0", " ")
            data_train.to_csv(dirname + "/../../Datasets/Kaggle_Toxicity/train_preprocessed.csv")
            data_train = pd.read_csv(
             dirname + "/../../Datasets/Kaggle_Toxicity/train_preprocessed.csv")
            del data_train["parent_id"]
        dataset = data_train

        if not max_length is None:
            try:
                token_num = np.load(dirname + "/../../Datasets/Kaggle_Toxicity/token_num.npy")
            except:
                tokenizer_lens = RobertaTokenizer.from_pretrained("roberta-base")
                token_num = dataset["comment_text"].map(lambda x: len(tokenizer_lens(x)["input_ids"]) if type(x) is str else 0)
                np.save(dirname + "/../../Datasets/Kaggle_Toxicity/token_num", token_num.values)
                token_num = np.load(dirname + "/../../Datasets/Kaggle_Toxicity/token_num.npy")
            dataset = dataset[token_num <= max_length]
        if dropna:
            dataset = dataset.dropna()
        self.x = tokenizer(list(dataset["comment_text"]))


        self.labels = ['male', 'female', 'transgender', 'other_gender', 'heterosexual', 'homosexual_gay_or_lesbian',
         'bisexual', 'other_sexual_orientation', 'christian', 'jewish', 'muslim', 'hindu',
         'buddhist', 'atheist', 'other_religion', 'black', 'white', 'asian', 'latino',
         'other_race_or_ethnicity', 'physical_disability', 'intellectual_or_learning_disability',
         'psychiatric_or_mental_illness', 'other_disability','target','severe_toxicity','obscene','identity_attack'
         ,'insult','threat','sexual_explicit']
        if not label_subset is None:
            self.labels = [self.labels[i] for i in label_subset]
        self.y = dataset[self.labels].values
        self.pos_weights = (1-(self.y>0.5).mean(0))/(self.y>0.5).mean(0)
        self.y = torch.tensor(self.y)

    def __getitem__(self, index):
        return (self.x[index],self.y[index])

    def __len__(self):
        return len(self.x)

class Data_Pool(Dataset):
    def __init__(self,test_fraction=10,max_length=256,
                 data_sources = {"gradxdrop_256": {"label": 1, "subsample_train":1, "subsample_val":1},
                                "word_replacement_simple":{"label":1, "subsample_train":1, "subsample_val":1},
                                 "word_replacement_improved": {"label": 1, "subsample_train":1, "subsample_val":1},
                                 "identity": {"label": 0, "subsample_train":1, "subsample_val":1},
                                 "bertrandom": {"label": 0, "subsample_train":1, "subsample_val":1},
                                 "wordsrandom": {"label": 0, "subsample_train":1, "subsample_val":1},
                                 "randomtexts": {"label": 0, "subsample_train":1, "subsample_val":1}
                                 },filter_length=True,dropna=False,id_label_only=False,load=None,remove_duplicate_comments=False,lowercase=False):
        assert load is None or data_sources is None
        self.max_length=max_length
        if data_sources:
            #Target label balance is the percentage of "1" labels we want to have in the final pool
            LABELS = {"male":0,"female":1,"transgender":2,"heterosexual":4,"homosexual_gay_or_lesbian":5,"bisexual":6,
                      "christian":8,"jewish":9,"muslim":10,"hindu":11,"buddhist":12,"atheist":13,"black":15,"white":16,
                      "asian":17,"latino":18}
            #TODO: Maybe subfilter all generation methods to avoid AL mostly looking at the presence of identity terms?
            try:
                data_train = pd.read_csv(
                    dirname + "/../../Datasets/Kaggle_Toxicity/Large/train_preprocessed.csv")
            except:
                data_train = pd.read_csv(dirname + "/Kaggle_Toxicity/train.csv")
                data_train["comment_text"] = data_train["comment_text"].str.replace("\n", " ").str.replace("\xa0", " ")
                data_train.to_csv(dirname + "/Kaggle_Toxicity/train_preprocessed.csv")
            data_full_base = data_train
            del data_full_base["parent_id"]
            try:
                token_num = np.load(dirname + "/../../Datasets/Kaggle_Toxicity/Large/token_num.npy")
            except:
                tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
                token_num = data_full_base["comment_text"].map(lambda x: len(tokenizer(x)["input_ids"]))
                np.save(dirname + "/../../Datasets/Kaggle_Toxicity/Large/token_num",token_num.values)

            print(len(data_full_base))
            if filter_length:
                data_full_base = data_full_base[token_num<=max_length]
                print(len(data_full_base))
            if dropna:
                data_full_base = data_full_base.dropna()
                print(len(data_full_base))
            if id_label_only:
                has_identity_label = data_full_base[['male', 'female', 'transgender', 'other_gender', 'heterosexual', 'homosexual_gay_or_lesbian',
                     'bisexual', 'other_sexual_orientation', 'christian', 'jewish', 'muslim', 'hindu',
                     'buddhist', 'atheist', 'other_religion', 'black', 'white', 'asian', 'latino',
                     'other_race_or_ethnicity', 'physical_disability', 'intellectual_or_learning_disability',
                     'psychiatric_or_mental_illness', 'other_disability']].any(axis=1)
                data_full_base = data_full_base[has_identity_label]
                print(len(data_full_base))


            val_indices = np.random.RandomState(seed=42).choice(np.arange(len(data_full_base)), size=len(data_full_base) // test_fraction,
                                                                replace=False)
            train_indices = np.delete(np.arange(len(data_full_base)), val_indices)
            if remove_duplicate_comments:
                checkset = set(list(data_full_base.iloc[val_indices]["comment_text"]))
                print(len(train_indices))
                train_indices = np.array([index for index in train_indices if data_full_base.iloc[index]["comment_text"] not in checkset ])
                print(len(train_indices))

            for mode in ["train","val"]:
                if mode == "val":
                    data_full = data_full_base.iloc[val_indices]
                elif mode == "train":
                    data_full = data_full_base.iloc[train_indices]
                data_full=data_full.reset_index(drop=True)
                dataset_0 = []
                dataset_1 = []
                groups = []
                methods = []
                if "word_replacement_simple" in data_sources:
                    generator = torch.Generator().manual_seed(42)
                    try:
                        contains= pd.read_csv(dirname + "/../../Datasets/Kaggle_Toxicity/perturbable_simple"+mode+"_"+str(max_length)*filter_length+"_dropna"*dropna+"_idonly"*id_label_only+".csv")
                        del contains["Unnamed: 0"]
                    except:
                        contains = pd.concat([data_full["comment_text"].str.contains(bound(term), flags=re.IGNORECASE) for term in IDENTITY_TERMS],1)
                        contains.to_csv(dirname + "/../../Datasets/Kaggle_Toxicity/perturbable_simple"+mode+"_"+str(max_length)*filter_length+"_dropna"*dropna+"_idonly"*id_label_only+".csv")
                    data = data_full[contains.any(axis=1).values]["comment_text"].astype(str).values
                    try:
                        assert data_sources["word_replacement_simple"]["triple"]
                        data = np.repeat(data,3)
                    except:
                        None

                    if data_sources["word_replacement_simple"]["subsample_"+mode] < 1:
                        indices=np.random.RandomState(seed=42).choice(np.arange(len(data)), size=int(len(data)*data_sources["word_replacement_simple"]["subsample_"+mode]),
                                                              replace=False)
                        data = data[indices]
                    print(len(data))
                    if data_sources["word_replacement_simple"]["subsample_"+mode] > 1:
                        indices=np.random.RandomState(seed=42).choice(np.arange(len(data)), size=int(data_sources["word_replacement_simple"]["subsample_"+mode]),
                                                              replace=False)
                        data = data[indices]
                    data_perturbed = replace_identity(data,generator)
                    failed = []
                    for i in range(len(data)):
                        if data[i].lower() == data_perturbed[i].lower():
                            failed.append(i)
                    data=np.delete(data,failed)
                    data_perturbed=np.delete(data_perturbed,failed)
                    if data_sources["word_replacement_simple"]["label"]==1:
                        dataset_1 += list(zip(data,data_perturbed))
                    else:
                        dataset_0 += list(zip(data, data_perturbed))
                    groups += [[None, None] for i in range(len(data))]
                    methods += ["word_replacement_simple" for i in range(len(data))]
                if "folders" in data_sources:
                    for element in data_sources["folders"]:
                        if len(element) == 2:
                            folder,label = element
                        else:
                            folder,label,subsample = element
                            if type(subsample) == tuple:
                                if mode == "train":
                                    subsample = subsample[0]
                                if mode == "val":
                                    subsample = subsample[1]
                        dataset_temp = []
                        groups_temp = []
                        methods_temp = []
                        for target_label in [0,1,2,4,5,8,9,10,11,12,13,15,16,17,18]:
                            for remove_label in [0,1,2,4,5,8,9,10,11,12,13,15,16,17,18]:
                                if target_label==remove_label:
                                    continue
                                if folder.split("/")[-1] == "word_replacement_pairs_50":
                                    if not (target_label,remove_label) in [(0,1),(1,0),(8,10),(10,8),(15,16),(16,15)]:
                                        continue
                                #print(target_label,remove_label)
                                texts = []
                                with open(folder+"/"+str(remove_label)+"_"+str(target_label)+"_base.jsonl","r") as f:
                                    for line in f:
                                        texts.append(json.loads(line))
                                texts_perturbed = []
                                with open(folder+"/"+str(remove_label)+"_"+str(target_label)+"_perturbed.jsonl","r") as f:
                                    for line in f:
                                        texts_perturbed.append(json.loads(line))

                                checkset = set(list(data_full["comment_text"]))
                                to_add = [element for element in list(zip(texts,texts_perturbed)) if element[0] in checkset]
                                dataset_temp += to_add
                                groups_temp += [[remove_label,target_label] for i in range(len(to_add))]
                                methods_temp += [folder.split("/")[-1] for i in range(len(to_add))]
                        print(folder.split("/")[-1])
                        print(len(dataset_temp))
                        if (not len(element) == 2) and subsample!=1:
                            if subsample<1:
                                indices = np.random.RandomState(seed=123).choice(np.arange(len(dataset_temp)), size=int(
                                    len(dataset_temp) * subsample),replace=False)
                            if subsample>1:
                                indices = np.random.RandomState(seed=123).choice(np.arange(len(dataset_temp)), size=int(
                                    subsample),replace=False)
                        else:
                            indices = np.arange(len(dataset_temp))
                        if label == 1:
                            dataset_1 += [dataset_temp[index] for index in indices]
                        else:
                            dataset_0 += [dataset_temp[index] for index in indices]
                        groups += [groups_temp[index] for index in indices]
                        methods += [methods_temp[index] for index in indices]

                if lowercase:
                    #for data in dataset_0:
                    #    print(data[1])
                    #    print(data[1].lower())
                    for i in range(5):
                        print(dataset_0[i])
                    dataset_0 = [(text1.lower(),text2.lower()) for text1,text2 in dataset_0]
                    dataset_1 = [(text1.lower(),text2.lower()) for text1,text2 in dataset_1]
                    for i in range(5):
                        print(dataset_0[i])

                #dataset = np.array(dataset_0 + dataset_1,dtype=object).astype(str) #This could plausibly explain the fuckup as well...
                if len(dataset_0)>0 and len(dataset_1)>0:
                    dataset = np.concatenate([np.array(dataset_0,dtype=object).astype(str),np.array(dataset_1,dtype=object).astype(str)])
                elif len(dataset_0)>0:
                    dataset = np.array(dataset_0,dtype=object).astype(str)
                else:
                    dataset = np.array(dataset_1, dtype=object).astype(str)
                labels = np.array([0 for i in range(len(dataset_0))] + [1 for i in range(len(dataset_1))])
                groups = np.array(groups,dtype=object).astype(str)
                methods = np.array(methods, dtype=object).astype(str)
                if mode == "val":
                    self.data_val = dataset
                    self.labels_val = labels
                    self.groups_val = groups
                    self.methods_val = methods
                    self.reset_pool_weights(eval=True)
                    print(len(self.data_val))
                    print(len(self.labels_val))
                    print(len(self.groups_val))
                    print(len(self.methods_val))
                    self.data_val_base_task = data_full["comment_text"].values.astype(str)
                    self.labels_val_base_task = data_full["target"].values
                    #print(len(self.data_val_base_task))
                    #print(len(self.labels_val_base_task))
                elif mode == "train":
                    self.data_train = dataset
                    self.labels_train = labels
                    self.groups_train = groups
                    self.methods_train = methods
                    self.reset_pool_weights()
                    print(len(self.data_train))
                    print(len(self.labels_train))
                    print(len(self.groups_train))
                    print(len(self.methods_train))

                    self.data_train_base_task = data_full["comment_text"].values.astype(str)
                    self.labels_train_base_task = data_full["target"].values
                    self.pos_weights_base_task = (1 - (self.labels_train_base_task >= 0.5).mean(0)) / (self.labels_train_base_task >= 0.5).mean(0)
        elif load:
            self.data_val=np.load(load + "/data_val.npy")
            self.labels_val=np.load(load + "/labels_val.npy")
            self.data_val_base_task=np.load(load + "/data_val_base_task.npy")
            self.labels_val_base_task=np.load(load + "/labels_val_base_task.npy")
            self.groups_val = np.load(load + "/groups_val.npy")
            self.data_train=np.load(load + "/data_train.npy")
            self.labels_train=np.load(load + "/labels_train.npy")
            self.pool_weights=np.load(load + "/pool_weights.npy")
            self.eval_pool_weights = np.load(load + "/eval_pool_weights.npy")
            self.data_train_base_task=np.load(load + "/data_train_base_task.npy")
            self.labels_train_base_task=np.load(load + "/labels_train_base_task.npy")
            self.groups_train = np.load(load + "/groups_train.npy")
            self.pos_weights_base_task=float(np.load(load + "/pos_weights_base_task.npy"))
            self.methods_train= np.load(load + "/methods_train.npy")
            self.methods_val = np.load(load + "/methods_val.npy")

    def __getitem__(self, index):
        return (index,(self.data_train[index,0],self.data_train[index,1]))
    def __len__(self):
        return len(self.labels_train)
    def save(self,name):
        try:
            os.mkdir(name)
        except:
            answer = None
            while answer not in ["y","n"]:
                answer = input("Save director: "+name+ " already exits. Overwrite [y/n]?")
            if answer == "n":
                print("Not saving")
                return None
        np.save(name+"/data_val.npy",self.data_val)
        np.save(name + "/labels_val.npy", self.labels_val)
        np.save(name + "/data_val_base_task.npy", self.data_val_base_task)
        np.save(name + "/labels_val_base_task.npy", self.labels_val_base_task)
        np.save(name + "/data_train.npy", self.data_train)
        np.save(name+"/labels_train.npy",self.labels_train)
        np.save(name + "/pool_weights.npy", self.pool_weights)
        np.save(name + "/eval_pool_weights.npy", self.eval_pool_weights)
        np.save(name + "/data_train_base_task.npy", self.data_train_base_task)
        np.save(name + "/labels_train_base_task.npy", self.labels_train_base_task)
        np.save(name + "/pos_weights_base_task.npy", self.pos_weights_base_task)
        np.save(name + "/groups_train.npy",self.groups_train)
        np.save(name + "/groups_val.npy", self.groups_val)
        np.save(name + "/methods_train.npy",self.methods_train)
        np.save(name + "/methods_val.npy", self.methods_val)



    def get_pool_weights(self,eval=False):
        if not eval:
            return self.pool_weights
        else:
            return self.eval_pool_weights
    def reset_pool_weights(self,eval=False):
        if not eval:
            self.pool_weights = np.ones(self.labels_train.shape)
        else:
            self.eval_pool_weights = np.ones(self.labels_val.shape)
    def set_pool_weights(self,model,tokenizer,batch_size=16,eval=False,mode="mse"):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if not eval:
            loader = DataLoader(np.arange(len(self.data_train)),batch_size=batch_size,shuffle=False)
            for indices in tqdm(loader, desc="setting pool weights",dynamic_ncols=True):
                data = tokenizer(list(self.data_train[indices,0]), return_tensors="pt", padding=True, truncation=True, max_length=self.max_length).to(device)
                data_perturbed = tokenizer(list(self.data_train[indices,1]), return_tensors="pt", padding=True, truncation=True, max_length=self.max_length).to(
                    device)
                output = model(data)[:, 0]
                output_perturbed = model(data_perturbed)[:, 0]
                if mode == "mse":
                    self.pool_weights[indices]=torch.nn.MSELoss(reduction="none")(output, output_perturbed).detach().cpu().numpy()
                elif mode == "robust":
                    self.pool_weights[indices] = 1 - ((output>0)*(output_perturbed>0)).int().detach().cpu().numpy()  - ((output<0)*(output_perturbed<0)).int().detach().cpu().numpy()
                else:
                    assert False, "Mode should be mse, or robust"
        else:
            loader = DataLoader(np.arange(len(self.data_val)), batch_size=batch_size, shuffle=False)
            for indices in tqdm(loader, desc="setting pool weights", dynamic_ncols=True):
                data = tokenizer(list(self.data_val[indices, 0]), return_tensors="pt", padding=True, truncation=True,
                                 max_length=self.max_length).to(device)
                data_perturbed = tokenizer(list(self.data_val[indices, 1]), return_tensors="pt", padding=True,
                                           truncation=True, max_length=self.max_length).to(
                    device)
                output = model(data)[:, 0]
                output_perturbed = model(data_perturbed)[:, 0]
                if mode == "mse":
                    self.eval_pool_weights[indices] = torch.nn.MSELoss(reduction="none")(output,
                                                                                output_perturbed).detach().cpu().numpy()
                elif mode == "robust":
                    self.eval_pool_weights[indices] = 1 - ((output>0)*(output_perturbed>0)).int().detach().cpu().numpy() - ((output<0)*(output_perturbed<0)).int().detach().cpu().numpy()
                else:
                    assert False, "Mode should be mse, or robust"


    def get_eval_data(self):
        return ArrayDataset(self.data_val[:,0],self.data_val[:,1],self.labels_val)
    def get_train_data(self):
        return ArrayDataset(self.data_train[:,0],self.data_train[:,1],self.labels_train)
    def get_train_data_base_task(self):
        return ArrayDataset(self.data_train_base_task,self.labels_train_base_task)
    def get_val_data_base_task(self):
        return ArrayDataset(self.data_val_base_task,self.labels_val_base_task)
    def get_labels(self,index):
        return self.labels_train[index]
    def get_dict(self):
        out_dict = {}
        for i in range(len(self.data_train)):
            if self.data_train[i,0] in out_dict:
                out_dict[self.data_train[i,0]].append((i,self.data_train[i,1]))
            else:
                out_dict[self.data_train[i,0]] = [(i,self.data_train[i,1])]
        return out_dict
    def extract_validation_indexes(self,methods,groups,n):
        index_list = []
        for method in methods:
            for group1,group2 in groups:
                group1_correct = np.array([self.groups_val[i,0]==group1 for i in range(len(self.groups_val))])
                group2_correct = np.array([self.groups_val[i,1]==group2 for i in range(len(self.groups_val))])
                method_correct = np.array([self.methods_val[i]==method for i in range(len(self.methods_val))])
                bool_indexes = group1_correct*group2_correct*method_correct
                indexes = [i for i in range(len(self.methods_val)) if bool_indexes[i]]
                print(method,group1,group2,len(indexes))
                index_list+=list(np.random.RandomState(seed=42).choice(indexes,size=n,replace=False))
        return index_list
    def get_classifier(self,tpr=1,tnr=1):
        labels = np.zeros(self.labels_train.shape)
        labels[self.labels_train==1] = 2*(np.random.rand(sum(self.labels_train==1))<tpr)-1
        labels[self.labels_train == 0] = -(2*(np.random.rand(sum(self.labels_train==0))<tnr)-1)
        outdict = {tuple(self.data_train[i]):labels[i] for i in range(len(self.data_train))}
        return lambda texts,perts: torch.tensor([[outdict[(text,pert)]] for text,pert in zip(texts,perts)])




    def get_eval_dict(self):
        out_dict = {}
        for i in range(len(self.data_val)):
            if self.labels_val[i] == 1:
                if self.data_val[i,0] in out_dict:
                    out_dict[self.data_val[i,0]].append((i,self.data_val[i,1]))
                else:
                    out_dict[self.data_val[i,0]] = [(i,self.data_val[i,1])]
        return out_dict
    def load_eval_labels(self,path):
        print(self.labels_val.shape)
        labels_val_new = np.zeros(self.labels_val.shape) + np.nan
        labelsindexes = np.load(path+".npy")
        labels = labelsindexes[:,0]
        indexes = labelsindexes[:,1]
        labels_val_new[indexes] = labels
        self.labels_val = labels_val_new

def index2group(index):
    label_names =['Men', 'Women', 'Transgender people', '__', 'Heterosexual people', 'Homosexual people',
     'Bisexual people', '__', 'Christian people', 'Jewish people', 'Muslim people', 'Hindu people',
     'Buddhist people', 'Atheist people', '__', 'Black people', 'White people', 'Asian people', 'Hispanic people',
     '__', '__', '__',
     '__', '__']
    return label_names[index]
