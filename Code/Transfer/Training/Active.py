import torch
from Transfer.Models.Models import RobertaClassifierMultiHead
from Transfer.Datasets.Perturbations import censor_identity,censor_identity_extended
from transformers import RobertaForSequenceClassification,RobertaTokenizer
from torch.utils.data import DataLoader,Dataset,Subset
from Transfer.Datasets.Datasets import ArrayDataset,BatchDataLoader
from Transfer.Models.Models import DualModel
from tqdm import tqdm
import numpy as np
import os
import pickle
from copy import deepcopy
import json
import scipy.stats
from scipy.stats import entropy,beta
import transformers



def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return "LB: " +str(m-h) + " Mean: " +str(m) + " UB: " +str(m+h)

def active_selection(data_pool,random_baseline,dropout_estimate,subsample_size,train_batch_size,model,cost_weight,use_pool_weights,
                     dropout_mode="Majority",labeled_dataset_dict={},overwrite_resamples=False,subsample_once=True):
    if subsample_size is None:
        pool_loader = DataLoader(data_pool, batch_size=train_batch_size, shuffle=False)
    else:
        if subsample_once:
            indices = np.random.RandomState(seed=42).choice(np.arange(len(data_pool)),
                                                            size=subsample_size, replace=False)
        else:
            indices = np.random.choice(np.arange(len(data_pool)),
                                                            size=subsample_size, replace=False)
        pool_loader = DataLoader(Subset(data_pool, indices), batch_size=train_batch_size, shuffle=False)
    model.eval()
    if not random_baseline:
        with torch.no_grad():
            scores = torch.zeros(len(data_pool))
            pred_probs = torch.zeros(len(data_pool))
            for indexes, data_pair in tqdm(pool_loader, desc="scoring_epoch",ascii=True,dynamic_ncols=True):
            #for indexes, data_pair in pool_loader:
                data_1 = data_pair[0]
                data_2 = data_pair[1]
                if not dropout_estimate:
                    logits = model(list(data_1), list(data_2)).detach().cpu()[:,0]
                    probs = torch.nn.Sigmoid()(logits)
                    min_prob = torch.min(probs,1-probs)
                    min_class = probs<=0.5
                    scores[indexes] = min_prob + (cost_weight - 1) * (min_class).float() * min_prob

                elif dropout_mode == "Majority" or dropout_mode == "Majority-Fast":
                    if not dropout_mode == "Majority-Fast":
                        model.train()
                    else:
                        model.dropout.train()
                        model.pooler.train()
                    counts = []
                    if dropout_mode == "Majority-Fast":
                        base = model(list(data_1), list(data_2),base_only=True)
                    for j in range(dropout_estimate):
                        if not dropout_mode == "Majority-Fast":
                            logits = model(list(data_1), list(data_2)).detach().cpu()[:,0]
                        else:
                            logits = model.output(model.dropout(model.pooler(base))).detach().cpu()[:,0]
                        classes = (logits>0).int()
                        counts.append(classes)
                    probs = torch.stack(counts, 0).float().mean(0).cpu()
                    min_prob = torch.min(probs,1-probs)
                    min_class = probs<=0.5
                    scores[indexes] = min_prob + (cost_weight - 1) * (min_class == 1).float() * min_prob
                elif dropout_mode == "Mean" or dropout_mode == "Mean-Fast":
                    if not dropout_mode == "Mean-Fast":
                        model.train()
                    else:
                        model.dropout.train()
                        model.pooler.train()
                    probs = []
                    if dropout_mode == "Mean-Fast":
                        base = model(list(data_1), list(data_2),base_only=True)
                    for j in range(dropout_estimate):
                        if not dropout_mode == "Mean-Fast":
                            logits = model(list(data_1), list(data_2)).detach()[:,0].detach().cpu()
                        else:
                            logits = model.output(model.dropout(model.pooler(base)))[:,0].detach().cpu()
                        probs.append(torch.nn.Sigmoid()(logits))
                    probs = torch.stack(probs, 0).float().mean(0).cpu()
                    min_prob = torch.min(probs,1-probs)
                    min_class = probs<=0.5
                    scores[indexes] = min_prob + (cost_weight - 1) * (min_class == 1).float() * min_prob
                elif dropout_mode == "Mean-Fast-Pos":
                    model.dropout.train()
                    model.pooler.train()
                    probs = []
                    base = model(list(data_1), list(data_2),base_only=True)
                    for j in range(dropout_estimate):
                        logits = model.output(model.dropout(model.pooler(base)))[:,0].detach().cpu()
                        probs.append(torch.nn.Sigmoid()(logits))
                    probs = torch.stack(probs, 0).float().mean(0).cpu()
                    probs_censored = probs * (probs>0.5) #Only look at probabilities for things that are classified as one.
                    min_prob = torch.min(probs_censored,1-probs_censored)
                    min_class = probs_censored<=0.5
                    scores[indexes] = min_prob + (cost_weight - 1) * (min_class == 1).float() * min_prob
                    pred_probs[indexes] = probs
                elif dropout_mode == "Varra-Fast":
                    ratios = []
                    base = model(list(data_1), list(data_2), base_only=True)
                    for j in range(dropout_estimate):
                        logits = model.output(model.dropout(model.pooler(base)))[:,0].detach().cpu()
                        probs = torch.nn.Sigmoid()(logits)
                        min_prob = torch.min(probs, 1 - probs)
                        min_class = probs <= 0.5
                        ratios.append(min_prob + (cost_weight - 1) * (min_class == 1).float() * min_prob)
                    scores[indexes] = torch.stack(ratios,0).float().mean(0).cpu()
                elif dropout_mode == "BALD" or dropout_mode == "BALD-Fast":
                    if not dropout_mode == "BALD-Fast":
                        model.train()
                    else:
                        model.dropout.train()
                        model.pooler.train()
                    probs0 = []
                    probs1 = []
                    if dropout_mode == "BALD-Fast":
                        base = model(list(data_1), list(data_2),base_only=True)
                    for j in range(dropout_estimate):
                        if not dropout_mode == "BALD-Fast":
                            logits = model(list(data_1), list(data_2)).detach()[:,0].detach().cpu()
                        else:
                            logits = model.output(model.dropout(model.pooler(base)))[:,0].detach().cpu()
                        probs = torch.nn.Sigmoid()(logits)
                        probs0.append(1-probs)
                        probs1.append(probs)
                    probs0 = torch.stack(probs0,0).cpu()
                    probs1 = torch.stack(probs1, 0).cpu()
                    avg_entropy = torch.mean(probs0,0)*torch.log(torch.mean(probs0,0))+torch.mean(probs1,0)*torch.log(torch.mean(probs1,0))
                    entropy_avg = torch.mean(probs0*torch.log(probs0),0)+torch.mean(probs1*torch.log(probs1),0)
                    scores[indexes] = entropy_avg-avg_entropy
    else:
        scores = torch.rand(len(data_pool))
    if use_pool_weights:
        assert data_pool.get_pool_weights().shape == scores.shape
        scores = scores * data_pool.get_pool_weights()
    print("a",torch.quantile(scores[scores>0],torch.tensor([0.05,0.1,0.25,0.5,0.75,0.9,0.95])))
    if overwrite_resamples:
        if overwrite_resamples is True:
            scores[list(labeled_dataset_dict.keys())] = 0
        else:
            if overwrite_resamples == "bayes_mean":
                for key in labeled_dataset_dict.keys():
                    labels = labeled_dataset_dict[key][2]
                    p_next_label = (sum(labels)+1)/(len(labels)+2)
                    mean = np.mean(labels)
                    mean_0 = (np.sum(labels))/(len(labels)+1)
                    mean_1 = (np.sum(labels)+1) / (len(labels) + 1)
                    scores[key] = np.abs(mean-mean_0)*(1-p_next_label) + np.abs(mean-mean_1)*p_next_label + np.random.randn()*1e-4 #topk would otherwise alway pick ties by index.
                print("b", torch.quantile(scores[scores>0], torch.tensor([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])))
            elif overwrite_resamples == "beta_bald":
                for key in labeled_dataset_dict.keys():
                    labels = labeled_dataset_dict[key][2]
                    a = np.sum(labels) + 1
                    b = len(labels) + 2 - a
                    scores[key] = entropy([beta.mean(a,b),1-beta.mean(a,b)]) + beta.expect(lambda x:x*np.log(x),args=(a,b)) + beta.expect(lambda x:(1-x)*np.log(1-x),args=(a,b)) + np.random.randn()*1e-4
                print("b", torch.quantile(scores[scores>0], torch.tensor([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])))
            elif overwrite_resamples == "max3":
                for key in labeled_dataset_dict.keys():
                    labels = labeled_dataset_dict[key][2]
                    if len(labels)>2:
                        scores[key]=0.0
            elif overwrite_resamples == "Inverse":
                scores[[i for i in range(len(scores)) if not i in list(labeled_dataset_dict.keys())]] = 0
            elif overwrite_resamples == "Inverse_Pos":
                scores_labeled_only = torch.clone(scores)
                scores_labeled_only[[i for i in range(len(scores)) if not i in list(labeled_dataset_dict.keys())]] = 0
                print("ab", torch.quantile(scores_labeled_only[scores_labeled_only > 0], torch.tensor([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])))

                scores[[i for i in range(len(scores)) if ((not i in list(labeled_dataset_dict.keys())) or (np.mean(labeled_dataset_dict[i][2])<0.5))]] = 0
            else:
                assert False, "overwrite mode unknown"
    return scores

def eval_active(eval_data,train_batch_size,model,return_metrics=False,dec_thresh=0.5,log_probs=False):
    try:
        model.eval()
    except:
        None
    with torch.no_grad():
        eval_loader = DataLoader(eval_data, batch_size=train_batch_size, shuffle=False)
        preds = []
        correct = []
        for data_1, data_2, labels in tqdm(eval_loader, desc="evaluating_epoch",ascii=True,dynamic_ncols=True):
            if sum(~np.isnan(np.array(labels)))>0:
                logits = model(np.array(data_1)[~np.isnan(np.array(labels))], np.array(data_2)[~np.isnan(np.array(labels))]).detach()
                if dec_thresh == 0.5:
                    probs = torch.nn.Sigmoid()(logits)
                    predictions = logits > 0
                else:
                    probs = torch.nn.Sigmoid()(logits)
                    predictions = probs>dec_thresh
                preds += list(predictions)
                correct += list(labels[~np.isnan(np.array(labels))])
                if log_probs:
                    for i in range(len(probs)):
                        print(np.array(data_1)[~np.isnan(np.array(labels))][i])
                        print(np.array(data_2)[~np.isnan(np.array(labels))][i])
                        print(probs[i])
                        print("\n")

        preds = torch.tensor(preds, dtype=int)
        correct = torch.stack(correct).int()
        tp = (preds * correct).sum()
        tn = ((1 - correct) * (1 - preds)).sum()
        fp = ((1 - correct) * (preds)).sum()
        fn = ((correct) * (1 - preds)).sum()
        print(tp,fp,tn,fn)
    if return_metrics:
        return {"tpr_active":tp / (tp + fn),"tnr_active":tn / (tn + fp),"ba_active": tp / (2*tp + 2*fn)+tn / (2*tn + 2*fp)  ,"acc_active":(tp+tn)/(tp+tn+fn+fp)}


def train_active(train_loader,model,optimizer,print_results=False,reweigh_loss=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.train()
    if reweigh_loss:
        pos_weight = reweigh_loss
    if print_results:
        preds = []
        correct = []
    for data_1, data_2, labels in train_loader:
        logits = model(list(data_1), list(data_2))
        if not reweigh_loss:
            loss = torch.nn.BCEWithLogitsLoss()(logits[:,0],torch.tensor(labels).to(device).float())
        else:
            loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))(logits[:,0],torch.tensor(labels).to(device).float())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if print_results:
            predictions = logits > 0
            preds += list(predictions)
            correct += list(labels)
        loss = loss.detach()
        logits = logits.detach()
    if print_results:
        preds = torch.tensor(preds, dtype=int)
        correct = torch.tensor(correct, dtype=int)
        tp = (preds * correct).sum()
        tn = ((1 - correct) * (1 - preds)).sum()
        fp = ((1 - correct) * (preds)).sum()
        fn = ((correct) * (1 - preds)).sum()
        print({"tpr_train":tp / (tp + fn),"tnr_train":tn / (tn + fp),"ba_train": tp / (2*tp + 2*fn)+tn / (2*tn + 2*fp)  ,"acc_train":(tp+tn)/(tp+tn+fn+fp)})

def save_active_state(model,optimizer,labeled_dataset_dict,name):
    try:
        os.mkdir(name)
    except OSError as error:
        print(error," Last saves will be overwritten")
    torch.save(model.state_dict(), name+"/model")
    torch.save(optimizer.state_dict(), name + "/optimizer")
    with open(name + "/labeled_dataset_dict", "wb") as fp:
        pickle.dump(labeled_dataset_dict, fp)

def load_active_state(name,lr,model):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.load_state_dict(torch.load(name+"/model"))
    optimizer.load_state_dict(torch.load(name + "/optimizer"))
    with open(name + "/labeled_dataset_dict", "rb") as fp:
        labeled_dataset_dict = pickle.load(fp)
    return model,optimizer,labeled_dataset_dict

def init_human_loop(data_pool,model,lr, random_baseline, dropout_estimate, subsample_size, train_batch_size, cost_weight,use_pool_weights,query_batch_size,
                    optimizer=None,labeled_dataset_dict={},softmax=False,name=None,out="indices",dropout_mode="Majority",overwrite_resamples=False,subsample_once=True):
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scores = active_selection(data_pool, random_baseline, dropout_estimate, subsample_size, train_batch_size,
                             model, cost_weight,use_pool_weights,dropout_mode=dropout_mode,labeled_dataset_dict=labeled_dataset_dict,overwrite_resamples=overwrite_resamples,subsample_once=subsample_once)
    if not softmax:
        selected_indexes = torch.topk(scores, query_batch_size).indices
    else:
        selected_indexes = torch.tensor(np.random.choice(np.arange(len(scores)),size=query_batch_size,p=torch.nn.Softmax()(scores.double()),replace=False))
    if not name is None:
        save_active_state(model, optimizer, labeled_dataset_dict, name)
        if out == "pairs":
            base = list(data_pool[selected_indexes][1][0])
            pert = list(data_pool[selected_indexes][1][1])
            return zip(base,pert)
        elif out == "indices":
            return selected_indexes
        elif out == "both":
            base = list(data_pool[selected_indexes][1][0])
            pert = list(data_pool[selected_indexes][1][1])
            return selected_indexes,zip(base,pert)
    else:
        if out == "pairs":
            base = list(data_pool[selected_indexes][1][0])
            pert = list(data_pool[selected_indexes][1][1])
            return zip(base,pert),model,optimizer,labeled_dataset_dict
        elif out == "indices":
            return selected_indexes,model,optimizer,labeled_dataset_dict
        elif out == "both":
            base = list(data_pool[selected_indexes][1][0])
            pert = list(data_pool[selected_indexes][1][1])
            return selected_indexes, zip(base,pert), model, optimizer, labeled_dataset_dict



def human_loop(selected_indexes,base_texts,pert_texts,labels,data_pool,random_baseline, dropout_estimate, subsample_size, cost_weight,use_pool_weights,query_batch_size,
               from_scratch=False,eval=False,no_selection=False, train_batch_size=None,load_name=None,save_name=None,lr=None,model_handle="roberta-base",merge_mode="features",
               model=None, optimizer=None, labeled_dataset_dict={},softmax=False,return_metrics=False,out="indices",AL_train_epochs=1,
               aggregate="mean",dropout_classifier=False,dropout_mode="Majority",overwrite_resamples=False,subsample_once=True,
               print_AL_train=False,eval_every_step=False,reweigh_loss=False,cv_train=False,adamw=False,freeze_layers=False,dec_thresh=0.5):
    if not load_name is None:
        assert optimizer is None and len(labeled_dataset_dict) == 0
        model, optimizer, labeled_dataset_dict = load_active_state(load_name,lr,model)
    print(len(labeled_dataset_dict.keys()))
    for i,index in enumerate(selected_indexes):
        index = int(index)
        if index in labeled_dataset_dict.keys():
            assert labeled_dataset_dict[index][0] == base_texts[i] and labeled_dataset_dict[index][1] == pert_texts[i]
            labeled_dataset_dict[index] = (base_texts[i], pert_texts[i], labeled_dataset_dict[index][2]+[labels[i]])
        else:
            labeled_dataset_dict[index] = (base_texts[i],pert_texts[i],[labels[i]])
    print(len(labeled_dataset_dict.keys()))
    print(max([len(labeled_dataset_dict[key]) for key in labeled_dataset_dict.keys()]))
    if aggregate == "mean":
        aggregate = np.mean
    elif aggregate == "laplace":
        def aggregate(label_list):
            return (np.sum(label_list)+1)/(len(label_list)+2)
    elif aggregate == "majority":
        def aggregate(label_list):
            mean = np.mean(label_list)
            if mean >0.5:
                return 1
            if mean <0.5:
                return 0
            if mean == 0.5:
                return mean
    elif aggregate == "first":
        def aggregate(label_list):
            return label_list[0]


    if from_scratch:
        if not from_scratch=="keep":
            model = DualModel(model_handle, merge_mode, dropout_classifier=dropout_classifier, max_length=128).to(
                torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            if freeze_layers:
                modules = [model.model.embeddings,*model.model.encoder.layer[:freeze_layers]]
                for module in modules:
                    for param in module.parameters():
                        param.requires_grad = False
            if not adamw:
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            else:
                print(adamw)
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr,weight_decay=adamw)

        if aggregate:
            train_dataset = ArrayDataset(np.array([entry[0] for entry in labeled_dataset_dict.values()], dtype=object),
                                         np.array([entry[1] for entry in labeled_dataset_dict.values()], dtype=object),
                                         np.array([aggregate(entry[2]) for entry in labeled_dataset_dict.values()]))
        else:
            train_dataset = ArrayDataset(np.array([entry[0] for entry in labeled_dataset_dict.values() for item in entry[2]], dtype=object),
                                         np.array([entry[1] for entry in labeled_dataset_dict.values() for item in entry[2]], dtype=object),
                                         np.array([item for entry in labeled_dataset_dict.values() for item in entry[2]]))
    else:
        if aggregate:
            train_dataset = ArrayDataset(np.array(base_texts, dtype=object),
                                         np.array(pert_texts, dtype=object),
                                         np.array([aggregate(labeled_dataset_dict[int(index)][2]) for index in selected_indexes]))
        else:
            assert False, "Aggregation=False not currently implemented in this setting"

    if cv_train:
        val_indices = np.random.choice(np.arange(len(train_dataset)),size=int(len(train_dataset)*cv_train),replace=False)
        train_indices = np.delete(np.arange(len(train_dataset)), val_indices)
        val_data = Subset(train_dataset,val_indices)
        train_dataset = Subset(train_dataset,train_indices)
    train_loader = BatchDataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    print(len(train_dataset))

    if reweigh_loss is True:
        pos_weight = (train_dataset[:][2] == 0).mean() / (train_dataset[:][2] == 1).mean()
        print(pos_weight)
    elif reweigh_loss:
        pos_weight = reweigh_loss
    else:
        pos_weight = False
    for i in range(AL_train_epochs):
        print("epoch",i)
        train_active(train_loader,  model, optimizer,print_results=print_AL_train,reweigh_loss=pos_weight)
        if eval_every_step or i == AL_train_epochs-1:
            if eval:
                if type(eval) == list:
                    try:
                        AL_result_dict
                    except:
                        AL_result_dict = {}
                    for path in eval:
                        data_pool.load_eval_labels(path)
                        eval_data = data_pool.get_eval_data()
                        AL_result_dict_temp = eval_active(eval_data, train_batch_size, model, return_metrics=return_metrics,dec_thresh=dec_thresh)
                        for key in AL_result_dict_temp.keys():
                            AL_result_dict[path+key] = AL_result_dict_temp[key]
                else:
                    eval_data = data_pool.get_eval_data()
                    AL_result_dict = eval_active(eval_data, train_batch_size, model,return_metrics=return_metrics,dec_thresh=dec_thresh)



                try:
                    AL_result_dict_temp = eval_active(val_data, train_batch_size, model, return_metrics=return_metrics,
                                                      dec_thresh=dec_thresh)
                    for key in AL_result_dict_temp.keys():
                        AL_result_dict["traincv" + key] = AL_result_dict_temp[key]
                except Exception as e: print(e)
                print("Marker",AL_result_dict)

    if not no_selection:
        scores = active_selection(data_pool, random_baseline, dropout_estimate, subsample_size, train_batch_size,
                                  model, cost_weight,use_pool_weights,dropout_mode=dropout_mode,labeled_dataset_dict=labeled_dataset_dict,overwrite_resamples=overwrite_resamples,subsample_once=subsample_once)
    else:
        scores = active_selection(data_pool, True, dropout_estimate, subsample_size, train_batch_size,
                                   model, cost_weight,use_pool_weights,dropout_mode=dropout_mode,labeled_dataset_dict=labeled_dataset_dict,overwrite_resamples=overwrite_resamples,subsample_once=subsample_once)

    if not softmax:
        selected_indexes = torch.topk(scores, query_batch_size).indices
    else:
        selected_indexes = torch.tensor(np.random.choice(np.arange(len(scores)),size=query_batch_size,p=torch.nn.Softmax()(scores.double()/softmax),replace=False))
    if not save_name is None:
        save_active_state(model, optimizer, labeled_dataset_dict, save_name)
        if out == "pairs":
            base = list(data_pool[selected_indexes][1][0])
            pert = list(data_pool[selected_indexes][1][1])
            if not return_metrics:
                return zip(base, pert)
            else:
                return zip(base, pert),AL_result_dict
        elif out == "indices":
            if not return_metrics:
                return selected_indexes
            else:
                return selected_indexes, AL_result_dict
        elif out == "both":
            base = list(data_pool[selected_indexes][1][0])
            pert = list(data_pool[selected_indexes][1][1])
            if not return_metrics:
                return selected_indexes,zip(base,pert)
            else:
                return selected_indexes,zip(base,pert),AL_result_dict
    else:
        if out == "pairs":
            base = list(data_pool[selected_indexes][1][0])
            pert = list(data_pool[selected_indexes][1][1])
            if not return_metrics:
                return zip(base, pert), model, optimizer, labeled_dataset_dict
            else:
                return zip(base,
                           pert), model, optimizer, labeled_dataset_dict,AL_result_dict
        elif out == "indices":
            if not return_metrics:
                return selected_indexes, model, optimizer, labeled_dataset_dict
            else:
                return selected_indexes, model, optimizer, labeled_dataset_dict,AL_result_dict
        elif out == "both":
            base = list(data_pool[selected_indexes][1][0])
            pert = list(data_pool[selected_indexes][1][1])
            if not return_metrics:
                return selected_indexes, zip(base,pert), model, optimizer, labeled_dataset_dict
            else:
                return selected_indexes, zip(base,
                                             pert), model, optimizer, labeled_dataset_dict,AL_result_dict
#Basic Workflow: init, select, save + query || load, train (+eval)?, select, save + query|| -> loop...
#With humans: Use save mode, use return both (directly process the outputs via the json maker, and use the indices for the next loop as currently).
def active_learning(data_pool,train_batch_size=16,query_batch_size=1000,initial_query_batch_size=1000,dropout_estimate=False,subsample_size=None,
                    n_query_batches=10,lr=1e-5,from_scratch=False,random_baseline=False,initial_random_baseline=True,cost_weight=1.0,softmax=False,
                    label_noise=0.0,classifier_weights=False,model_handle="roberta-base",merge_mode="features",return_metrics=False,return_dataset=False,
                    labeled_dataset_dict={},AL_train_epochs=1,return_AL_classifier=True,aggregate="mean",dropout_classifier=False,dropout_mode="Majority",
                    overwrite_resamples=False,subsample_once=True,post_train=False,eval_every_step=False):
    model = DualModel(model_handle, merge_mode, dropout_classifier=dropout_classifier, max_length=128).to(
        torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    #Implement data_pool as a dataset with indexes. Query by indexes (for now, have another dataset for these!)
    selected_indexes, model, optimizer, labeled_dataset_dict =\
        init_human_loop(data_pool, model,  lr, initial_random_baseline, dropout_estimate, subsample_size, train_batch_size,
                        cost_weight,classifier_weights,initial_query_batch_size,softmax=softmax,
                        labeled_dataset_dict=labeled_dataset_dict,dropout_mode=dropout_mode,overwrite_resamples=overwrite_resamples,subsample_once=subsample_once)
    for i in tqdm(range(n_query_batches),desc="query_number"):
        print("query_number:" +str(i)+"/"+str(n_query_batches))
        if ((not return_metrics) or i<n_query_batches-1):
            selected_indexes, model, optimizer, labeled_dataset_dict = \
                human_loop(selected_indexes,data_pool[selected_indexes][1][0],data_pool[selected_indexes][1][1],apply_label_noise(data_pool.get_labels(selected_indexes),label_noise),
                           data_pool,random_baseline, dropout_estimate, subsample_size, cost_weight,classifier_weights,query_batch_size,from_scratch=from_scratch,train_batch_size=train_batch_size,
                           model=model,optimizer=optimizer,labeled_dataset_dict=labeled_dataset_dict,softmax=softmax,lr=lr,
                        eval= i==n_query_batches-1 or eval_every_step,no_selection=i==n_query_batches-1,model_handle=model_handle,merge_mode=merge_mode,
                           return_metrics=False,AL_train_epochs=AL_train_epochs,aggregate=aggregate,dropout_classifier=dropout_classifier,dropout_mode=dropout_mode,overwrite_resamples=overwrite_resamples,subsample_once=subsample_once)
        else:
            selected_indexes, model, optimizer, labeled_dataset_dict, AL_result_dict = \
                human_loop(selected_indexes,data_pool[selected_indexes][1][0],data_pool[selected_indexes][1][1],apply_label_noise(data_pool.get_labels(selected_indexes),label_noise),
                           data_pool,random_baseline, dropout_estimate, subsample_size, cost_weight,classifier_weights,query_batch_size,from_scratch=from_scratch,train_batch_size=train_batch_size,
                           model=model,optimizer=optimizer,labeled_dataset_dict=labeled_dataset_dict,softmax=softmax,lr=lr,
                        eval=i==n_query_batches-1 or eval_every_step,no_selection=i==n_query_batches-1,model_handle=model_handle,merge_mode=merge_mode,
                           return_metrics=return_metrics,AL_train_epochs=AL_train_epochs,aggregate=aggregate,dropout_classifier=dropout_classifier,dropout_mode=dropout_mode,overwrite_resamples=overwrite_resamples,subsample_once=subsample_once)
        vote_info = np.array([len(labeled_dataset_dict[key][2]) for key in labeled_dataset_dict.keys()])
        print(np.mean(vote_info))
        print(vote_info[vote_info.argsort()[-25:][::-1]])
    if post_train:
        selected_indexes, model, optimizer, labeled_dataset_dict, AL_result_dict = human_loop([], [], [],
                   [],
                   data_pool, random_baseline, dropout_estimate, subsample_size, cost_weight, classifier_weights,
                   query_batch_size, from_scratch="keep", train_batch_size=train_batch_size,
                   model=model, optimizer=optimizer, labeled_dataset_dict=labeled_dataset_dict, softmax=softmax, lr=lr,
                   eval=i == True, no_selection=True, model_handle=model_handle,
                   merge_mode=merge_mode,
                   return_metrics=return_metrics, AL_train_epochs=1, aggregate=aggregate,
                   dropout_classifier=dropout_classifier, dropout_mode=dropout_mode,
                   overwrite_resamples=overwrite_resamples, subsample_once=subsample_once)


    if not return_metrics:
        if not return_dataset:
            if return_AL_classifier:
                return model
            else:
                return None
        else:
            if return_AL_classifier:
                return model,selected_indexes, labeled_dataset_dict
            else:
                return selected_indexes, labeled_dataset_dict
    else:
        if not return_dataset:
            if return_AL_classifier:
                return model,AL_result_dict
            else:
                return AL_result_dict
        else:
            if return_AL_classifier:
                return model, AL_result_dict,selected_indexes, labeled_dataset_dict
            else:
                return AL_result_dict,selected_indexes, labeled_dataset_dict

def apply_label_noise(labels,noise_level):
    if noise_level>0:
        p = np.random.binomial(1,noise_level,size=labels.shape)
        return (1-p)*labels + p*(1-labels)
    elif noise_level == 0:
        return labels
    else:
        p = np.random.binomial(1, -noise_level, size=labels.shape)
        return labels - p*labels # If p==1 and label==1 flip label, else do nothing.

def train_regularized_filtered(pool,similarity_classifier,lam=5.0,
    epochs=1,mode="logitsL2",batch_size=4,lr=1e-5,reweigh=False,save=None,max_length=256,max_perts=1000000,
    hard_labels=False,return_metrics=False,confidence_threshold=0.5,log_pool_weights=False,return_model=True,eval=True,dual_logits=False,
                               disable_selfloss=False,train_eval_mode=False,eval_every=False,maskwr50=False,maskwr=False):

    print("train_eval_mode:", train_eval_mode)
    if log_pool_weights:
        weightlist = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    if not dual_logits:
        model = RobertaClassifierMultiHead(1).to(device)
    else:
        model = RobertaClassifierMultiHead(1,d_out=2).to(device)
    model = model.to(device)
    if not train_eval_mode:
        model.train()
    else:
        model.eval()
    train_loader = DataLoader(pool.get_train_data_base_task(), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(pool.get_val_data_base_task(), batch_size=batch_size, shuffle=True)
    perturbation_dict = pool.get_dict() 
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    if log_pool_weights:
        pool.set_pool_weights(model,tokenizer)
        weightlist.append(pool.get_pool_weights())
    out_dict = {}
    for epoch in range(epochs):
        losses_main = []
        losses_constraints = []
        losses_constraints_alt = []
        selected = []
        totals = []
        for data,labels in tqdm(train_loader,leave=False,ascii=True,dynamic_ncols=True):
            #labels = labels[:,0].to(device).float()
            labels = labels.to(device).float()
            if hard_labels:
                labels = (labels>=0.5).float()
                if dual_logits:
                    labels = labels.long()
            optimizer.zero_grad()
            regularization_texts = []
            if disable_selfloss:
                mask = []
            for text in data:
                try:
                    perturbations = perturbation_dict[text][:max_perts]
                except:
                    perturbations = []
                if len(perturbations)>0:
                    #print(perturbations)
                    scores = torch.nn.Sigmoid()(similarity_classifier([text for i in range(len(perturbations))], [pert[1] for pert in perturbations])).detach()
                    selected.append((scores>confidence_threshold).sum().detach().cpu().item())
                    totals.append(len(scores))
                    #print(max(scores))
                if len(perturbations)==0 or not torch.any(scores>confidence_threshold):
                    regularization_texts.append(text)
                    if disable_selfloss:
                        mask.append(0.0)
                else:
                    index = np.random.choice(list(range(len(perturbations))),p=(scores>confidence_threshold)[:,0].int().cpu().numpy()/(scores>confidence_threshold)[:,0].cpu().numpy().astype("float64").sum())
                    #print(scores[index])
                    regularization_texts.append(perturbations[index][1])
                    if disable_selfloss:
                        mask.append(1.0)
                #print(mask)
            #for i in range(len(data)):
            #    if type(data[i]) != str:
            #        print(type(data[i]),data[i])
            if maskwr:
                data_new = censor_identity_extended(data,"")
                regularization_texts_new = censor_identity_extended(regularization_texts,"")
                data = data_new
                regularization_texts = regularization_texts_new
            if maskwr50:
                data_new = censor_identity(data,"")
                regularization_texts_new = censor_identity(regularization_texts,"")
                data = data_new
                regularization_texts = regularization_texts_new

            """
            for i in range(len(data)):
                if not data_new[i] == regularization_texts_new[i]:
                    print(type(data_new[i]))
                    print(type(regularization_texts_new[i]))
                    for j,s in enumerate(data_new[i]):
                        if regularization_texts_new[i][j] != s:
                            print(j,s,regularization_texts_new[i][j])
                    print(data[i],regularization_texts[i])
                    print(data_new[i], regularization_texts_new[i])
                    print(len(data_new[i]),len(regularization_texts_new[i]))
                    pointwise = []
                    for j in range(len(data_new[i])):
                        print(j)
                        print(data_new[i][j])
                        print(regularization_texts_new[i][j])
                        if data_new[i][j] == regularization_texts_new[i][j]:
                            pointwise.append(True)
                        else:
                            pointwise.append(False)
                            print("different!",j)
                    print(pointwise)
                    print()
            """


            data = tokenizer(data, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
            perturbations = tokenizer(regularization_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)

            if not dual_logits:
                output = model(data)[:,0]
                if reweigh:
                    loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pool.pos_weights_base_task]).to(device))(output, labels).mean()
                else:
                    loss = torch.nn.BCEWithLogitsLoss()(output,labels).mean()
            else:
                output = model(data)
                if reweigh:
                    frac_pos = 1/(pool.pos_weights_base_task+1)
                    loss = torch.nn.CrossEntropyLoss(torch.tensor([2*frac_pos,2-2*frac_pos]).to(device).float())(output, labels).mean()
                else:
                    loss = torch.nn.CrossEntropyLoss()(output, labels).mean()
            losses_main.append(loss.detach().cpu().item())

            if lam>0.0:
                if not dual_logits:
                    output_perturbed = model(perturbations)[:,0]
                else:
                    output_perturbed = model(perturbations)
                if mode == "logitsCE":
                    similarity_loss = (torch.nn.functional.cross_entropy(output, torch.nn.Softmax(-1)(output_perturbed)) + \
                                      torch.nn.functional.cross_entropy(output_perturbed, torch.nn.Softmax(-1)(output)))/2
                elif mode == "logitsL1":
                    similarity_loss = torch.nn.L1Loss()(output, output_perturbed)
                elif mode == "logitsL2":
                    if not disable_selfloss:
                        similarity_loss = torch.nn.MSELoss()(output, output_perturbed)
                    else:
                        similarity_loss = (torch.nn.MSELoss(reduction="none")(output, output_perturbed)*torch.tensor(mask).to(device)).mean()
                        #print(similarity_loss)
                        #print(torch.nn.MSELoss()(output, output_perturbed))
                elif mode == "argmax":
                    similarity_loss = (torch.nn.CrossEntropyLoss()(output,torch.argmax(output_perturbed.detach(),-1))+
                    torch.nn.CrossEntropyLoss()(output_perturbed, torch.argmax(output.detach(), -1)))/2
                loss = loss + lam * similarity_loss
            loss.backward()
            optimizer.step()
            loss.detach()
            output.detach()
            if lam > 0.0:
                losses_constraints.append(loss.detach().cpu().item() - losses_main[-1])
                losses_constraints_alt.append(lam*similarity_loss.detach().cpu().item())
            #print(np.mean(losses_constraints_alt))
            #print("loss main", np.mean(losses_main))
            #print("loss constraint", np.mean(losses_constraints))
            #print("loss constraint alt", np.mean(losses_constraints_alt))
            if lam > 0.0:
                output_perturbed.detach()
                similarity_loss.detach()
        if not save is None:
            torch.save(model.state_dict(), save)
        #print(losses_main)
        print("loss main",np.mean(losses_main))
        if lam > 0.0:
            print("loss constraint",np.mean(losses_constraints))
            print("loss constraint alt", np.mean(losses_constraints_alt))
        print("negative ratio",np.sum(selected)/np.sum(totals))

        out_dict["loss_main"+str(epoch)] = np.mean(losses_main)
        if lam > 0.0:
            out_dict["loss_constraint"+str(epoch)] = np.mean(losses_constraints)
            out_dict["loss_constraint_alt" + str(epoch)] = np.mean(losses_constraints_alt)
        out_dict["negative ratio"] = np.sum(selected)/np.sum(totals)

        if return_metrics and (epoch==(epochs-1) or eval_every):
            model.eval()
            if log_pool_weights:
                pool.set_pool_weights(model,tokenizer)
                weightlist.append(pool.get_pool_weights())
                print(np.corrcoef(weightlist))
                if len(weightlist) == 2:
                    print(weightlist[0]==weightlist[1])
                    print(weightlist[0])
                #break
            if type(eval) != list:
                if not eval is False:
                    eval = [""]
                else:
                    eval = []
            for path in eval:
                if not path == "":
                    pool.load_eval_labels(path)
                perturbation_dict_eval = pool.get_eval_dict()
                out_dict = eval_robustness(test_loader,perturbation_dict_eval,tokenizer,max_length,max_perts,model,path+eval_every*(str(epoch)),out_dict,dual_logits=dual_logits)
                if eval_every:
                    print(out_dict)
    if not return_metrics:
        if return_model:
            return model
        else:
            return None
    else:
        if return_model:
            return model,out_dict
        else:
            return out_dict


def get_negative_ratio(pool,similarity_classifier,
    batch_size=4,max_perts=1000000,
    confidence_threshold=0.5):
    train_loader = DataLoader(pool.get_train_data_base_task(), batch_size=batch_size, shuffle=True)
    perturbation_dict = pool.get_dict()
    out_dict = {}
    selected = []
    totals = []
    for data,labels in tqdm(train_loader,leave=False,ascii=True,dynamic_ncols=True):
        regularization_texts = []
        for text in data:
            try:
                perturbations = perturbation_dict[text][:max_perts]
            except:
                perturbations = []
            if len(perturbations)>0:
                scores = torch.nn.Sigmoid()(similarity_classifier([text for i in range(len(perturbations))], [pert[1] for pert in perturbations])).detach()
                selected.append((scores>confidence_threshold).sum().detach().cpu().item())
                totals.append(len(scores))
        out_dict["negative ratio"]=np.sum(selected) / np.sum(totals)


    return out_dict


def eval_robustness(test_loader,perturbation_dict_eval,tokenizer,max_length,max_perts,model,path,out_dict,dual_logits=False,
                    maskwr=False,maskwr50=False):
    tp = []
    tn = []
    fp = []
    fn = []
    robust = []
    robust_all = []
    corr = []
    for data, labels in tqdm(test_loader, leave=False, ascii=True, dynamic_ncols=True):
        labels = (labels >= 0.5).float()
        for k, text in enumerate(data):
            perturbations = [text]
            try:
                perturbations += [perturbation_dict_eval[text][i][1] for i in range(len(perturbation_dict_eval[text]))]
            except:
                None
            if maskwr:
                perturbations = censor_identity_extended(perturbations,"")
            if maskwr50:
                perturbations = censor_identity(perturbations,"")

            inputs = tokenizer(perturbations[:max_perts], return_tensors="pt", padding=True, truncation=True,
                               max_length=max_length).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            if not dual_logits:
                outputs = (model(inputs) > 0)[:, 0].detach().int().cpu().numpy()
            else:
                outputs = torch.argmax(model(inputs),-1).detach().cpu().numpy()
            tp.append(outputs[0] * labels[k].numpy())
            tn.append((1 - outputs[0]) * (1 - labels[k].numpy()))
            fp.append(outputs[0] * (1 - labels[k].numpy()))
            fn.append((1 - outputs[0]) * labels[k].numpy())
            corr.append(outputs[0] == labels[k].numpy())
            # print(len(perturbations),perturbations,outputs)
            if len(outputs) > 1:
                robust += list(outputs[0] == outputs[1:])
                robust_all.append((np.all(outputs) or not np.any(outputs)))

    out_dict[path + "tpr_classifier"] = np.sum(tp) / np.sum(tp + fn)
    out_dict[path + "tnr_classifier"] = np.sum(tn) / np.sum(tn + fp)
    out_dict[path + "ba_classifier"] = np.sum(tp) / (2 * np.sum(tp + fn)) + np.sum(tn) / (2 * np.sum(tn + fp))
    out_dict[path + "acc_classifier"] = np.mean(corr)
    out_dict[path + "robust_single"] = np.mean(robust)
    out_dict[path + "robust_all"] = np.mean(robust_all)
    # print("tpr: " + str(np.sum(tp) / (np.sum(tp + fn))))
    # print("tnr: " + str(np.sum(tn) / np.sum(tn + fp)))
    # print("acc: ", np.mean(corr))
    # print("robust_single: ", np.mean(robust))
    # print("robust_all: ", np.mean(robust_all))
    return out_dict

def experiment(data_pool,max_length,kwargs,pool_name="random_contrast"):
    print(kwargs["name"])
    AL_result_dicts = []
    classifier_result_dicts = []
    for i in range(kwargs["iterations"]):
        print(kwargs["name"],i)
        data_pool.reset_pool_weights()
        labeled_dataset_dict = {}
        AL_result_dict = {}
        classifier_result_dict = {}

        if kwargs["classifier_weights"]:
            similarity_classifier = lambda x, y: torch.tensor([0 for i in range(len(x))])
            classifier = train_regularized_filtered(data_pool, similarity_classifier,
                                                    max_length=max_length, batch_size=32, lam=kwargs["lam"], reweigh=True,
                                                    epochs=1, hard_labels=True)
            tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
            if kwargs["classifier_weights"]=="robust":
                mode = "robust"
            else:
                mode = "mse"
            data_pool.set_pool_weights(classifier, tokenizer,mode=mode)
            print(data_pool.pool_weights)
            if kwargs["classifier_weights"] == "robust":
                data_pool.set_pool_weights(classifier, tokenizer, eval=True, mode=mode)
                data_pool.labels_val = np.array([data_pool.labels_val[i] if data_pool.eval_pool_weights[i] else np.nan for i in range(len(data_pool.labels_val))])

        if kwargs["constant_1"]:
            similarity_classifier = lambda x,y: torch.tensor([[1] for i in range(len(x))])
        elif kwargs["constant_0"]:
            print("code modified compared to old runs 0 => -1...")
            similarity_classifier = lambda x,y: torch.tensor([[-1] for i in range(len(x))])
        else:
            try:
                kwargs["tprtnr"]
                similarity_classifier = data_pool.get_classifier(kwargs["tprtnr"][0], kwargs["tprtnr"][1])
            except:
                # We have filtered everything to use at most 64 Roberta token, but BERT's vocabulary is roughly half the size.
                AL_kwargs = deepcopy(kwargs)
                del AL_kwargs["constant_0"]
                del AL_kwargs["constant_1"]
                try:
                    del AL_kwargs["tprtnr"]
                except:
                    None
                del AL_kwargs["name"]
                del AL_kwargs["lam"]
                del AL_kwargs["iterations"]
                del AL_kwargs["confidence_threshold"]
                try:
                    del AL_kwargs["skip_downstream"]
                except:
                    None
                try:
                    del AL_kwargs["downstream_epochs"]
                except:
                    None
                try:
                    del AL_kwargs["downstream_evalmode"]
                except:
                    None
                #if not labels_queried is None:
                #    print(labels_queried)
                if "return_AL_classifier" not in AL_kwargs.keys() or AL_kwargs["return_AL_classifier"]:
                    similarity_classifier,AL_result_dict= \
                        active_learning(data_pool,return_metrics=True,return_dataset=False,
                        labeled_dataset_dict=labeled_dataset_dict,**AL_kwargs)
                else:
                    AL_result_dict = \
                        active_learning(data_pool, return_metrics=True, return_dataset=False,
                                        labeled_dataset_dict = labeled_dataset_dict,
                                        **AL_kwargs)
                AL_result_dicts.append(AL_result_dict)
            #print(eval_active(data_pool.get_train_data(), 16, similarity_classifier, return_metrics=True, dec_thresh=0.5))
        if not kwargs["skip_downstream"]:
            print(kwargs["downstream_epochs"])
            try:
                epochs = kwargs["downstream_epochs"]
            except:
                epochs = 1
            print(epochs)
            classifier_result_dict = train_regularized_filtered(data_pool,similarity_classifier,max_length=max_length,batch_size=32,lam=kwargs["lam"],confidence_threshold=kwargs["confidence_threshold"],
                                                              reweigh=True,epochs=epochs,hard_labels=True,return_metrics=True,return_model=False,train_eval_mode=kwargs["downstream_evalmode"])
            classifier_result_dicts.append(classifier_result_dict)
    print(AL_result_dicts)
    with open(kwargs["name"]+".json","w") as fp:
        json.dump({"Pool_name":pool_name},fp,indent=2)
    with open(kwargs["name"]+".json","w") as fp:
        json.dump(conf_interval_dict(classifier_result_dicts),fp,indent=2)
    with open(kwargs["name"]+".json","a") as fp:
        json.dump(conf_interval_dict(AL_result_dicts),fp,indent=2)
    with open(kwargs["name"]+".json","a") as fp:
        json.dump(kwargs,fp,indent=2)

def conf_interval_dict(result_dicts):
    if len(result_dicts) == 0:
        return {}
    else:
        out_dict = {key:mean_confidence_interval([result_dicts[i][key] for i in range(len(result_dicts))]) for key in result_dicts[0].keys()}
        for key in out_dict.keys():
            print(key,out_dict[key])
        return out_dict