import torch
from Transfer.Models.Models import RobertaClassifierMultiHead
from Transfer.Datasets.Datasets import Kaggle_Toxicity
from Transfer.Datasets.Saliency_utils import attention,gradient_saliency,gradient_x_saliency,mask_attribution,gibbs,mask_tokens,prepare_bart,prepend_labels,get_label_list
from transformers import RobertaTokenizer,BartTokenizer,BartForConditionalGeneration,RobertaForMaskedLM
from transformers.models.bart.modeling_bart import shift_tokens_right
from torch.utils.data import DataLoader,Subset
import numpy as np
from tqdm import tqdm

def train_roberta_multi(batch_size=8,max_length=256, lr=1e-5, epochs=3, save="Roberta_test",model_handle="roberta-base",
                        reweigh=True,freeze_shared=False,label_subset=np.arange(31),loss_subset=None,
                        test_split=10,reverse_split=False,random_mask=False,eval_only=False):
    dataset = Kaggle_Toxicity(label_subset=label_subset,max_length=max_length)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = RobertaClassifierMultiHead(len(label_subset),freeze_shared=freeze_shared,model_handle=model_handle).to(device)
    if eval_only:
        model.load_state_dict(torch.load(save))
    tokenizer = RobertaTokenizer.from_pretrained(model_handle)
    print(len(dataset))
    test_split = np.random.RandomState(42).choice(np.arange(len(dataset)),size=len(dataset)//test_split,replace=False)
    print(len(test_split))
    if not reverse_split:
        train_split = np.delete(np.arange(len(dataset)), test_split)
    else:
        train_split = test_split
        test_split = np.delete(np.arange(len(dataset)), train_split)


    train_loader = DataLoader(Subset(dataset,train_split), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(Subset(dataset, test_split), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if reweigh:
        loss_functions = [torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([dataset.pos_weights[i]]).to(device)) for i in range(len(label_subset))]
    else:
        loss_functions = [torch.nn.BCEWithLogitsLoss() for i in range(len(label_subset))]
    if not eval_only:
        for epoch in range(epochs):
            print("epoch: ",epoch)
            model.train()
            t = tqdm(train_loader,dynamic_ncols=True)
            for data, labels in t:
                optimizer.zero_grad()
                data = tokenizer(data, return_tensors="pt",padding=True, truncation=True, max_length=max_length)
                p = torch.rand(1)
                if random_mask:
                    mask = torch.bernoulli(torch.ones_like(data["input_ids"])*p).bool()
                    mask = torch.logical_and(mask,data["input_ids"]!=tokenizer.pad_token_id)
                    mask = torch.logical_and(mask, data["input_ids"] != tokenizer.bos_token_id)
                    mask = torch.logical_and(mask, data["input_ids"] != tokenizer.eos_token_id)
                    data["input_ids"][mask] = tokenizer.mask_token_id
                data = data.to(device)
                labels = labels.to(device)
                output = model(data)
                loss_components = [loss_functions[i](output[:,i],labels[:,i].float()) for i in range(len(label_subset))]
                if loss_subset or type(loss_subset)==int:
                    loss = torch.sum(loss_components[loss_subset])
                else:
                    loss = torch.sum(torch.stack(loss_components))
                loss.backward()
                optimizer.step()
                loss.detach()
                for key in data:
                    data[key].detach()
                labels.detach()
                t.set_description("Batch_loss ${:.4f}".format(loss.item()), refresh=True)
            model.eval()
            if not save is None:
                torch.save(model.state_dict(), save)
    for name in ["train","test"]:
        if name == "train":
            loader = train_loader
        if name == "test":
            loader = test_loader
        print("Testing on "+ name)
        tp = [0 for i in range(len(label_subset))]
        tn = [0 for i in range(len(label_subset))]
        fp = [0 for i in range(len(label_subset))]
        fn = [0 for i in range(len(label_subset))]
        loss_total = [0 for i in range(len(label_subset))]
        with torch.no_grad():
            for data, labels in tqdm(loader,dynamic_ncols=True):
                #We are throwing away the attention weights here?
                data = tokenizer(data, return_tensors="pt",padding=True,truncation=True, max_length=max_length)
                data = data.to(device)
                labels = (labels>0.5).to(device).float()
                output = model(data)
                loss_components = [loss_functions[i](output[:, i], labels[:, i]) for i in range(len(label_subset))]
                prediction = output>0
                for i in range(len(label_subset)):
                    tp[i] += (labels[:,i]*prediction[:,i]).sum()
                    tn[i] += ((1-labels[:,i]) * (~prediction[:,i])).sum()
                    fp[i] += ((1-labels[:,i])*prediction[:,i]).sum()
                    fn[i] += (labels[:, i] * (~prediction[:, i])).sum()
                    loss_total[i] += loss_components[i]
            for i in range(len(label_subset)):
                print(name+" "+dataset.labels[i])
                print("tpr: " + str(tp[i]/(tp[i]+fn[i])))
                print("tnr: " + str(tn[i] / (tn[i] + fp[i])))
                print("loss: " + str(loss_total[i]/(tp[i]+fp[i]+fn[i]+tn[i])))

def train_generator(batch_size=8,max_length=256, lr=1e-5, epochs=3, save="Bart_Unlabeled",p_mask=0.1,mode="Bart",
               lambda_mask=3.0,accumulate_grads=1,
               drop_classifier=None,drop_q=None,drop_mode=None,subsample=1.0,n_labels_drop=24,attention_layer=11):
    #Load dataset first, as loading takes up more RAM than the processed dataset
    assert bool(drop_classifier) == bool(drop_q) and bool(drop_q)==bool(drop_mode)
    assert mode == "Roberta" or subsample == 1.0
    dataset = Kaggle_Toxicity(label_subset=[0,1,2,4,5,6,8,9,10,11,12,13,15,16,17,18,22],max_length=max_length)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if mode == "Bart":
        model = BartForConditionalGeneration.from_pretrained("facebook/bart-large").to(device)
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    elif mode == "Roberta":
        model = RobertaForMaskedLM.from_pretrained("roberta-base").to(device)

    test_split = np.random.RandomState().choice(np.arange(len(dataset)),size=len(dataset)//10,replace=False)
    train_split =  np.delete(np.arange(len(dataset)), test_split)
    train_loader = DataLoader(Subset(dataset,train_split), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(Subset(dataset, test_split), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if not drop_mode is None:
        tokenizer_roberta = RobertaTokenizer.from_pretrained("roberta-base")
        drop_model = RobertaClassifierMultiHead(n_labels_drop).to(device)
        drop_model.load_state_dict(torch.load(drop_classifier))
    lambda_mask = torch.tensor(lambda_mask)
    p_mask = torch.tensor(p_mask)
    for epoch in range(epochs):
        print("train epoch: ",epoch)
        model.train()
        loss_total = 0
        batches_total = 0
        t = tqdm(enumerate(train_loader),total=len(train_loader),dynamic_ncols=True)
        for step,data in t:
            data,labels = data
            label_list = get_label_list(labels)
            if drop_mode is None:
                assert not mode == "Roberta", "non-model-based dropping is not implemented for Roberta yet"
                data = tokenizer(data, return_tensors="pt", truncation=True, max_length=max_length, padding=True).to(
                    device)
                data_in = prepare_bart(data,tokenizer,p_mask,lambda_mask)
                data_in = torch.tensor(data_in).to(device)
            else:
                data = tokenizer_roberta(data, return_tensors="pt", truncation=True, max_length=max_length, padding=True).to(
                    device)
                if drop_mode == "attention":
                    attributions = attention(data["input_ids"], drop_model, batch_size=batch_size,layer=attention_layer).detach()
                elif drop_mode == "grad":
                    attributions = gradient_saliency(data["input_ids"], drop_model, batch_size=batch_size, target_index=None).detach()
                elif drop_mode == "grad_x":
                    attributions = gradient_x_saliency(data["input_ids"], drop_model, batch_size=batch_size,
                                                           target_index=label_list).detach()
                if mode == "Bart":
                    masked_text = [
                        mask_attribution(tokenizer_roberta.convert_ids_to_tokens(data["input_ids"][i]), attributions[i], q=drop_q, delete_repeat=mode=="Bart") for
                        i in range(len(attributions))]
                    data_in = tokenizer(masked_text, return_tensors="pt", truncation=True, max_length=max_length, padding=True).to(device)["input_ids"]
                else:
                    data_in = mask_tokens(data["input_ids"],tokenizer,attributions,q=drop_q,subsample=subsample).to(device)

            if mode == "Bart":
                loss = model(prepend_labels(data_in,label_list,tokenizer), labels=data["input_ids"]).loss
            elif mode == "Roberta":
                targets = torch.clone(data["input_ids"])
                targets[data_in!=tokenizer.mask_token_id] = -100
                if torch.all(targets==-100):
                    #For some reason, this seems to produce a Nan otherwise.
                    loss = 0.0*model(data_in, labels = targets).logits.sum()
                else:
                    loss = model(data_in, labels = targets).loss
                #scaler.scale(loss).backward()
                #scaler.step(optimizer)
                #scaler.update()
            loss = loss/accumulate_grads
            loss.backward()
            if accumulate_grads==1 or step%accumulate_grads==-1:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            loss=loss.detach()
            for key in data:
                data[key]=data[key].detach()
            data_in=data_in.detach()
            if step%1000==1000-1:
                print(step)
                print(tokenizer.decode(data["input_ids"][0].detach(), skip_special_tokens=True))
                print("")
                print(tokenizer.decode(data_in[0].detach(), skip_special_tokens=True))
                print("")
                if mode == "Bart":
                    print(tokenizer.decode(model.generate(prepend_labels(data_in, label_list, tokenizer)[:1], max_length=max_length)[0].detach(),
                                               skip_special_tokens=True))
                else:
                    if subsample != 1.0:
                        data_in = mask_tokens(data["input_ids"], tokenizer, attributions, q=drop_q,subsample=1.0).to(device)
                    generated_ids = gibbs(model, tokenizer, data_in[:1], sample_tokens=True,iterations=100)
                    print(tokenizer.decode(generated_ids[0],skip_special_tokens=True))
            data_in = None
            output = None
            loss_total = loss_total + loss.item()*accumulate_grads
            batches_total = batches_total + 1
            #torch.cuda.empty_cache()
            t.set_description("Batch_loss ${:.4f}".format(loss_total / batches_total), refresh=True)
            data = None
            loss = None

        model.eval()
        if not save is None:
            torch.save(model.state_dict(), save+str(epoch))
        print("train_total_loss: " + str(loss_total / batches_total))

        print("test epoch: ",epoch)
        loss_total = 0
        batches_total = 0
        for step,data in tqdm(enumerate(test_loader),total=len(test_loader),dynamic_ncols=True):
            data,labels = data
            label_list = get_label_list(labels)
            optimizer.zero_grad()
            data = tokenizer(data, return_tensors="pt",truncation=True, max_length=max_length, padding=True).to(device)
            if drop_mode is None:
                data_in = prepare_bart(data, tokenizer, p_mask, lambda_mask)
                data_in = torch.tensor(data_in).to(device)
            else:
                if drop_mode == "attention":
                    attributions = attention(data["input_ids"], drop_model, batch_size=batch_size).detach()
                elif drop_mode == "grad":
                    attributions = gradient_saliency(data["input_ids"], drop_model, batch_size=batch_size,
                                                     target_index=None).detach()
                elif drop_mode == "grad_x":
                    attributions = gradient_x_saliency(data["input_ids"], drop_model, batch_size=batch_size,
                                                           target_index=label_list).detach()
                if mode == "Bart":
                    masked_text = [
                        mask_attribution(tokenizer_roberta.convert_ids_to_tokens(data["input_ids"][i]), attributions[i], q=drop_q,
                             delete_repeat=mode == "Bart") for
                        i in range(len(attributions))]
                    data_in = tokenizer(masked_text, return_tensors="pt", truncation=True, max_length=max_length,
                                        padding=True).to(device)["input_ids"]
                else:
                    data_in = mask_tokens(data["input_ids"], tokenizer, attributions, q=drop_q).to(device)
            if mode == "Bart":
                loss = model(prepend_labels(data_in, label_list, tokenizer), labels=data["input_ids"]).loss
            elif mode == "Roberta":
                targets = torch.clone(data["input_ids"])
                targets[data_in != tokenizer.mask_token_id] = -100
                loss = model(data_in, labels=targets).loss
            if step<10:
                print(tokenizer.decode(data["input_ids"][0].detach(), skip_special_tokens=True))
                print("")
                print(tokenizer.decode(data_in[0].detach(), skip_special_tokens=True))
                print("")
                if mode == "Bart":
                    print(tokenizer.decode(model.generate(prepend_labels(data_in, label_list, tokenizer)[:1], max_length=max_length)[0].detach(),
                                               skip_special_tokens=True))
                else:
                    if subsample != 1.0:
                        data_in = mask_tokens(data["input_ids"], tokenizer, attributions, q=drop_q,subsample=1.0).to(device)
                    generated_ids = gibbs(model, tokenizer, data_in[:1], sample_tokens=True,iterations=100)
                    print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
            loss=loss.detach()
            for key in data:
                data[key]=data[key].detach()
            data_in=data_in.detach()
            loss_total = loss_total + loss.item()
            batches_total = batches_total + 1
        print("test_total_loss: " + str(loss_total / batches_total))
