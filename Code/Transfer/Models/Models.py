import torch
from transformers import RobertaModel,RobertaTokenizer,BertModel,BertTokenizer,DebertaTokenizer,DebertaModel
import copy

class DualModel(torch.nn.Module):
    def __init__(self, model_handle="roberta-base",mode="merge",max_length=64,dropout_classifier=False,freeze=False):
        super(DualModel, self).__init__()
        if model_handle == "roberta-base":
            self.model = RobertaModel.from_pretrained(model_handle)
            self.tokenizer = RobertaTokenizer.from_pretrained(model_handle)
        elif model_handle in ["bert-base-uncased","bert-large-uncased"]:
            self.model = BertModel.from_pretrained(model_handle)
            self.tokenizer = BertTokenizer.from_pretrained(model_handle)
        elif model_handle == "microsoft/deberta-base":
            self.model = DebertaModel.from_pretrained(model_handle)
            self.tokenizer = DebertaTokenizer.from_pretrained(model_handle)
        if not mode == "bilinear":
            if not model_handle == "bert-large-uncased":
                self.output = torch.nn.Linear(768,1)
                self.pooler = Pooler()
            else:
                self.output = torch.nn.Linear(1024,1)
                self.pooler = Pooler(1024)
        else:
            if not model_handle == "bert-large-uncased":
                self.register_parameter(name='output', param=torch.nn.Parameter(torch.randn(768,768)))
                self.pooler = Pooler(dropout=dropout_classifier)
            else:
                self.register_parameter(name='output', param=torch.nn.Parameter(torch.randn(1024, 1024)))
                self.pooler = Pooler(1024,dropout=dropout_classifier)
        if dropout_classifier:
            self.dropout = torch.nn.Dropout(p=0.1)
        else:
            self.dropout = torch.nn.Identity()


        self.mode = mode
        self.max_length=max_length
        self.freeze = freeze
    def forward(self, input_1,input_2,base_only=False):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if self.mode != "merge":
            input_1 = self.tokenizer(list(input_1), max_length=self.max_length, return_tensors="pt", padding=True, truncation=True)[
                "input_ids"].to(device).detach()
            input_2 = self.tokenizer(list(input_2), max_length=self.max_length, return_tensors="pt", padding=True, truncation=True)[
                "input_ids"].to(device).detach()
            if base_only:
                assert self.mode == "concat", "This dropout mode is currently only available when model mode = concat"
                return self.model(torch.concat([input_1, input_2], dim=-1)).last_hidden_state
            if self.mode=="features":
                return self.output(self.dropout(self.pooler(self.model(input_1).last_hidden_state)-self.pooler(self.model(input_2).last_hidden_state)))
            elif self.mode == "concat":
                if self.freeze:
                    embeds = self.model(torch.concat([input_1,input_2],dim=-1)).last_hidden_state.detach()
                    return self.output(self.dropout(self.pooler(embeds)))
                else:
                    return self.output(self.dropout(self.pooler(self.model(torch.concat([input_1,input_2],dim=-1)).last_hidden_state)))
            elif self.mode == "bilinear":
                feature_diff = self.pooler(self.dropout(self.model(input_1).last_hidden_state) - self.pooler(self.model(input_2).last_hidden_state))
                return ((feature_diff@self.output)*feature_diff).sum(-1,keepdim=True)
        elif self.mode == "merge":
            input = self.tokenizer([input_1[i]+" "+self.tokenizer.sep_token+" "+input_2[i] for i in range(len(input_1))], max_length=2*self.max_length+3, return_tensors="pt", padding=True,
                                     truncation=True)[
                "input_ids"].to(device).detach()
            return self.output(self.dropout(self.pooler(self.model(input).last_hidden_state)))


class Pooler(torch.nn.Module):
    def __init__(self,size=768,dropout=False):
        super().__init__()
        self.dense = torch.nn.Linear(size,size)
        self.activation = torch.nn.Tanh()
        if dropout:
            self.dropout = torch.nn.Dropout(p=0.1)
        else:
            self.dropout = torch.nn.Identity()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = self.dropout(hidden_states[:, 0])
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output



class RobertaClassifierMultiHead(torch.nn.Module):
    def __init__(self,n_heads,freeze_shared=False,model_handle="roberta-base",d_out=1):
        super(RobertaClassifierMultiHead, self).__init__()
        self.n_heads = n_heads
        self.Roberta = RobertaModel.from_pretrained(model_handle)
        self.poolers = [self.Roberta.pooler]
        for i in range(self.n_heads-1):
            self.poolers.append(copy.deepcopy(self.poolers[-1]))
        if model_handle == "roberta-base":
            self.outs = torch.nn.ModuleList([torch.nn.Linear(768,d_out) for i in range(self.n_heads)])
        else:
            self.outs = torch.nn.ModuleList([torch.nn.Linear(1024, d_out) for i in range(self.n_heads)])
        self.poolers = torch.nn.ModuleList(self.poolers)
        self.freeze_shared = freeze_shared
    def forward(self,inputs,embeds=None,subset = None):
        input = inputs["input_ids"]
        attention = inputs["attention_mask"]
        if subset is None:
            subset = list(range(self.n_heads))
        if embeds is None:
            embeds = self.Roberta.embeddings(input)
        for i in range(len(self.Roberta.encoder.layer)):
            embeds = self.Roberta.encoder.layer[i](embeds,encoder_attention_mask=attention)[0]
        if self.freeze_shared:
            embeds = embeds.detach()
        outputs = []
        for i in subset:
            outputs.append(self.outs[i](self.poolers[i]((embeds))))
        return torch.concat(outputs,-1)

    def get_attentions(self,inputs,layer=11):
        input = inputs["input_ids"]
        attention = inputs["attention_mask"]
        embeds = self.Roberta.embeddings(input)
        for i in range(layer):
            embeds = self.Roberta.encoder.layer[i](embeds,encoder_attention_mask=attention)[0]
        return self.Roberta.encoder.layer[layer].attention(embeds,encoder_attention_mask=attention,output_attentions=True)[1]