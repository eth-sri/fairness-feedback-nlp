import torch
import re
import random
import numpy as np
import string
from tqdm import tqdm

def cosine_sim(x,y):
  return np.sum(x*y)/(np.linalg.norm(x)*np.linalg.norm(y))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def mask_attribution(token_list,attributions,q=0.9,delete_repeat=True):
  while token_list[-1] == "<pad>":
    del token_list[-1]
  for i in range(len(token_list)):
    token_list[i] = token_list[i].replace("Ġ"," ")
    if type(q)==float:
      if attributions[i]>torch.quantile(attributions[:len(token_list)][1:-1],q):
        token_list[i] = "<mask>"
    elif q=="mean":
      if attributions[i] > torch.mean(attributions[:len(token_list)][1:-1]):
        token_list[i] = "<mask>"
    elif q=="max":
      if attributions[i] > torch.max(attributions[:len(token_list)][1:-1])/4:
        token_list[i] = "<mask>"
    elif q=="nonzero":
      if attributions[i] > 0:
        token_list[i] = "<mask>"
    elif len(q)>3 and q[:3] == "max" and q[3] == "t":
        if attributions[i] > torch.max(attributions[:len(token_list)][1:-1])/4 and attributions[i] > float(q[4:]):
          token_list[i] = "<mask>"
    else:
      raise Exception("q should be float or 'max'")
  if delete_repeat:
    token_list = [v for i, v in enumerate(token_list) if i == 0 or v != token_list[i-1] or v != "<mask>"]
  return "".join(token_list[1:-1])

def mask_tokens(token_list,tokenizer,attributions,q=0.9,subsample=1.0):
  out = []
  for j in range(len(token_list)):
    padded = (token_list[j] == tokenizer.pad_token_id).sum()
    tokens = torch.clone(token_list[j])
    for i in range(len(tokens)-padded):
      if type(q) == float:
        if attributions[j,i]>torch.quantile(attributions[j,:len(tokens)-padded][1:-1],q):
          if torch.rand(1)>(1-subsample):
            tokens[i]  = tokenizer.mask_token_id
      elif q=="mean":
        if attributions[j,i]>torch.mean(attributions[j,:len(tokens)-padded][1:-1]):
          if torch.rand(1) > (1-subsample):
            tokens[i]  = tokenizer.mask_token_id
      elif q=="max":
        if attributions[j,i]>torch.max(attributions[j,:len(tokens)-padded][1:-1])/4:
          if torch.rand(1) > (1-subsample):
            tokens[i]  = tokenizer.mask_token_id
      elif q=="nonzero":
        if attributions[j,i]>0:
          if torch.rand(1) > (1-subsample):
            tokens[i]  = tokenizer.mask_token_id
      elif len(q)>3 and q[:3] == "max" and q[3] == "t":
        if attributions[j,i]>torch.max(attributions[j,:len(tokens)-padded][1:-1])/4 and attributions[j,i]>float(q[4:]):
          if torch.rand(1) > (1-subsample):
            tokens[i]  = tokenizer.mask_token_id
      else:
        raise Exception("q should be float or 'mean'")
    out.append(tokens)
  return torch.stack(out)



def gibbs(model,tokenizer, input_tokens,iterations=25,sample_tokens=True):
  mask = (input_tokens == tokenizer.mask_token_id)
  if sample_tokens:
    input_tokens = input_tokens*(~mask) + torch.distributions.categorical.Categorical(logits=model(input_tokens).logits).sample()*mask
  else:
    input_tokens = input_tokens*(~mask) + torch.argmax(model(input_tokens).logits,-1)*mask
  for iteration in range(iterations):
    masked_indices = torch.distributions.categorical.Categorical(probs=mask).sample()
    new_mask = torch.zeros(mask.shape,dtype=bool).to(mask.device)
    for i,j in enumerate(masked_indices):
      new_mask[i,j]=1
      input_tokens[i,j] = tokenizer.mask_token_id
    if sample_tokens:
      input_tokens = input_tokens*(~new_mask) + torch.distributions.categorical.Categorical(logits=model(input_tokens).logits).sample()*new_mask
    else:
      input_tokens= input_tokens*(~new_mask) + torch.argmax(model(input_tokens).logits,-1)*new_mask
  return input_tokens

def drop_token(data,model,batch_size=2,target_index=0,ord=None,use_abs=True):
  data_full = []
  index_map = []
  if not type(target_index)==int:
    target_index_full = []
  for i in range(len(data)):
    indexes = []
    data_full.append(data[i])
    if not type(target_index) == int:
      target_index_full.append(target_index[i])
    indexes.append(len(data_full)-1)
    for j in range(len(data[0])):
      if data[i,j] == 1:
        break
      else:
        dropped = torch.clone(data[i])
        dropped[j] = 50264
        data_full.append(dropped)
        if not type(target_index) == int:
          target_index_full.append(target_index[i])
        indexes.append(len(data_full)-1)
    index_map.append(indexes)
  data_full = torch.stack(data_full)

  scores = []
  i = 0
  while i < len(data_full):
    j=i+batch_size
    if type(target_index) is int:
      scores.append(model({"input_ids":data_full[i:j].to(device),"attention_mask":torch.ones_like(data_full[i:j]).to(device)})[:,target_index].detach())
    else:
      scores.append(model({"input_ids": data_full[i:j].to(device),
                           "attention_mask": torch.ones_like(data_full[i:j]).to(device)})[list(range(len(target_index_full[i:j]))), target_index_full[i:j]].detach())
    i=j
  scores=torch.concat(scores)
  if use_abs:
    out = [torch.abs(scores[indices[1]:indices[-1]+1]-scores[indices[0]]) for indices in index_map]
  else:
    out = [-(scores[indices[1]:indices[-1] + 1] - scores[indices[0]]) for indices in index_map]
  for i in range(len(out)):
    out[i] = torch.concat([out[i],torch.zeros(data.shape[1]-len(out[i])).to(device)])
  return torch.stack(out)

def drop_thresh(data,model,batch_size=2,target_index=0,hard_threshold=0.25,tol=1e-4,reverse=False):
  data = data.to(device)
  data_new = torch.clone(data)
  if type(target_index) is int:
    ps = torch.nn.Sigmoid()(model({"input_ids":data,"attention_mask":torch.ones_like(data)})[:,target_index])
  else:
    ps = torch.nn.Sigmoid()(model({"input_ids": data, "attention_mask": torch.ones_like(data)})[range(len(target_index)), target_index])

  attributions = torch.ones_like(data)
  while ((torch.any(ps>hard_threshold) and not reverse) or (torch.any((1-ps)>hard_threshold) and reverse)) and torch.max(attributions)>tol:
    if not reverse:
      attributions = drop_token(data_new[(ps>hard_threshold)],model,batch_size=batch_size,target_index=target_index,use_abs=False)
    else:
      attributions = -drop_token(data_new[(1-ps > hard_threshold)], model, batch_size=batch_size,
                                target_index=target_index, use_abs=False)
    if not reverse:
      data_new[(ps>hard_threshold),torch.argmax(attributions,-1)] = 50264
    else:
      data_new[((1-ps) > hard_threshold), torch.argmax(attributions, -1)] = 50264
    if type(target_index) is int:
      ps_new = torch.nn.Sigmoid()(model({"input_ids":data_new,"attention_mask":torch.ones_like(data_new)})[:,target_index])
    else:
      ps_new = torch.nn.Sigmoid()(
        model({"input_ids": data_new, "attention_mask": torch.ones_like(data_new)})[range(len(target_index)), target_index])
    ps = ps_new
  return (data_new == 50264).float()

def drop_token_iterative(data,model,batch_size=2,target_index=0,ord=None,steps=100):
  with torch.no_grad():
    data_c = torch.clone(data)
    out_final = torch.zeros_like(data).float()
    for k in range(steps):
      data_full = []
      index_map = []
      for i in range(len(data)):
        indexes = []
        data_full.append(data_c[i])
        indexes.append(len(data_full)-1)
        for j in range(len(data[0])):
          if data[i,j] == 1:
            break
          else:
            dropped = torch.clone(data_c[i])
            dropped[j] = 50264
            data_full.append(dropped)
            indexes.append(len(data_full)-1)
        index_map.append(indexes)
      data_full = torch.stack(data_full)

      scores = []
      i = 0
      while i < len(data_full):
        j=i+batch_size
        if type(target_index) is int:
          scores.append(model({"input_ids":data_full[i:j].to(device),"attention_mask":torch.ones_like(data_full[i:j]).to(device)})[:,target_index].detach())
        else:
          scores.append(model({"input_ids": data_full[i:j].to(device),
                               "attention_mask": torch.ones_like(data_full[i:j]).to(device)})[range(len(target_index)), target_index].detach())

        i=j
      scores=torch.concat(scores)
      out = [torch.abs(scores[indices[1]:indices[-1]+1]-scores[indices[0]]) for indices in index_map]
      for i in range(len(out)):
        out[i] = torch.concat([out[i],torch.zeros(data.shape[1]-len(out[i])).to(device)])
        data_c[i][torch.argmax(out[i])] = 50264
        out_final[i][torch.argmax(out[i])] = out[i][torch.argmax(out[i])]
  return out_final

def attention(data,model,batch_size=2,layer=11):
  i = 0
  atts = []
  while i < len(data):
    j=i+batch_size
    atts.append(torch.max(model.get_attentions({"input_ids":data[i:j].to(device),"attention_mask":torch.ones_like(data[i:j]).to(device)},layer=layer),1).values.mean(1)[:,:].detach()) #could as well be mean over 2...
    i=j
  return torch.concat(atts)

def bound(pattern):
  return r'\b{}[s]?\b'.format(pattern)

def check_word_list(texts,target_index=0,replace_index=None,delete_repeat=True,replace_missing_neutral=False,mask=True,bar=False,list50=False,generator=None):
  LABELS = ['male', 'female', 'transgender', 'other_gender', 'heterosexual', 'homosexual_gay_or_lesbian',
         'bisexual', 'other_sexual_orientation', 'christian', 'jewish', 'muslim', 'hindu',
         'buddhist', 'atheist', 'other_religion', 'black', 'white', 'asian', 'latino',
         'other_race_or_ethnicity', 'physical_disability', 'intellectual_or_learning_disability',
         'psychiatric_or_mental_illness', 'other_disability','target','severe_toxicity','obscene','identity_attack'
         ,'insult','threat','sexual_explicit',"neutral"]
  if generator is None:
    generator = np.random
  #Words are ordered to have superstrings (such as "grandfather")to appear before their substrings ("father")  in order to have the longer string replaced if applicable
  if not list50:
    NOUNS ={"male":
              ["man", "men","grandfather", "bro","guy", "boy","father", "dad",
       "son","husbands","husband", "grandpa","brother"],
            "female":
          ["woman", "women", "grandmother","lady", "ladies", "girl", "mother", "mom",
           "daughter", "wives","wife","grandma", "sister"],
            "none":[ "individual","person", "people","kid","children",
                    "child", "spouse", "grandparent", "sibling", "veteran","parent"]
            }
    PRONOUNS ={"male":
              ["himself","his","him","he"],
            "female":
          ["herself","hers","her","she"],
            "none":["themselves","they","them","their"]
            }
    ADJECTIVES = {"male":["male","manly","masculine"],
                  "female":["female","pregnant","feminine","femme","womanly"],
                  "transgender":["transsexual","FTM","F2M","MTF", "M2F","transgender","trans female","trans feminine",
                                 "trans femme","trans fem","trans male","trans masculine","trans masc","trans"],
                  "heterosexual": ["straight", "heterosexual", "hetero"],
                  "homosexual_gay_or_lesbian": ["gay", "homosexual", "lesbian"],
                  "bisexual":["bisexual","bi"],
                  "christian": ["Amish", "Catholic", "Christian", "Evangelical", "Lutheran", "Mormon", "Protestant",
                                "Unitarian"],
                  "jewish": ["Conservative Jewish", "Orthodox Jewish", "Reform Jewish", "Hasidic", "Jewish"],
                  "muslim": ["Muslim", "Shia", "Sunni"],
                  "hindu": ["Hindu"],
                  "buddhist": ["Buddhist"],
                  "atheist": ["atheist", "irreligious"],
                  "black": ["African-American", "African", "Black", "BIPOC"],
                  "white": ["European-American", "European", "Caucasian", "White"],
                  "asian": ["Chinese-American", "American-Born Chinese", "Chinese", "Indian-American", "Indian",
                            "Korean-American",
                            "Korean", "Vietnamese-American", "Vietnamese", "AAPI"],
                  "latino":["Cuban-American","Cuban","Dominican-American","Dominican","Salvadoran-American","Salvadoran",
                            "Guatemalan-American","Guatemalan","Mexican-American","Mexican","Filipina-American","Filipina",
                            "Filipino-American","Filipino","Hispanic","Latinx","Latine","Latino","Latina","Latin American"]}
  else:
    PRONOUNS = {}
    NOUNS = {}
    ADJECTIVES = {"male": ["male"],
                  "female": ["female"],
                  "transgender": ["transgender","trans"],
                  "heterosexual": ["straight", "heterosexual"],
                  "homosexual_gay_or_lesbian": ["gay", "homosexual", "lesbian"],
                  "bisexual": ["bisexual"],
                  "christian": ["Christian", "Catholic", "Protestant"],
                  "jewish": ["Jewish"],
                  "muslim": ["Muslim"],
                  "buddhist": ["Buddhist"],
                  "black": ["African American", "African", "Black"],
                  "white": ["European", "Caucasian", "White"],
                  "asian": ["Japanese","Chinese","Asian"],
                  "latino": ["Hispanic","Latino", "Latina","Latinx","Mexican"]}

  if mask:
    texts_masked = []
  if not replace_index is None:
    texts_out = []
  if not bar:
    t = range(len(texts))
  else:
    t = tqdm(range(len(texts)),total=len(texts),desc="word_replacement",dynamic_ncols=True)
  for i in t:
    if mask:
      text_temp = texts[i]
      if type(target_index) is int:
        target_index_round = target_index
      else:
        target_index_round = target_index[i]
      if LABELS[target_index_round] in ADJECTIVES:
        terms = list(set([term for term in ADJECTIVES[LABELS[target_index_round]]
                          if re.search(bound(term), text_temp, flags=re.IGNORECASE)]))
        for term in terms:
          text_temp = re.sub(bound(term), "<mask>", text_temp, flags=re.IGNORECASE)
      if LABELS[target_index_round] in NOUNS:
        terms = list(set([term for term in NOUNS[LABELS[target_index_round]]
                          if re.search(bound(term), text_temp, flags=re.IGNORECASE)]))
        for term in terms:
          text_temp = re.sub(bound(term), "<mask>", text_temp, flags=re.IGNORECASE)
      if LABELS[target_index_round] in PRONOUNS:
        terms = list(set([term for term in NOUNS[LABELS[target_index_round]]
                          if re.search(bound(term), text_temp, flags=re.IGNORECASE)]))
        for term in terms:
          text_temp = re.sub(bound(term), "<mask>", text_temp, flags=re.IGNORECASE)

      text_temp = re.sub("(?:<mask>+\s*)+", "<mask> ", text_temp, flags=re.IGNORECASE) #Remove duplicate masks
      texts_masked.append(text_temp)
    if not replace_index is None:
      if type(replace_index) is int:
        replace_index_round = replace_index
      else:
        replace_index_round = replace_index[i]
      if type(target_index) is int:
        target_index_round = target_index
      else:
        target_index_round = target_index[i]
      text_temp = texts[i]
      if LABELS[target_index_round] in ADJECTIVES:
        terms = list(set([term for term in ADJECTIVES[LABELS[target_index_round]]
                          if re.search(bound(term), text_temp, flags=re.IGNORECASE)]))
        for term in terms:
          if LABELS[replace_index_round] in ADJECTIVES:
            text_temp = re.sub(bound(term), generator.choice(ADJECTIVES[LABELS[replace_index_round]]), text_temp, flags=re.IGNORECASE)
      if LABELS[target_index_round] in NOUNS:
        terms = list(set([term for term in NOUNS[LABELS[target_index_round]]
                          if re.search(bound(term), text_temp, flags=re.IGNORECASE)]))
        for term in terms:
          if LABELS[replace_index_round] in NOUNS:
            text_temp = re.sub(bound(term), generator.choice(NOUNS[LABELS[replace_index_round]]), text_temp,
                               flags=re.IGNORECASE)
          else:
            if replace_missing_neutral:
              text_temp = re.sub(bound(term), generator.choice(NOUNS["none"]), text_temp,
                                 flags=re.IGNORECASE)
            else:
              text_temp = re.sub(bound(term), generator.choice(ADJECTIVES[LABELS[replace_index_round]]), text_temp,
                                 flags=re.IGNORECASE)
      """
      if LABELS[target_index_round] in PRONOUNS:
        terms = list(set([term for term in PRONOUNS[LABELS[target_index_round]]
                          if re.search(bound(term), text_temp, flags=re.IGNORECASE)]))
        print(terms)
        for term in terms:
          if LABELS[replace_index_round] in PRONOUNS:
            text_temp = re.sub(bound(term), generator.choice(PRONOUNS[LABELS[replace_index_round]]), text_temp,
                               flags=re.IGNORECASE)
          else:
            text_temp = re.sub(bound(term), generator.choice(PRONOUNS["none"]), text_temp,
                               flags=re.IGNORECASE)
      """ #This was broken during the generation of the pool (called NOUNS instead of prononouns in the first line after the if statement).
      # As Nouns was filtered in a previous step, this should not have caused any additional damage.
      if delete_repeat:
        text_temp = re.sub("(?:<mask>+\s*)+", "<mask> ", text_temp, flags=re.IGNORECASE) #Remove duplicate masks
      texts_out.append(text_temp)
  if not replace_index is None:
    if mask:
      return texts_masked,texts_out
    else:
      return texts_out
  else:
    return texts_masked

def mask_word_vectors(texts,model,vector_dict,target_index, q, delete_repeat=True):
  LABELS = ['male', 'female', 'transgender', 'other_gender', 'heterosexual', 'homosexual_gay_or_lesbian',
            'bisexual', 'other_sexual_orientation', 'christian', 'jewish', 'muslim', 'hindu',
            'buddhist', 'atheist', 'other_religion', 'black', 'white', 'asian', 'latino',
            'other_race_or_ethnicity', 'physical_disability', 'intellectual_or_learning_disability',
            'psychiatric_or_mental_illness', 'other_disability', 'target', 'severe_toxicity', 'obscene',
            'identity_attack'
    , 'insult', 'threat', 'sexual_explicit', "neutral"]
  out = []
  for text in texts:
    if type(target_index) is int:
      target_index_round = target_index
    else:
      target_index_round = target_index[i]
    word_list = text.split(" ")
    modded_word_list = [word.translate(str.maketrans('', '', string.punctuation)) for word in word_list]
    attributions = [cosine_sim(model[word.lower()],vector_dict[LABELS[target_index_round]]) if (word.lower() in model) else np.nan for word in modded_word_list]
    for i in range(len(word_list)):
      if type(q) == float:
        if attributions[i] > np.nanquantile(attributions, q):
          word_list[i] = "<mask>"
      elif q == "mean":
        if attributions[i] > np.nanmean(attributions):
          word_list[i] = "<mask>"
      elif q == "max":
        if attributions[i] > np.nanmax(attributions) / 4:
          word_list[i] = "<mask>"
      elif q == "nonzero":
        if attributions[i] > 0:
          word_list[i] = "<mask>"
      elif len(q) > 3 and q[:3] == "max" and q[3] == "t":
        if attributions[i] > np.nanmax(attributions) / 4 and attributions[i]>float(q[4:]):
          word_list[i] = "<mask>"
      else:
        raise Exception("q should be float or 'max'")

    if delete_repeat:
      word_list = [v for i, v in enumerate(word_list) if i == 0 or v != word_list[i - 1] or v != "<mask>"]
    out.append(" ".join(word_list))
  return out





def gradient_saliency(data,model,batch_size=2,target_index=0):
  i = 0
  sals = []
  while i < len(data):
    j=i+batch_size
    embedds = model.Roberta.embeddings(data[i:j].to(device)).detach()
    with torch.enable_grad():
        embedds.requires_grad = True
    loss = model({"input_ids":data[i:j].to(device),"attention_mask":torch.ones_like(data[i:j]).to(device)},embeds=embedds)[:,target_index].sum()
    loss.backward()
    with torch.no_grad():
        g = embedds.grad
    sals.append(torch.linalg.norm(g,ord=2,dim=-1).detach())
    i=j
  return torch.concat(sals)

def gradient_x_saliency(data,model,batch_size=2,target_index=0):
  i = 0
  sals = []
  while i < len(data):
    j=i+batch_size
    embedds = model.Roberta.embeddings(data[i:j].to(device)).detach()
    with torch.enable_grad():
        embedds.requires_grad = True
    if type(target_index) == int:
      loss = model({"input_ids":data[i:j].to(device),"attention_mask":torch.ones_like(data[i:j]).to(device)},embeds=embedds)[:,target_index].sum()
    else:
      target_index_current = [target_index[i+k] if not target_index[i+k] is None else 0 for k in range(len(data[i:j]))]
      loss = model({"input_ids": data[i:j].to(device), "attention_mask": torch.ones_like(data[i:j]).to(device)},
                   embeds=embedds)[range(len(target_index_current)), target_index_current].sum()
    loss.backward()
    with torch.no_grad():
        g = embedds.grad
    sals.append(torch.linalg.norm(g*embedds,ord=2,dim=-1).detach())
    i=j
  return torch.concat(sals)

def prepend_lm_saliency(texts,model,tokenizer,target_label,contrast="mean",censor_encoder=True):
  with torch.no_grad():
    labels = tokenizer(texts,return_tensors="pt")["input_ids"].to(device)
    if censor_encoder:
      text_in = [text[:0] for text in texts]
    else:
      text_in = texts
    baseline = torch.nn.LogSoftmax(-1)(model(prepend_labels(tokenizer(text_in,return_tensors="pt")["input_ids"].to(device),[target_label],tokenizer),labels=labels).logits[0])[torch.arange(len(labels)),labels]
    if contrast == "mean":
      return baseline - torch.mean(torch.stack([torch.nn.LogSoftmax(-1)(model(prepend_labels(tokenizer(text_in,return_tensors="pt")["input_ids"].to(device),[i],tokenizer),labels=labels).logits[0])[torch.arange(len(labels)),labels] for i in range(16)]),0)
    elif contrast is None:
      return baseline -  torch.nn.LogSoftmax(-1)(model(prepend_labels(tokenizer(text_in,return_tensors="pt")["input_ids"].to(device),[None],tokenizer),labels=labels).logits[0])[torch.arange(len(labels)),labels]
    else:
      return baseline -  torch.nn.LogSoftmax(-1)(model(prepend_labels(tokenizer(text_in,return_tensors="pt")["input_ids"].to(device),[contrast],tokenizer),labels=labels).logits[0])[torch.arange(len(labels)),labels]

def prepare_bart(data,tokenizer,p_mask,lambda_mask):
  data_in = []
  for i in range(len(data["input_ids"])):
    base = []
    sentence_in = data["input_ids"][i].detach().cpu().numpy()
    buffer = 0
    for j in range(len(sentence_in)):
      if sentence_in[j] == tokenizer.eos_token_id or sentence_in[j] == tokenizer.bos_token_id:
        # Always add special eos tokens.
        base.append(sentence_in[j])
        buffer = 0
      elif sentence_in[j] == tokenizer.pad_token_id:
        # Don't pad right now, as we need to adjust the length in the end anyways.
        continue
      else:
        if buffer > 0:
          # If skips are in the buffer, skip next token
          buffer -= 1
          continue
        else:
          if torch.bernoulli(p_mask).detach() == 0:
            # Just add the token
            base.append(sentence_in[j])
          else:
            p = torch.poisson(lambda_mask).detach()
            if p == 0:
              base.append(sentence_in[j])
              base.append(tokenizer.mask_token_id)
              # Add additional mask after if p=0
            elif p == 1:
              base.append(tokenizer.mask_token_id)
            else:
              base.append(tokenizer.mask_token_id)
              buffer = p - 1
              # Add to skip buffer if p>1
    while len(base) < len(sentence_in):
      # Pad if too short
      base.append(tokenizer.pad_token_id)
    while len(base) > len(sentence_in):
      # Delete from the back (but not the eos token) if too long
      del base[-2]
    data_in.append(base)
  return data_in

def get_label_list(labels):
  label_list = []
  for i in range(len(labels)):
    if sum(labels[i]) > 0:
      label_list.append(torch.distributions.categorical.Categorical(labels[i]).sample())
    else:
      label_list.append(None)
  return label_list

def prepend_labels(data,label_list,tokenizer,all_labels=False):
  if not all_labels:
    label_names = ['male', 'female', 'trans', 'hetero', 'gay', 'bisexual',
     'christian', 'jewish', 'muslim', 'hindu', 'buddhist', 'atheist',
     'black', 'white', 'asian', 'latin', 'mental']
  else:
    label_names =['male', 'female', 'transgender', 'other_gender', 'heterosexual', 'homosexual_gay_or_lesbian',
     'bisexual', 'other_sexual_orientation', 'christian', 'jewish', 'muslim', 'hindu',
     'buddhist', 'atheist', 'other_religion', 'black', 'white', 'asian', 'latino',
     'other_race_or_ethnicity', 'physical_disability', 'intellectual_or_learning_disability',
     'psychiatric_or_mental_illness', 'other_disability']
  max_token_len = 5
  data_out = []
  for i in range(len(data)):
    if not label_list[i] is None:
      label_tokens = tokenizer(label_names[label_list[i]],return_tensors="pt")["input_ids"].to(data.device)[0]
    else:
      label_tokens = tokenizer("no label", return_tensors="pt")["input_ids"].to(data.device)[0]
    data_out.append(torch.concat([label_tokens,data[i][1:],torch.tensor([tokenizer.pad_token_id for k in range(max_token_len-len(label_tokens))]).to(data.device)]))
  return torch.stack(data_out).long()



