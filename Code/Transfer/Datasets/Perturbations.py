import torch
from random import shuffle,choice,randint
import re
import numpy as np
from tqdm import tqdm

IDENTITY_TERMS = ["queer","bisexual","homosexual","gay","transgender","sikh","blind","lgbt","female","black","hispanic","male","young","jewish","elderly","millenial","taoist","deaf","latinx","paralyzed","latino","lgbtq","heterosexual","nonbinary","latina","white","middle aged","trans","christian","catholic","protestant","younger","older","old","straight","mexican","african american", "american","african","indian","asian","canadian","middle eastern","european","buddhist","japanese","chinese","teenage","lesbian","muslim"]
LIST_2_TOKEN = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 19, 20, 22, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 38, 39, 40, 41, 43, 44, 45, 46, 47, 48, 49]
IDENTITY_TERMS_EXTEND = ["man", "men", "grandfather", "bro", "guy", "boy", "father", "dad", "son", "husbands", "husband", "grandpa", "brother",
"woman", "women", "grandmother", "lady", "ladies", "girl", "mother", "mom","daughter", "wives", "wife", "grandma", "sister","individual", "person", "people", "kid", "children",
            "child", "spouse", "grandparent", "sibling", "veteran", "parent","himself", "his", "him", "he","herself", "hers", "her", "she",
            "themselves", "they", "them", "their","male", "manly", "masculine","female", "pregnant", "feminine", "femme", "womanly","transsexual",
            "FTM", "F2M", "MTF", "M2F", "transgender", "trans female", "trans feminine","trans femme", "trans fem", "trans male", "trans masculine",
            "trans masc", "trans","straight", "heterosexual", "hetero", "gay", "homosexual", "lesbian","bisexual", "bi","Amish", "Catholic", "Christian",
            "Evangelical", "Lutheran", "Mormon", "Protestant", "Unitarian","Conservative Jewish", "Orthodox Jewish", "Reform Jewish", "Hasidic", "Jewish",
            "Muslim", "Shia", "Sunni","Hindu","Buddhist", "atheist", "irreligious","African-American", "African", "Black", "BIPOC",
            "European-American", "European", "Caucasian", "White","Chinese-American", "American-Born Chinese", "Chinese", "Indian-American", "Indian",
            "Korean-American","Korean", "Vietnamese-American", "Vietnamese", "AAPI","Cuban-American", "Cuban", "Dominican-American", "Dominican",
            "Salvadoran-American","Salvadoran","Guatemalan-American", "Guatemalan", "Mexican-American", "Mexican", "Filipina-American","Filipina",
            "Filipino-American", "Filipino", "Hispanic", "Latinx", "Latine", "Latino", "Latina","Latin American"]

def bound(pattern):
    return r'\b{}[s]?\b'.format(pattern)
def replace_identity(texts,generator=None):
    #Replaces each term with a sampled counterfactual
    out = []
    #for text in tqdm(texts,desc="replacing identities",dynamic_ncols=True):
    for text in texts:
        terms = list(set([term for term in IDENTITY_TERMS
                         if re.search(bound(term), text, flags=re.IGNORECASE)]))
        for term in terms:
            index = torch.randint(0,len(IDENTITY_TERMS),[],generator=generator)
            #TODO: This is not truly independent as already sampled terms could be replaced again, but this is unlikely to matter...
            text = re.sub(bound(term), IDENTITY_TERMS[index], text, flags=re.IGNORECASE)
        out.append(text)
    return out

def censor_identity(texts,target=" "):
    out = []
    for text in texts:
        terms = list(set([term for term in IDENTITY_TERMS
                          if re.search(bound(term), text, flags=re.IGNORECASE)]))
        for term in terms:
            text = re.sub(bound(term),target, text, flags=re.IGNORECASE)
        out.append(text)
    return out

def censor_identity_extended(texts,target=" "):
    out = []
    for text in texts:
        terms = list(set([term for term in IDENTITY_TERMS_EXTEND
                          if re.search(bound(term), text, flags=re.IGNORECASE)]))
        for term in terms:
            text = re.sub(bound(term),target, text, flags=re.IGNORECASE)
        out.append(text)
    return out

def replace_identity_limited(texts,identity_terms,generator=None):
    # Replaces each term with a sampled counterfactual using a subset of all base/replacement terms
    out = []
    for text in texts:
        terms = list(set([term for term in identity_terms
                         if re.search(bound(term), text, flags=re.IGNORECASE)]))
        for term in terms:
            index = torch.randint(0,len(identity_terms),[],generator=generator)
            #TODO: This is not truly independent as already sampled terms could be replaced again, but this is unlikely to matter...
            text = re.sub(bound(term), identity_terms[index], text, flags=re.IGNORECASE)
        out.append(text)
    return out


def replace_identity_all(texts,identity_terms=IDENTITY_TERMS,subsample=None):
    # Replaces each term in identity terms with either a complete set of alternative terms (subsample=None),
    # or a sampled subset of these (subsample=n)
    # Note, that this replaces all occurences by the same term...
    # I assume that's what was done in https://arxiv.org/pdf/2006.14168.pdf
    if subsample is None:
        out = [[] for i in range(len(identity_terms))]
    else:
        out = [[] for i in range(subsample)]
    for text in texts:
        terms = list(set([term for term in identity_terms
                  if re.search(bound(term), text, flags=re.IGNORECASE)]))

        if subsample is None:
            indices = list(range(len(identity_terms)))
        else:
            indices = np.random.choice(list(range(len(identity_terms))),subsample,replace=False)
        for i in range(len(indices)):
            text_mod = text
            for term in terms:
                text_mod = re.sub(bound(term), identity_terms[indices[i]], text_mod, flags=re.IGNORECASE)
            out[i].append(text_mod)
    return out