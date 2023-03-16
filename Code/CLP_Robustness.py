from Transfer.Training.Active import experiment
from Transfer.Datasets.Datasets import Data_Pool
import os
import numpy as np
import argparse
max_length = 64

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source_folder")
    args = parser.parse_args()

    data_pool = Data_Pool(data_sources={
        "folders":[(args.source_folder+"/style_transfer_pairs",0,(42500,10625)),
                    (args.source_folder+"/gpt_davinci_edit_direct_pairs",0,(5300,1325)),
                    (args.source_folder+"/gpt_davinci_pairs",0,(6200,1550)),
                    (args.source_folder+"/gpt_edit_pairs",0,(3500,875)),
                    (args.source_folder+"/word_replacement_pairs",1,(42500,10625))
                   ],
    }, max_length=max_length, dropna=True, id_label_only=False, filter_length=True,test_fraction=4,remove_duplicate_comments=True)

    kwargs = {"name":"Results/AL/Baseline","constant_0":False,"constant_1":False,"tprtnr":(0,1),"lam":0,"iterations":5,
              "classifier_weights":False,"skip_downstream":False,"confidence_threshold":0.5,"downstream_epochs":3,"downstream_evalmode":True}
    experiment(data_pool,max_length,kwargs)

    kwargs = {"name":"Results/AL/Baseline","constant_0":False,"constant_1":False,"tprtnr":(1,0),"lam":5,"iterations":5,
              "classifier_weights":False,"skip_downstream":False,"confidence_threshold":0.5,"downstream_epochs":3,"downstream_evalmode":True}
    experiment(data_pool,max_length,kwargs)

    kwargs = {"name":"Results/AL/Baseline","constant_0":False,"constant_1":False,"tprtnr":(1,1),"lam":5,"iterations":5,
              "classifier_weights":False,"skip_downstream":False,"confidence_threshold":0.5,"downstream_epochs":3,"downstream_evalmode":True}
    experiment(data_pool,max_length,kwargs)

