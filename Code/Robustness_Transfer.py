from Transfer.Datasets.Datasets import Data_Pool
from Transfer.Training.Active import human_loop,init_human_loop,load_active_state,save_active_state,eval_active,conf_interval_dict,eval_robustness,train_regularized_filtered
from Transfer.Models.Models import DualModel
from transformers import RobertaTokenizer
from torch.utils.data import DataLoader
import torch
import argparse
import pandas as pd
import numpy as np
import os
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("lam",type=float)
    parser.add_argument("source_pool")
    parser.add_argument("source_folder")
    parser.add_argument("iterations",type=int)
    parser.add_argument("epochs")
    parser.add_argument("--train_maskedwr", action='store_true')
    parser.add_argument("--train_maskedwr50", action='store_true')
    parser.add_argument("--eval_maskedwr", action='store_true')
    parser.add_argument("--eval_maskedwr50", action='store_true')
    args = parser.parse_args()

    max_length = 64
    AL_result_dicts = []
    for step in range(args.iterations):
        model = DualModel("bert-base-uncased", "concat", max_length=128).to(
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

        if args.source_pool == "pool":
            data_pool = Data_Pool(data_sources={
                "folders": [(args.source_folder+"style_transfer_pairs", 1, (42500, 10625)),
                            (args.source_folder+"gpt_davinci_edit_direct_pairs", 1, (5300, 1325)),
                            (args.source_folder+"gpt_davinci_pairs", 1, (6200, 1550)),
                            (args.source_folder+"gpt_edit_pairs", 1, (3500, 875)),
                            (args.source_folder+"word_replacement_pairs", 1, (42500, 10625))
                            ],
            }, max_length=64, dropna=True, id_label_only=False, filter_length=True, test_fraction=4,
                remove_duplicate_comments=True)

        elif args.source_pool == "pool_nogpt":
            data_pool = Data_Pool(data_sources={
                "folders": [(args.source_folder+"style_transfer_pairs", 1, (42500, 10625)),
                            (args.source_folder+"word_replacement_pairs", 1, (42500, 10625))
                            ],
            }, max_length=64, dropna=True, id_label_only=False, filter_length=True, test_fraction=4,
                remove_duplicate_comments=True)


        elif args.source_pool == "WR":
            data_pool = Data_Pool(data_sources={
                "folders": [
                            (args.source_folder+"word_replacement_pairs", 1, (100000, 25000))
                            ],
            }, max_length=64, dropna=True, id_label_only=False, filter_length=True, test_fraction=4,
                remove_duplicate_comments=True)
        elif args.source_pool == "ST":
            data_pool = Data_Pool(data_sources={
                "folders": [
                            (args.source_folder+"style_transfer_pairs", 1, (100000, 25000))
                            ],
            }, max_length=64, dropna=True, id_label_only=False, filter_length=True, test_fraction=4,
                remove_duplicate_comments=True)
        elif args.source_pool == "WR50":
            data_pool = Data_Pool(data_sources={
                "word_replacement_simple": {"label": 1, "subsample_train": 100000, "subsample_val": 25000, "triple": True}
            }, max_length=64, dropna=True, id_label_only=False, filter_length=True, test_fraction=4,
                remove_duplicate_comments=True)


        model =  lambda x, y: torch.tensor([[1] for i in range(len(x))])

        classifier = train_regularized_filtered(data_pool, model, max_length=max_length,
                                                            batch_size=32, lam=args.lam,
                                                            confidence_threshold=0.5,
                                                            reweigh=True, epochs=int(args.epochs), hard_labels=True,
                                                            return_metrics=False, return_model=True,
                                                            maskwr=args.train_maskedwr,maskwr50=args.train_maskedwr50)

        classifier.eval()
        out_dict = {}


        for path in ["wr","wr50","pool","st"]:
            if path == "pool":
                data_pool = Data_Pool(data_sources=None,
                                      load=args.source_folder+"data_pool_final_fixed_downstream",
                                      max_length=max_length, dropna=True, id_label_only=False, filter_length=True,
                                      test_fraction=4, remove_duplicate_comments=True)
            elif path == "wr":
                data_pool = Data_Pool(data_sources={
                    "folders": [
                        (args.source_folder+"word_replacement_pairs", 1, (100000, 25000))
                    ],
                }, max_length=64, dropna=True, id_label_only=False, filter_length=True, test_fraction=4,
                    remove_duplicate_comments=True)
            elif path == "wr50":
                data_pool = Data_Pool(data_sources={
                    "word_replacement_simple": {"label": 1, "subsample_train": 100000, "subsample_val": 25000,"triple":True}
                }, max_length=64, dropna=True, id_label_only=False, filter_length=True, test_fraction=4,
                    remove_duplicate_comments=True)
            elif path == "st":
                data_pool = Data_Pool(data_sources={
                    "folders": [
                        (args.source_folder+"style_transfer_pairs", 1, (100000, 25000))
                    ],
                }, max_length=64, dropna=True, id_label_only=False, filter_length=True, test_fraction=4,
                    remove_duplicate_comments=True)
            else:
                assert False
            test_loader = DataLoader(data_pool.get_val_data_base_task(), batch_size=32, shuffle=True)
            tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
            perturbation_dict_eval = data_pool.get_eval_dict()
            print(len(perturbation_dict_eval))
            out_dict = eval_robustness(test_loader, perturbation_dict_eval, tokenizer, 64, 100000, classifier,path,
                                       out_dict,maskwr=args.eval_maskedwr,maskwr50=args.eval_maskedwr50)

        AL_result_dicts.append(out_dict)
    print(conf_interval_dict(AL_result_dicts))

    name = ""
    if args.train_maskedwr:
        name += "mask_wr"
    if args.train_maskedwr50:
        name += "mask_wr50"

    with open("/Results/AL/table1"+args.source_pool+str(args.lam)+name+".json","w") as fp:
        json.dump(conf_interval_dict(AL_result_dicts),fp,indent=2)



