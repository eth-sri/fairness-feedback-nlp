from Transfer.Datasets.Attribute_Transfer import generate
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source_folder")
    parser.add_argument("max_generations", default=10000000,type=int)
    args = parser.parse_args()

    for target_label in [0,1,2,4,5,8,9,10,11,12,13,15,16,17,18]:
        for remove_label in [0,1,2,4,5,8,9,10,11,12,13,15,16,17,18]:
            print("Transfering from:",target_label," to:",remove_label)
            if target_label == remove_label:
                continue
            generate(model_type=None,removal_mode="drop_thresh",q="mean",remove_label=remove_label,target_label=target_label,n_beams=5,max_length=64,
                     batch_size=64,total_items=1000000,max_generations=args.max_generations,add_successes_only=True,overwrite_labels=True,log_metrics=False,
                     n_batches=1000000,generator_path=args.source_folder+"Bart_test_label_attention_mean_640",
                     classifier_path=args.source_folder+"/Roberta_Kaggle_64",add_label=True,gpt_mode=None,word_replacement_mode="Full",
                     eval_classifier_path=args.source_folder+"/Roberta_Kaggle_64",attention_layer=10,save=args.source_folder+"/word_replacement_pairs/"+str(remove_label)+"_"+str(target_label))