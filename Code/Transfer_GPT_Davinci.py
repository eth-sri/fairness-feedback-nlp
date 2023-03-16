from Transfer.Datasets.Attribute_Transfer import generate
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source_folder")
    args = parser.parse_args()

    answer = None
    while answer not in ["y", "n"]:
        answer = input("WARNING: This code will run Queries to the OpenAI API, incurring monetary costs. Continue [y/n]?")
    if answer == "n":
        assert False

    for target_label, remove_label in [(0,1),(1,0),(4,5),(5,4),(8,10),(10,8),(15,16),(16,15)]:
        print("Transfering from:",target_label," to:",remove_label)
        if target_label == remove_label:
            continue
        generate(model_type="GPT3", removal_mode="drop_thresh", q="mean", remove_label=remove_label,
                 target_label=target_label, n_beams=5, max_length=64,
                 batch_size=32, total_items=250,max_generations=2250, add_successes_only=True, overwrite_labels=True,
                 n_batches=1000000, generator_path=args.source_folder+"Bart_test_label_attention_mean_640",
                 classifier_path=args.source_folder+"/Roberta_Kaggle_64", add_label=True,log_metrics=False,
                 gpt_mode="davinci_zero", word_replacement_mode="Full",
                 eval_classifier_path=args.source_folder+"/Roberta_Kaggle_64", attention_layer=10,
                 save=args.source_folder+"/gpt_davinci_pairs/" + str(remove_label) + "_" + str(target_label))
    
    
    for target_label in [0,1,2,4,5,8,9,10,11,12,13,15,16,17,18]:
        for remove_label in [0,1,2,4,5,8,9,10,11,12,13,15,16,17,18]:
            if (target_label,remove_label) in [(0,1),(1,0),(4,5),(5,4),(8,10),(10,8),(15,16),(16,15)]:
                continue
            print("Transfering from:", target_label, " to:", remove_label)
            if target_label == remove_label:
                continue
            generate(model_type="GPT3",removal_mode="drop_thresh",q="mean",remove_label=remove_label,target_label=target_label,n_beams=5,max_length=64,
                     batch_size=32,total_items=75,max_generations=75,add_successes_only=True,overwrite_labels=True,log_metrics=False,
                     n_batches=10000,generator_path=args.source_folder+"Bart_test_label_attention_mean_640",
                     classifier_path=args.source_folder+"/Roberta_Kaggle_64",add_label=True,gpt_mode="davinci_zero",word_replacement_mode="Full",
                     eval_classifier_path=args.source_folder+"/Roberta_Kaggle_64",attention_layer=10,save=args.source_folder+"/gpt_davinci_pairs/"+str(remove_label)+"_"+str(target_label))
