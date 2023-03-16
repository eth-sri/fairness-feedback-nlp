from Transfer.Training.Training import train_roberta_multi
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source_folder")
    args = parser.parse_args()
    train_roberta_multi(label_subset=list(range(24)),save=args.source_folder+"/Roberta_Kaggle_64",model_handle="roberta-base",
                        epochs=3,max_length=64,batch_size=16,lr=1e-5,test_split=10,reverse_split=False,freeze_shared=False)

