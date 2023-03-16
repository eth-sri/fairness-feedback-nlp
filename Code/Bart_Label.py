from Transfer.Training.Training import train_generator
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source_folder")
    args = parser.parse_args()
    train_generator(batch_size=3,max_length=64, lr=1e-5, epochs=3 ,mode="Bart",attention_layer=10,
               save=args.source_folder+"/Bart_test_label_attention_mean_64",drop_q="mean",drop_mode="attention",drop_classifier=args.source_folder+"/Roberta_Kaggle_64")
