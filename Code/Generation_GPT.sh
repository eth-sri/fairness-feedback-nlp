#!/bin/sh

python Roberta.py generations
python Bart_Label.py generations

python Transfer_GPT_Davinci.py generations
python Transfer_GPT_Davinci_Edit.py generations
python Transfer_GPT_Edit.py generations

python Transfer.py generations
python Transfer_WR_Full.py generations
python Transfer_WR_50.py generations




