#!/bin/sh

python Robustness_Transfer.py 0.0 pool_nogpt generations 5 3
python Robustness_Transfer.py 0.0 pool_nogpt generations 5 3 --eval_maskedwr50 --train_maskedwr50
python Robustness_Transfer.py 0.0 pool_nogpt generations 5 3 --eval_maskedwr --train_maskedwr
python Robustness_Transfer.py 0.0 pool_nogpt generations 5 3 --eval_maskedwr --train_maskedwr --eval_maskedwr50 --train_maskedwr50
python Robustness_Transfer.py 5.0 WR50 generations 5 3
python Robustness_Transfer.py 5.0 WR generations 5 3
python Robustness_Transfer.py 5.0 ST generations 5 3
python Robustness_Transfer.py 5.0 pool_nogpt generations 5 3


