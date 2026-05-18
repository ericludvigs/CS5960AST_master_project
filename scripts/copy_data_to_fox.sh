#!/bin/bash

# copy the relevant training data to the fox system
# --dry-run
rsync -avz --progress --filter="+ /train_smudgy/" --filter="+ /train_smudgy/*-*-0.npy" --filter="+ /train_smudgy/*-*-1.npy" --filter="+ /train_smudgy/*-*-5.npy" --filter="+ /train_smudgy/*-*-20.npy" --filter="+ /train_smudgy/*-*-27.npy" --filter="+ /train_smudgy/" --filter="+ /train_smudgy/*-10-2.npy" --filter="+ /train_smudgy/*-*-2.npy" --filter="+ /train_smudgy/*-*-1.npy" --filter="+ /train_smudgy/*-*-19.npy" --filter="+ /train_smudgy/*-*-27.npy" --filter="+ /train_smudgy/*-*-33.npy" --filter="- *" /mn/stornext/d13/euclid_nobackup/dennisfr/master-project/data/ ec-ericlu@fox.educloud.no:/fp/projects01/ec12/ec-ericlu/CS5960AST_data/

# reference for getting a model checkpoint from fox
#rsync -avz --progress ec-ericlu@fox.educloud.no:/fp/projects01/ec12/ec-ericlu/CS5960AST/models/runs/node/3468780/model_checkpoints/model_state_161.pt /mn/stornext/d5/data/ericlu/CS5960AST/models/runs/node/3468780/model_checkpoints/model_state_161.pt
