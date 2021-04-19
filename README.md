# LingoRobo

Authors: 
* [Kung-Hsiang Steeve Huang](http://khuangaf.github.io/)*
* [Shikhar Singh](https://www.linkedin.com/in/shikhar-singh-730910a7).*

*equal contribution

Source code for **commonsesne knowledge incorporation** please navigate to branch [know_inject](https://github.com/USC-CSCI527-Spring2021/LingoRobo/tree/know_inject).
## Environment
### Clone repository
```
$ git clone https://github.com/gistvision/moca.git moca
$ export ALFRED_ROOT=$(pwd)/moca
```

### Install requirements
```
$ virtualenv -p $(which python3) --system-site-packages moca_env
$ source moca_env/bin/activate

$ cd $ALFRED_ROOT
$ pip install --upgrade pip
$ pip install -r requirements.txt
```


## Download
### Dataset
Dataset includes visual features extracted by ResNet-18 with natural language annotations (~135.5GB after unzipping).
For details of the ALFRED dataset, see the repository of <a href="https://github.com/askforalfred/alfred">ALFRED</a>.
```
$ cd $ALFRED_ROOT/data
$ sh download_data.sh
```
**Note**: The downloaded data includes expert trajectories with both original and color-swapped frames.

### Pretrained Model
We provide our pretrained weight used for the experiments in the paper and the leaderboard submission.
To download the pretrained weight of MOCA, use the command below.
```
$ cd $ALFRED_ROOT
$ sh download_model.sh
```

## Training
To train MOCA, run `train_seq2seq.py` with hyper-parameters below. <br>
```
python models/train/train_seq2seq.py --data <path_to_dataset> --model seq2seq_im_mask --dout <path_to_save_weight> --splits data/splits/oct21.json --gpu --batch <batch_size> --pm_aux_loss_wt <pm_aux_loss_wt_coeff> --subgoal_aux_loss_wt <subgoal_aux_loss_wt_coeff> --preprocess
```
**Note**: As mentioned in the repository of <a href="https://github.com/askforalfred/alfred/tree/master/models">ALFRED</a>, run with `--preprocess` only once for preprocessed json files. <br>
**Note**: All hyperparameters used for the experiments in the paper are set as default.

For example, if you want train MOCA and save the weights for all epochs in "exp/moca" with all hyperparameters used in the experiments in the paper, you may use the command below. <br>
```
python models/train/train_seq2seq.py --dout exp/moca --gpu --save_every_epoch
```
**Note**: The option, `--save_every_epoch`, saves weights for all epochs and therefore could take a lot of space.

### Fine-tuning LM
`--use-bert` must be passed to the argument. The name of the LM (e.g. `bert-base-uncased`, `roberta-base`) can be passed using `--bert_model`. `--max_length` refers to the max sub-tokens for each instruction and goal. Sequences longer than `--max_length` will be truncated. `--bert_lr` refers to the learning rate for fine-tuning LM. Usually, it is beneficial to set LM learning rate to be smaller than the learning rate for training the other component of the model. An example of fine-tuning LM.

```
python models/train/train_seq2seq.py --data data/json_feat_2.1.0 --model seq2seq_im_mask --dout exp/model:{model},name:pm_and_subgoals_02_bert_base1e-05  --splits data/splits/oct21.json --gpu 3 --batch 2 --pm_aux_loss_wt 0.2  --subgoal_aux_loss_wt 0.2  --pp_folder pp_bert --use_bert --bert_model bert-base-uncased --bert_lr  1e-05 --max_length 512
```



## Evaluation
### Task Evaluation
To evaluate MOCA, run `eval_seq2seq.py` with hyper-parameters below. <br>
To evaluate a model in the `seen` or `unseen` environment, pass `valid_seen` or `valid_unseen` to `--eval_split`.
```
python models/eval/eval_seq2seq.py --data <path_to_dataset> --model models.model.seq2seq_im_mask --model_path <path_to_weight> --eval_split <eval_split> --gpu --num_threads <thread_num>
```
**Note**: All hyperparameters used for the experiments in the paper are set as default.

If you want to evaluate our pretrained model saved in `exp/pretrained/pretrained.pth` in the `seen` validation, you may use the command below.
```
python models/eval/eval_seq2seq.py --model_path "exp/pretrained/pretrained.pth" --eval_split valid_seen --gpu --num_threads 4
```


