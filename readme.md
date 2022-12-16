# CWBA
## environment setup
```
pip install -r requirements.txt
```
## generate visual tensor
first change to generate_visual_tensor directory
```
cd generate_visual_tensor
```
```
python generate_tensor.py
```
then copy visual tensor to target root path(token_level_attack or sentence_level_attack)
```
cp suffix_visual_tensor_bert-base-uncased.pth ../target_root_name
```
## token level attack
first change to token_level_attack directory
```
cd token_level_attack
```
### train ner model
if the model is bert, run preprocess for bert, or just skip this step
```
python preprocess_data.py
```
train token classification model
```
python run_ner.py --model_name_or_path bert-base-uncased --train_file data/atis/train_temp_processed.json --validation_file data/atis/dev_temp_processed.json --test_file data/atis/test_temp_processed.json --output_dir logdir/atis-demo-new1 --do_train --do_eval --num_train_epochs 3 --per_gpu_train_batch_size 8 --overwrite_output_dir
```
### process data for attack
```
python process_data.py --data_folder atis --model_checkpoint_folder logdir/atis-demo-new1/pytorch_model.bin
```
### attack model
```
python attack_model.py --num_samples 10 --initial_coeff 1 --lr 0.3 --num_iters 100 --print_every 20 --gumbel_samples 10 --data_folder atis --model_checkpoint_folder logdir/atis-demo-new1/pytorch_model.bin --tau 1 --run_id 1 --target_mode char_entity --ind_ent False --suffix_gumbel True --output_name myOutput --adv_loss cw --visual_loss True --length_loss True --lam_visual 0.1 --lam_length 2 --generate_dataset True
```

### evaluate f1 score
```
python evaluate_adv_results.py --data_folder atis --model_checkpoint_folder logdir/atis-demo-new1/pytorch_model.bin
```

## sentence level attack

### train text classification model
```
python text_classification.py --data_folder dataset --dataset ag_news --model bert-base-uncased  --finetune True  --checkpoint_folder  logdir/ag_news_bert
```

### process data for attack
```
python pre_select_token.py --dataset ag_news --model_dir logdir/ag_news_bert/pytorch_model.bin  --validation_output data/ag_news/test_processed_bert.json --model_name xlnet-base-cased
```

### attack model
```
CUDA_VISIBLE_DEVICES=1 python attack_model.py --num_samples 100 --initial_coeff 1 --model bert-base-uncased --lr 0.4 --num_iters 300 --print_every 20 --gumbel_samples 5 --data_folder ag_news --model_checkpoint_folder logdir/ag_news_bert/pytorch_model.bin --tau 1 --target_mode grad --suffix_gumbel True --output_name bert_ag_news --adv_loss cw --visual_loss True --length_loss True --lam_visual 0.1 --lam_length 0.5 --generate_dataset True --start_index 4 --kappa 7 --run_id 53
```