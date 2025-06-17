export CUDA_VISIBLE_DEVICES=0

model_name=SFNN

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_96 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --enc_in 7 \
  --des 'Exp' \
  --d_model 768 \
  --d_ff 768 \
  --num_blocks 2 \
  --batch_size 32 \
  --learning_rate 0.0005 \
  --itr 1
#  # 0.2825 AdamW
#
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_192 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --enc_in 7 \
  --des 'Exp' \
  --d_model 768 \
  --d_ff 768 \
  --num_blocks 2 \
  --batch_size 32 \
  --learning_rate 0.0005 \
  --itr 1
#  # 0.3530 Adam
#
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_336 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --enc_in 7 \
  --des 'Exp' \
  --d_model 768 \
  --d_ff 768 \
  --num_blocks 2 \
  --batch_size 32 \
  --learning_rate 0.0005 \
  --itr 1
  # 0.3976 AdamW

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_720 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --enc_in 7 \
  --des 'Exp' \
  --d_model 768 \
  --d_ff 768 \
  --num_blocks 1 \
  --batch_size 64 \
  --learning_rate 0.0005 \
  --itr 1

  # 0.4187  Adam
  # 0.4091 Adam frequency_loss
