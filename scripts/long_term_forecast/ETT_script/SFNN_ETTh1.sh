export CUDA_VISIBLE_DEVICES=0

model_name=SFNN

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --enc_in 7 \
  --des 'Exp' \
  --d_model 768 \
  --d_ff 768 \
  --num_blocks 2 \
  --batch_size 64 \
  --learning_rate 0.0005 \
  --itr 1
  # 0.3750 AdamW
  # 0.3636 AdamW MSE_loss->frequency_loss
  # 0.3631 batch_size 32->64

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_192 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --enc_in 7 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --num_blocks 3 \
  --itr 1
#
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_336 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --enc_in 7 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --num_blocks 3 \
  --itr 1
#
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_720 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --enc_in 7 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --num_blocks 3 \
  --batch_size 64 \
  --itr 1
