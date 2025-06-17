export CUDA_VISIBLE_DEVICES=0

model=TimeMixer
seq_len=336
down_sampling_layers=2
down_sampling_window=2
learning_rate=0.0001
batch_size=32
train_epochs=100
patience=10

python -u run.py \
  --task_name long_term_forecast \
   --is_training 1 \
   --root_path ./dataset/solar/ \
   --data_path solar_AL.txt \
   --model_id Solar \
   --model $model \
   --data Solar \
   --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 96 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --itr 1 \
  --use_norm 1 \
  --d_model 512 \
  --d_ff 2048 \
  --channel_independence 1 \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window

python -u run.py \
  --task_name long_term_forecast \
   --is_training 1 \
   --root_path ./dataset/solar/ \
   --data_path solar_AL.txt \
   --model_id Solar \
   --model $model \
   --data Solar \
   --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 192 \
  --e_layers 3 \
  --factor 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --itr 1 \
  --use_norm 1 \
  --d_model 512 \
  --d_ff 2048 \
  --channel_independence 1 \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window

python -u run.py \
  --task_name long_term_forecast \
   --is_training 1 \
   --root_path ./dataset/solar/ \
   --data_path solar_AL.txt \
   --model_id Solar \
   --model $model \
   --data Solar \
   --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 336 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --itr 1 \
  --use_norm 1 \
  --d_model 512 \
  --d_ff 2048 \
  --channel_independence 1 \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window

python -u run.py \
  --task_name long_term_forecast \
   --is_training 1 \
   --root_path ./dataset/solar/ \
   --data_path solar_AL.txt \
   --model_id Solar \
   --model $model \
   --data Solar \
   --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 720 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --itr 1 \
  --use_norm 1 \
  --d_model 512 \
  --d_ff 2048 \
  --channel_independence 1 \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window