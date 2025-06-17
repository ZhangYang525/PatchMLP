export CUDA_VISIBLE_DEVICES=1

model=TimesNet
seq=96
python -u run.py \
  --task_name long_term_forecast \
   --is_training 1 \
   --root_path ./dataset/solar/ \
   --data_path solar_AL.txt \
   --model_id Solar \
   --model $model \
   --data Solar \
   --features M \
  --seq_len $seq \
  --label_len 0 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --d_model 256 \
  --d_ff 512 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
   --is_training 1 \
   --root_path ./dataset/solar/ \
   --data_path solar_AL.txt \
   --model_id Solar \
   --model $model \
   --data Solar \
   --features M \
  --seq_len $seq \
  --label_len 0 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --d_model 256 \
  --d_ff 512 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
   --is_training 1 \
   --root_path ./dataset/solar/ \
   --data_path solar_AL.txt \
   --model_id Solar \
   --model $model \
   --data Solar \
   --features M \
  --seq_len $seq \
  --label_len 0 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --d_model 256 \
  --d_ff 512 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
   --is_training 1 \
   --root_path ./dataset/solar/ \
   --data_path solar_AL.txt \
   --model_id Solar \
   --model $model \
   --data Solar \
   --features M \
  --seq_len $seq \
  --label_len 0 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --d_model 256 \
  --d_ff 512 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1