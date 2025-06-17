export CUDA_VISIBLE_DEVICES=2

model=FEDformer

for preLen in 96 192 336 720
do

python -u run.py \
--task_name long_term_forecast \
 --is_training 1 \
 --root_path ./dataset/solar/ \
 --data_path solar_AL.txt \
 --model_id Solar \
 --model $model \
 --data Solar \
 --features M \
 --seq_len 96 \
 --label_len 48 \
 --pred_len $preLen \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
 --des 'Exp' \
 --itr 1
done