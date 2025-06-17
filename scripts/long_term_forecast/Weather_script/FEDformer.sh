export CUDA_VISIBLE_DEVICES=2

model_name=FEDformer

for preLen in 96 192 336 720
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather \
  --model $model_name \
  --data custom \
  --features M \
 --seq_len 96 \
 --label_len 48 \
 --pred_len $preLen \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 21 \
 --dec_in 21 \
 --c_out 21 \
 --des 'Exp' \
 --itr 1

done