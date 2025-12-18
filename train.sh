CUDA_VISIBLE_DEVICES=0 python3 -u train_single_database.py \
--num_epochs 100 \
--batch_size 30 \
--resize 384 \
--crop_size 320 \
--lr 0.00005 \
--decay_ratio 0.9 \
--decay_interval 10 \
--snapshot G:/SJTU3/数字图像处理/课程项目/StairIQA/ \
--database_dir G:/SJTU3/数字图像处理/课程项目/StairIQA/data/ \
--model stairIQA_resnet \
--multi_gpu False \
--print_samples 20 \
--database Koniq10k \
--test_method five \
>> logfiles/train_Koniq10k_stairIQA_resnet.log




# CUDA_VISIBLE_DEVICES=0 python -u train_imdt.py \
# --num_epochs 3 \
# --batch_size 30 \
# --lr 0.00001 \
# --decay_ratio 0.9 \
# --decay_interval 1 \
# --snapshot /data/sunwei_data/ModelFolder/StairIQA/ \
# --model stairIQA_resnet \
# --multi_gpu False \
# --print_samples 100 \
# --test_method five \
# --results_path results \
# --exp_id 0 \
# >> logfiles/train_stairIQA_resnet_imdt_exp_id_0.log

