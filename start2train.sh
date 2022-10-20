# --using_attention --using_diff --discriminator --distill --thermal  --compute --transloss 
# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=3 python train.py --height 192 --width 640 --scheduler_step_size 14 \
#                 --batch_size 4 --frame_ids 0 --use_stereo --model_name stereo_woAttention_wodiff --png --data_path ./kitti_data --log_dir ./tmp --debug 
# ## RGB
# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 python train.py --height 448 --width 512 --scheduler_step_size 14 --dataset kaist --split kaist --num_epochs 50 --max_depth 50 \
#                 --batch_size 4 --frame_ids 0 --use_stereo --model_name Kaist_transdssl_RGB_lrhalf --scales 0 \
#                 --data_path MTN_data --log_dir ./tmp --using_attention --using_diff --model transdssl \
# #                 --load_weights_folder diffnet_640x192_ms #--debug #--num_workers 0


# Thermal
# OMP_NUM_THREADS=5 CUDA_VISIBLE_DEVICES=2 python train.py --height 448 --width 512 --scheduler_step_size 14 --dataset kaist --split kaist --num_epochs 50 --max_depth 50 \
#                 --batch_size 4 --frame_ids 0 --use_stereo --model_name Kaist_resnet_thermal_lrhalf \
#                 --data_path MTN_data --log_dir ./tmp --using_attention --using_diff --model resnet --thermal \
#                 --load_weights_folder diffnet_640x192_ms #--debug #--num_workers 0

# Distill
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 python train.py --height 448 --width 512 --scheduler_step_size 14 --dataset kaist --split kaist --num_epochs 50 --max_depth 50 \
                --batch_size 1 --frame_ids 0 --use_stereo --model_name Kaist_DIFF_distill_SSIM_VGG_patchvlad_VGG_patchvlad  --scales 0 1 2 3 \
                --data_path MTN_data --log_dir ./tmp --using_attention --using_diff --model DIFF --scale_depth \
                --thermal --distill --compute --SSIM_d --vggloss_d --patchvlad_d --vggloss --patchvlad \
                 --load_weights_folder diffnet_640x192_ms \

# # # Distill
# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=3 python train.py --height 448 --width 512 --scheduler_step_size 14 --dataset kaist --split kaist --num_epochs 50 --max_depth 50 \
#                 --batch_size 4 --frame_ids 0 --use_stereo --model_name Kaist_diff_distill_patch --scale_depth\
#                 --data_path MTN_data --log_dir ./tmp --using_attention --using_diff --model DIFF --thermal \
#                 --load_weights_folder diffnet_640x192_ms --distill --compute --patchvlad \

# OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 train.py --height 448 --width 512 --scheduler_step_size 14 --dataset kaist --split kaist --num_epochs 50 --max_depth 50 \
#                 --batch_size 1 --frame_ids 0 --use_stereo --model_name Kaist_diff_distill_Only_vggloss_d_patchvlad_d_v --scale_depth\
#                 --data_path MTN_data --log_dir ./tmp --using_attention --using_diff --model DIFF --thermal --debug \
#                 --load_weights_folder diffnet_640x192_ms --distill --compute --SSIM_d --vggloss_d --patchvlad_d --vggloss --patchvlad\

