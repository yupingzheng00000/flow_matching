nohup setsid torchrun --standalone --nnodes=1 --nproc_per_node=8 ./examples/image/train.py --dataset=cifar10 --discrete_flow_matching --use_ema \
--metric_induced --batch_size=256 --eval_batch_size=625 --lr=2e-4 --accum_iter=1 \
--epochs=5900 --fid_samples=5000 --eval_frequency=50 --eval_start_epoch 5600 \
--resume=./models/baseline.pth --output_dir=./output_dir/learned_emb_gamma1.0_lam_1.5 \
--mi_use_gumbel --mi_gumbel_tau=1.0 --mi_gumbel_tau_start=2.0 \
--mi_gumbel_tau_end=0.5 --mi_gumbel_tau_anneal_steps=10000 --mi_gumbel_tau_schedule=cosine \
--mi_learnable_lut --mi_lut_num_channels 3 --mi_lut_param_mode none --mi_lut_init_method linear --mi_lut_init_noise_scale 0.1 \
--mi_metric lp --mi_lp 2.0 --mi_lut_emb_dim 16 --mi_freeze_beta_schedule --compute_fid --save_eval_gif --sym_func --cfg_scale=0.0 \
--mi_lut_weight_decay=0 --mi_lut_renorm_init_norm --mi_use_normalized_distance \
--wandb --wandb_project 111 --wandb_run_name learned_emb_noise_renorm_16d\
--no_cosine_attention --no_head_weight_norm --force_display_start_epoch 4500 \
--t_bias_gamma 1.0 --t_weight_mode linear_t --t_weight_lambda 1.5 --t_weight_normalize \
--lut_recon_weight=0.5 --lut_recon_sample_frac=0.25 --bf16

git config --global user.name "zhengyuping00000"
git config --global user.email "dd719876254@outlook.com"

nohup setsid torchrun --standalone --nnodes=1 --nproc_per_node=8 ./examples/image/train.py --dataset=cifar10 --discrete_flow_matching --use_ema \
--metric_induced --batch_size=625 --lr=2e-4 --accum_iter=1 \
--epochs=5900 --fid_samples=5000 --eval_frequency=50 \
--resume=./models/baseline.pth --output_dir=./output_dir/learned_emb_gamma1.0_lam_1.5 \
--mi_use_gumbel --mi_gumbel_tau=1.0 --mi_gumbel_tau_start=2.0 \
--mi_gumbel_tau_end=0.5 --mi_gumbel_tau_anneal_steps=10000 --mi_gumbel_tau_schedule=cosine \
--mi_learnable_lut --mi_lut_num_channels 3 --mi_lut_param_mode none --mi_lut_init_method linear --mi_lut_init_noise_scale 0.1 \
--mi_metric lp --mi_lp 2.0 --mi_lut_emb_dim 16 --mi_freeze_beta_schedule --compute_fid --save_eval_gif --sym_func --cfg_scale=0.0 \
--mi_lut_weight_decay=0 --mi_lut_renorm_init_norm --mi_use_normalized_distance \
--wandb --wandb_project 111 --wandb_run_name learned_emb_noise_renorm_16d\
--no_cosine_attention --no_head_weight_norm --force_display_start_epoch 4500 \
--bf16 --eval_only