# Image example

## Training instructions

1. Download and unpack blurred ImageNet from the [official website](https://image-net.org/download.php).

```
export IMAGENET_DIR=~/flow_matching/examples/image/data/
export IMAGENET_RES=64
tar -xf ~/Downloads/train_blurred.tar.gz -C $IMAGENET_DIR
```

2. Downsample Imagenet to the desired resolution.

```
cd ~/
git clone git@github.com:PatrykChrabaszcz/Imagenet32_Scripts.git
python Imagenet32_Scripts/image_resizer_imagent.py -i ${IMAGENET_DIR}train_blurred -o ${IMAGENET_DIR}train_blurred_$IMAGENET_RES -s $IMAGENET_RES -a box  -r -j 10 
```

3. Set up the virtual environment. First, set up the virtual environment by following the steps in the repository's `README.md`. Then,

```
conda activate flow_matching

cd examples/image
pip install -r requirements.txt
```

4. [Optional] Test-run training locally. A test run executes one step of training followed by one step of evaluation.

```
python train.py --data_path=${IMAGENET_DIR}train_blurred_$IMAGENET_RES/box/ --test_run
```

5. Launch training on a SLURM cluster

```
python submitit_train.py --data_path=${IMAGENET_DIR}train_blurred_$IMAGENET_RES/box/ 
```

6. Evaluate the model using the `--eval_only` flag. The evaluation script will generate snapshots under the `/snapshots` folder. Specify the `--compute_fid` flag to also compute the FID with respect to the training set. Make sure to specify your most recent checkpoint to resume from. The results are printed to `log.txt`.

```
python submitit_train.py --data_path=${IMAGENET_DIR}train_blurred_$IMAGENET_RES/box/ --resume=./output_dir/checkpoint-899.pth --compute_fid --eval_only
```

### Metric-induced geometry diagnostics

When you enable the metric-induced Gibbs path (`--discrete_flow_matching --ko_metric_induced`), you can ask the evaluation loop to measure how far the learned Mahalanobis metric drifts from the baseline distances. Typical settings that balance coverage and runtime are:

```bash
python train.py \
  --eval_only --compute_fid --discrete_flow_matching --ko_metric_induced \
  --resume=./output_dir/checkpoint-899.pth \
  --mi_learnable_metric --mi_metric_use_ema \
  --mi_metric_eval_geometry --mi_metric_eval_subset=128 \
  --mi_metric_eval_pair_samples=20000 --mi_metric_eval_knn_k=5 \
  --mi_metric_eval_seed=0 --mi_metric_eval_heatmap \
  --mi_metric_eval_heatmap_subset=64
```

- `--mi_metric_eval_subset=128` probes a 128-token slice of the vocabulary (set `0` to use everything; useful for CIFAR/ImageNet scales).
- `--mi_metric_eval_pair_samples=20000` randomly samples 20k off-diagonal pairs when computing Spearman ρ, which keeps memory low but gives a stable estimate.
- `--mi_metric_eval_knn_k=5` matches the defaults used in the literature for k-NN overlap probes.
- Heatmaps require `matplotlib`; lowering `--mi_metric_eval_heatmap_subset` keeps the figure legible.

The diagnostics run once per evaluation and log the Spearman/k-NN scores alongside FID. Heatmaps and CSVs land under `${output_dir}/metric_geometry/`.

### Evaluation GIF logging

To capture the full sampling trajectories as animated GIFs during evaluation, toggle `--save_eval_gif`. A typical configuration that trades detail for file size is:

```bash
python train.py \
  --eval_only --compute_fid --resume=./output_dir/checkpoint-899.pth \
  --save_eval_gif --eval_gif_max_batch=16 --eval_gif_stride=8 --eval_gif_fps=12
```

- `--eval_gif_max_batch=16` tiles up to 16 samples (4×4 grid) per frame; increase for wider grids.
- `--eval_gif_stride=8` stores every 8th solver step to keep animations compact.
- `--eval_gif_fps=12` plays the GIF at 12 frames per second for smooth but concise playback.

GIFs are written to `${output_dir}/gifs/` and, if Weights & Biases is enabled, are uploaded as videos to the run dashboard.


## Results
| Data                  | Model type                       | Epochs | FID  | Command                                                                                                                                                                                                                                                                                                                                                   |
|-----------------------|----------------------------------|-------|------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Cifar10               | Unconditional UNet               | 1800  | 2.07 | `python submitit_train.py \`<br>`--dataset=cifar10 \`<br>`--batch_size=64 \`<br>`--nodes=1 \`<br>`--accum_iter=1 \`<br>`--eval_frequency=100 \`<br>`--epochs=3000 \`<br>`--class_drop_prob=1.0 \`<br>`--cfg_scale=0.0 \`<br>`--compute_fid \`<br>`--ode_method heun2 \`<br>`--ode_options '{"nfe": 50}' \`<br>`--use_ema \`<br>`--edm_schedule \`<br>`--skewed_timesteps` |
| ImageNet32 (Blurred)  | Class conditional Unet           | 900   | 1.14 | `export IMAGENET_RES=32 \`<br>`python submitit_train.py \`<br>`--data_path=${IMAGENET_DIR}train_blurred_$IMAGENET_RES/box/ \`<br>`--batch_size=32 \`<br>`--nodes=8 \`<br>`--accum_iter=1 \`<br>`--eval_frequency=100 \`<br>`--decay_lr \`<br>`--compute_fid \`<br>`--ode_method dopri5 \`<br>`--ode_options '{"atol": 1e-5, "rtol":1e-5}'` |
| ImageNet64 (Blurred)  | Class conditional Unet           | 900   | 1.64 | `export IMAGENET_RES=64 \`<br>`python submitit_train.py \`<br>`--data_path=${IMAGENET_DIR}train_blurred_$IMAGENET_RES/box/ \`<br>`--batch_size=32 \`<br>`--nodes=8 \`<br>`--accum_iter=1 \`<br>`--eval_frequency=100 \`<br>`--decay_lr \`<br>`--compute_fid \`<br>`--ode_method dopri5 \`<br>`--ode_options '{"atol": 1e-5, "rtol":1e-5}'` |
| Cifar10 (Discrete Flow) | Unconditional Unet           | 2500   | 3.58 | `python submitit_train.py \`<br>`--dataset=cifar10 \`<br>`--nodes=1 \`<br>`--discrete_flow_matching \`<br>`--batch_size=32 \`<br>`--accum_iter=1 \`<br>`--cfg_scale=0.0 \`<br>`--use_ema \`<br>`--epochs=3000 \`<br>`--class_drop_prob=1.0 \`<br>`--compute_fid \`<br>`--sym_func` |



## Acknowledgements

This example partially use code from:
- [Guided diffusion](https://github.com/openai/guided-diffusion/)
- [ConvNext](https://github.com/facebookresearch/ConvNeXt)

## License

The majority of the code in this example is licensed under CC-BY-NC, however portions of the project are available under separate license terms: 
- The UNet model is under MIT license.
- The distributed computing and the grad scaler code is under MIT license.

## Citations

Deng, Jia, et al. "Imagenet: A large-scale hierarchical image database." 2009 IEEE conference on computer vision and pattern recognition. Ieee, 2009.

Karras, Tero, et al. "Elucidating the design space of diffusion-based generative models." Advances in neural information processing systems 35 (2022): 26565-26577.

Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18. Springer International Publishing, 2015.
