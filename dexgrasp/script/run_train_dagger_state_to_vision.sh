CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python train.py \
--task=ShadowHandRandomLoadVision \
--algo=dagger \
--seed=0 \
--rl_device=cuda:0 \
--sim_device=cuda:0 \
--logdir=logs/test \
--expert_model_dir=example_model/state_based_model.pt \
--headless \
--vision \
--backbone_type pn # pn/transpn 
# --test
