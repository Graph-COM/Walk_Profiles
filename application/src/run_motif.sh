for dataset in 'npz-transformer' 'Cora_ml' 'FIG'
do
for y in '101' '1011' '10111' '1011111' '1011111111'
do
d=${#y}

for q in $(seq 1 $((d+1)))
do
python runners/run_node.py --dataset_name ${dataset} --target "cycle"${y} --node_label wp --max_dist ${d} --reps 10 --eval_metric auc --binarize_target --x_norm --wandb --wandb_run_name ${dataset} --q_dim $q --wp2mw --mw2wp 
python runners/run_node.py --dataset_name ${dataset} --target "cycle"${y} --node_label wp --max_dist ${d} --reps 10 --eval_metric auc --binarize_target --x_norm --wandb --wandb_run_name ${dataset} --q_dim $q --wp2mw --mw2wp --norm 
done

python runners/run_node.py --dataset_name ${dataset} --target "cycle"${y} --node_label wp --max_dist ${d} --reps 10 --eval_metric auc --binarize_target --x_norm --wandb --wandb_run_name ${dataset}
python runners/run_node.py --dataset_name ${dataset} --target "cycle"${y} --node_label rw --max_dist ${d} --reps 10 --eval_metric auc --binarize_target --x_norm --wandb --wandb_run_name ${dataset} 
python runners/run_node.py --dataset_name ${dataset} --target "cycle"${y} --node_label rw+ --max_dist ${d} --reps 10 --eval_metric auc --binarize_target --x_norm --wandb --wandb_run_name ${dataset}
python runners/run_node.py --dataset_name ${dataset} --target "cycle"${y} --node_label wp --max_dist ${d} --reps 10 --eval_metric auc --binarize_target --x_norm --wandb --wandb_run_name ${dataset} --norm 
python runners/run_node.py --dataset_name ${dataset} --target "cycle"${y} --node_label rw --max_dist ${d} --reps 10 --eval_metric auc --binarize_target --x_norm --wandb --wandb_run_name ${dataset} --norm 
python runners/run_node.py --dataset_name ${dataset} --target "cycle"${y} --node_label rw+ --max_dist ${d} --reps 10 --eval_metric auc --binarize_target --x_norm --wandb --wandb_run_name ${dataset} --norm 
done
done
