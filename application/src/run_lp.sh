for dataset in 'Cora_ml' 'Citeseer' 'PB' 'bio_drug-drug' 'bio_function-function' 'bio_protein-protein'
do
for c in 1 1e-3 1e-5
do
#baselines
python runners/run.py --dataset_name $dataset --wandb --wandb_run_name lg_de --reps 3 --num_hops 1 --node_label degree --penalty l2 --penalty_c $c
python runners/run.py --dataset_name $dataset --wandb --wandb_run_name lg_de+ --reps 3 --num_hops 1 --node_label degree+ --penalty l2 --penalty_c $c
python runners/run.py --dataset_name $dataset --wandb --wandb_run_name lg_cn --reps 3 --num_hops 1 --node_label cn --penalty l2 --penalty_c $c
python runners/run.py --dataset_name $dataset --wandb --wandb_run_name lg_cn+ --reps 3 --num_hops 1 --node_label cn+ -penalty l2 --penalty_c $c
#global methods
python runners/run.py --dataset_name $dataset --wandb --wandb_run_name lg_ppr --reps 3 --node_label ppr --penalty l2 --penalty_c $c --use_no_subgraph
python runners/run.py --dataset_name $dataset --wandb --wandb_run_name lg_ppr+ --reps 3 --node_label ppr+ --penalty l2 --penalty_c $c --use_no_subgraph


for m in 2 3 4
do
hop=$(( m / 2 ))
#walk baselines
python runners/run.py --dataset_name $dataset --wandb --wandb_run_name lg_rw --reps 3 --num_hops $hop --max_dist $m --node_label rw --penalty l2 --penalty_c $c
python runners/run.py --dataset_name $dataset --wandb --wandb_run_name lg_rw+ --reps 3 --num_hops $hop --max_dist $m --node_label rw+ --penalty l2 --penalty_c $c

# walk profiles
python runners/run.py --dataset_name $dataset --wandb --wandb_run_name lg_wp --reps 3 --num_hops $hop --node_label wp --max_dist $m --penalty l2 --penalty_c $c

for q in 1 2 3
do
python runners/run.py --dataset_name $dataset --wandb --wandb_run_name lg_mw --reps 3 --num_hops $hop --node_label mw --max_dist $m --q_dim $q --compact_q --penalty l2 --penalty_c $c

done
done
done
done
