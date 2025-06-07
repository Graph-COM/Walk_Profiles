n=5000
for d in 1 2 4 8 16
do
        python generate_random_graphs.py --n $n --p_order 1-n --p_const $d --seed 0 --type er
done

for k in 1 4 16 64
do
        for scale in 1 2 3
        do
        python SCDG.py --n $n --k $k --deg_scale $scale
done
done

