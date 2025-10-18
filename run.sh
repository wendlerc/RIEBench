for n_steps in 4; do
    for k_transfer in 80 160 320 640; do 
        for m1 in 1. 2.; do
            papermill main.ipynb ./results/camera_ready/sae_k${k}_exp${exp}_t${n_steps}_ktrans${k_transfer}_m1${m1}.ipynb \
                -p n_steps ${n_steps} \
                -p k_transfer ${k_transfer} \
                -p m1 ${m1} \
                -p mode "sae" \
                -p keep_spatial_info True \
                -p dtype "float32" \
                -p k 160 \
                -p exp 4 \
                -p n_examples_per_edit 50 \
                -p prefix "./results/camera_ready" \
                -p aggregate_timesteps "first"
        done
    done
done

for n_steps in 4; do
        for m1 in 0.25 0.5 1. 2.; do
            papermill main.ipynb ./results/camera_ready/steering_k${k}_exp${exp}_t${n_steps}_ktrans${k_transfer}_m1${m1}.ipynb \
                -p n_steps ${n_steps} \
                -p k_transfer ${k_transfer} \
                -p m1 ${m1} \
                -p mode "steering" \
                -p keep_spatial_info True \
                -p dtype "float32" \
                -p k 160 \
                -p exp 4 \
                -p n_examples_per_edit 50 \
                -p prefix "./results/camera_ready" \ 
    done
done



for n_steps in 4; do # 4 * 51200 neurons total
    for k_transfer in 5000 10000 20000 40000 80000; do 
        for m1 in 1. 2.; do
            papermill main.ipynb ./results/camera_ready/neurons_mean_k${k}_exp${exp}_t${n_steps}_ktrans${k_transfer}_m1${m1}.ipynb \
                -p n_steps ${n_steps} \
                -p k_transfer ${k_transfer} \
                -p m1 ${m1} \
                -p mode "neurons" \
                -p keep_spatial_info True \
                -p dtype "float32" \
                -p k 160 \
                -p exp 4 \
                -p n_examples_per_edit 50 \
                -p prefix "./results/camera_ready" \
                -p aggregate_timesteps "mean"
        done
    done
done


for n_steps in 4; do # 4 * 51200 neurons total
    for k_transfer in 5000 10000 20000 40000 80000; do 
        for m1 in 1. 2.; do
            papermill main.ipynb ../results/camera_ready/neurons_first_k${k}_exp${exp}_t${n_steps}_ktrans${k_transfer}_m1${m1}.ipynb \
                -p n_steps ${n_steps} \
                -p k_transfer ${k_transfer} \
                -p m1 ${m1} \
                -p mode "neurons" \
                -p keep_spatial_info True \
                -p dtype "float32" \
                -p k 160 \
                -p exp 4 \
                -p n_examples_per_edit 50 \
                -p prefix "../results/camera_ready" \
                -p aggregate_timesteps "first"
        done
    done
done