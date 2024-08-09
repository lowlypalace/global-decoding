#!/bin/bash -l

# Define the array of top-k and top-p values
top_k_values=(5 10 50 100 500 1000 5000 10000)
top_p_values=(0.01 0.05 0.25 0.5 0.75 0.9 0.95 0.99)

# SLURM job parameters
mail_user="email@email.com"
mail_type="ALL"
nodes=1
ntasks_per_node=1
cpus=1
gpus=1
partition="gpu"

# Model configurations
declare -A model_time=(
    [pythia-70m]="04:00:00"
    [pythia-410m]="16:00:00"
    [pythia-1.4b]="48:00:00"
    [pythia-2.8b]="160:00:00"
)

declare -A batch_size_seq=(
    [pythia-70m]=128 # 16GB V100 GPU
    [pythia-410m]=64 # 32GB V100 GPU
    [pythia-1.4b]=16 # 32GB V100 GPU
    [pythia-2.8b]=4 # 32 GB V100 GPU
)

declare -A batch_size_prob=(
    [pythia-70m]=8 # 16GB V100 GPU
    [pythia-410m]=8 # 32GB V100 GPU
    [pythia-1.4b]=8 # 32GB V100 GPU
    [pythia-2.8b]=4 # 32 GB V100 GPU
)

declare -A memory=(
    [pythia-70m]=10GB
    [pythia-410m]=10GB
    [pythia-1.4b]=16GB
    [pythia-2.8b]=32GB
)

declare -A node_type=(
    [pythia-70m]="volta"
    [pythia-410m]="volta32"
    [pythia-1.4b]="volta32"
    [pythia-2.8b]="volta32"
)

declare -A qos=(
    [pythia-70m]="normal"
    [pythia-410m]="normal"
    [pythia-1.4b]="normal"
    [pythia-2.8b]="long"
)

# Loop over models
for model in "${!model_time[@]}"; do
    # Current model parameters
    current_time="${model_time[$model]}"
    current_batch_seq="${batch_size_seq[$model]}"
    current_batch_prob="${batch_size_prob[$model]}"
    current_memory="${memory[$model]}"
    current_node_type="${node_type[$model]}"
    current_qos="${qos[$model]}"

    # Loop over top-k values and submit jobs
    for top_k in "${top_k_values[@]}"; do
        sbatch <<EOF
#!/bin/bash -l
#SBATCH -A $account
#SBATCH --qos $current_qos
#SBATCH --mail-user=$mail_user
#SBATCH --mail-type=$mail_type
#SBATCH -N $nodes
#SBATCH --ntasks-per-node=$ntasks_per_node
#SBATCH -c $cpus
#SBATCH -G $gpus
#SBATCH --time=$current_time
#SBATCH -p $partition
#SBATCH --mem=$current_memory
#SBATCH -C $current_node_type
module load lang/Python
source global-decoding/bin/activate
cd global-decoding/global-decoding
python main.py \
  --top_k $top_k \
  --sequence_count 100000 \
  --batch_size_seq $current_batch_seq \
  --batch_size_prob $current_batch_prob \
  --model_name $model \
  --max_length 512 \
  --mcmc_num_samples 200 \
  --seed 0
EOF
    done

    # Loop over top-p values and submit jobs
    for top_p in "${top_p_values[@]}"; do
        sbatch <<EOF
#!/bin/bash -l
#SBATCH -A $account
#SBATCH --qos $current_qos
#SBATCH --mail-user=$mail_user
#SBATCH --mail-type=$mail_type
#SBATCH -N $nodes
#SBATCH --ntasks-per-node=$ntasks_per_node
#SBATCH -c $cpus
#SBATCH -G $gpus
#SBATCH --time=$current_time
#SBATCH -p $partition
#SBATCH --mem=$current_memory
#SBATCH -C $current_node_type
module load lang/Python
source global-decoding/bin/activate
cd global-decoding/global-decoding
python main.py \
  --top_p $top_p \
  --sequence_count 100000 \
  --batch_size_seq $current_batch_seq \
  --batch_size_prob $current_batch_prob \
  --model_name $model \
  --max_length 512 \
  --mcmc_num_samples 200 \
  --seed 0
EOF
    done
done
