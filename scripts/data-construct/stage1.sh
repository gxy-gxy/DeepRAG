# small model for generate konwledge-based intermediate answer
REMOTE_URL=""
for ip in "0.0.0.0"; do
    for port in "8000" "8001" "8002" "8003" "8004" "8005" "8006" "8007";do
        REMOTE_URL="${REMOTE_URL}http://${ip}:${port}/v1 "
    done
done

# qwen-72b for generate follow-up
FOLLOW_UP_REMOTE_URL=""
for ip in "0.0.0.0"; do
    for port in "8000";do
        FOLLOW_UP_REMOTE_URL="${FOLLOW_UP_REMOTE_URL}http://${ip}:${port}/v1 "
    done
done

export ES_HOST=0.0.0.0

CEPH_HOME=~

cd $CEPH_HOME/DeepRAG

python3 src/main.py \
    --remote_url ${REMOTE_URL} \
    --method srag-sample \
    --model_name_or_path $CEPH_HOME/hf_models/Qwen2.5-7B-Instruct \
    --dataset hotpotqa \
    --data_path data/hotpot \
    --split sft_new \
    --num_processes 64 \
    --generate_max_length 4096 \
    --sample 10000 \
    --output_dir construct/sft/hotpotqa \
    --temperature 0.0 \
    --follow_up_remote_url ${FOLLOW_UP_REMOTE_URL}
    


python3 src/main.py \
    --remote_url ${REMOTE_URL} \
    --method srag-sample \
    --model_name_or_path $CEPH_HOME/hf_models/Qwen2.5-7B-Instruct \
    --dataset 2wikimultihopqa \
    --data_path data/wikihop \
    --split sft_new \
    --num_processes 64 \
    --generate_max_length 4096 \
    --sample 100000 \
    --output_dir construct/sft/wikihop \
    --temperature 0.0 \
    --follow_up_remote_url ${FOLLOW_UP_REMOTE_URL}
    
