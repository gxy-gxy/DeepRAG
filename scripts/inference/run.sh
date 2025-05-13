cd DeepRAG/src


REMOTE_URL=""
for i in `seq 0 7`; do
    REMOTE_URL="${REMOTE_URL}http://0.0.0.0:800$i/v1 "
done

FOLLOW_UP_REMOTE_URL=${REMOTE_URL}

export ES_HOST=0.0.0.0


python main.py \
    --remote_url ${REMOTE_URL} \
    --method srag-sftv2 \
    --model_name_or_path hf_models/Meta-Llama-3-8B-Instruct \
    --dataset hotpotqa \
    --data_path data/hotpotqa \
    --split dev \
    --num_processes 32 \
    --generate_max_length 4096 \
    --sample 1000 \
    --output_dir result/hotpotqa \
    --temperature 0.0 \
    --follow_up_remote_url ${FOLLOW_UP_REMOTE_URL}

python main.py \
    --remote_url ${REMOTE_URL} \
    --method srag-sftv2 \
    --model_name_or_path hf_models/Meta-Llama-3-8B-Instruct \
    --dataset 2wikimultihopqa \
    --data_path data/wikihop/data \
    --split dev \
    --num_processes 64 \
    --generate_max_length 4096 \
    --sample 1000 \
    --output_dir result/wikihop \
    --temperature 0.0 \
    --follow_up_remote_url ${FOLLOW_UP_REMOTE_URL}


python main.py \
    --remote_url ${REMOTE_URL} \
    --method srag-sftv2 \
    --model_name_or_path hf_models/Meta-Llama-3-8B-Instruct \
    --dataset hotpotqa \
    --data_path data/webquestions \
    --split test_with_id \
    --num_processes 64 \
    --generate_max_length 4096 \
    --sample 1000 \
    --output_dir result/webquestions \
    --temperature 0.0 \
    --follow_up_remote_url ${FOLLOW_UP_REMOTE_URL}

python main.py \
    --remote_url ${REMOTE_URL} \
    --method srag-sftv2 \
    --model_name_or_path hf_models/Meta-Llama-3-8B-Instruct \
    --dataset hotpotqa \
    --data_path data/crag \
    --split test_with_id \
    --num_processes 32 \
    --generate_max_length 4096 \
    --sample 1000 \
    --es_index_name crag \
    --output_dir result/crag \
    --temperature 0.0 \
    --follow_up_remote_url ${FOLLOW_UP_REMOTE_URL}


python3 main.py \
    --remote_url ${REMOTE_URL} \
    --method srag-sftv2 \
    --model_name_or_path hf_models/Meta-Llama-3-8B-Instruct \
    --dataset hotpotqa \
    --data_path data/webquestions \
    --split test_with_id \
    --num_processes 32 \
    --generate_max_length 4096 \
    --sample 200 \
    --output_dir result/webquestions \
    --temperature 0.0 \
    --follow_up_remote_url ${FOLLOW_UP_REMOTE_URL}


python main.py \
    --remote_url ${REMOTE_URL} \
    --method srag-sftv2 \
    --model_name_or_path hf_models/Meta-Llama-3-8B-Instruct \
    --dataset hotpotqa \
    --data_path data/popqa \
    --split test_with_id \
    --num_processes 32 \
    --generate_max_length 4096 \
    --sample 1000 \
    --output_dir result/popqa \
    --temperature 0.0 \
    --follow_up_remote_url ${FOLLOW_UP_REMOTE_URL}




python3 main.py \
    --remote_url ${REMOTE_URL} \
    --method srag-sftv2 \
    --model_name_or_path hf_models/Meta-Llama-3-8B-Instruct \
    --dataset 2wikimultihopqa \
    --data_path data/musique \
    --split dev \
    --num_processes 32 \
    --generate_max_length 4096 \
    --sample 1000 \
    --output_dir result/musique \
    --temperature 0.0 \
    --follow_up_remote_url ${FOLLOW_UP_REMOTE_URL}