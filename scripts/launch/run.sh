CEPH_HOME=~
MODEL_PATH=$CEPH_HOME/hf_models/Meta-Llama-3-8B-Instruct



for i in `seq 0 7`; do
    CUDA_VISIBLE_DEVICES=$i nohup python3 -m vllm.entrypoints.openai.api_server --model $MODEL_PATH --tensor-parallel-size 1 --port 800$i  > deploy_llama_800$i.log 2>&1 &
done
