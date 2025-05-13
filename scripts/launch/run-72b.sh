CEPH_HOME=/mnt/geminiszgmcephfs/geminicephfs/pr-others-prctrans/xinyanguan

MODEL_PATH=$CEPH_HOME/hf_models/Qwen2.5-72B-Instruct


CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python3 -m vllm.entrypoints.openai.api_server --model $MODEL_PATH --tensor-parallel-size 4 --port 8000  > deploy_qwen_8000.log 2>&1 &

