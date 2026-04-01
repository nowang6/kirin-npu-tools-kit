mkdir out
export MODEL_NAME=Qwen2.5-0.5B-Instruct
export MODEL_BASE_PATH=/data/models
cp onnx_embedding_out_no_output_pos/model_64_2048.embedding_dequant_scale out/
cp onnx_embedding_out_no_output_pos/model_64_2048.embedding_weights out/
cp omc_out/* out/
cp ${MODEL_BASE_PATH}/${MODEL_NAME}/tokenizer.json out/
cp app_config/${MODEL_NAME}.json out/executor.json
cp app_config/context.json out/context.json