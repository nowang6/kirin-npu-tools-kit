import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.constants import model_path, dopt_config, quanted_ckpt, onnx_save_path, quant_params_output_dir
from dopt.dopt_lm.do_opt import (
    generate_config_file,   ## 生成量化配置文件
    optimize_model,         ## 使用生成和配置好的量化配置文件将浮点nn.module 转成插入量化算子的nn.module
    set_quant_state,        ## 分别对激活和量化设置量化推理使能
    set_calibrate_state,    ## 设置量化参数可更新状态
    generate_quant_params,  ## 导出量化参数接口
)


def _load_model_and_tokenizer(model_path: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = _load_model_and_tokenizer(model_path)

    model = optimize_model(model, dopt_config)
    model.eval()

    ## 打开量化器
    set_quant_state(model, weight_state=True, input_state=True)
    ## 打开量化参数可标定状态 
    set_calibrate_state(model, True)  
    ## 使用实际推理数据进行推理 即量化标定
    with open(calib_dataset, encoding="utf-8") as f:
        datasets = json.load(f)
    for row in datasets:
        text = row["text"]
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(model.device)
        model(**inputs)
        
    set_quant_state(model, weight_state=True, input_state=True)
    set_calibrate_state(model, False)
    generate_quant_params(
        model,
        output_dir,
        quant_param_2=False,
        embedding_separate=True,
    )