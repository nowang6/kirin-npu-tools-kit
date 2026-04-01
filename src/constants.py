"""Qwen2.5-0.5B-Instruct DOPT 默认路径与输出位置。"""

model_path: str = "/data/models/Qwen2.5-0.5B-Instruct"
dopt_config: str = "conf/qwen25_dopt.json"
quanted_ckpt: str = "out_put/trained.pth"
onnx_save_path: str = "out_pub/model.onnx"
quant_params_output_dir: str = "output/dopt"
calib_dataset: str = "data/c4_demo.json"
