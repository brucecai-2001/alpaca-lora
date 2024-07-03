import os
import sys

import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

# 假设 utils.callbacks 和 utils.prompter 模块已经定义了 Iteratorize, Stream, Prompter 等类和函数
from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

def load_model(base_model, lora_weights, load_8bit, device):
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )
    return model

def generate_text(instruction, input=None, **kwargs):
    global model, tokenizer, prompter

    prompt = prompter.generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    generation_config = GenerationConfig(max_length=50, **kwargs)

    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    return prompter.get_response(output)

def get_user_input_and_generate_text():
    print("请输入您的指令（输入'exit'退出程序）：")
    instruction = input()
    while instruction.lower() != 'exit':
        input_text = input("请输入附加的输入文本（可选，直接按回车跳过）：")
        generated_text = generate_text(instruction, input_text)
        print("\n生成的文本：")
        print(generated_text)
        print("\n请输入下一条指令（输入'exit'退出程序）：")
        instruction = input()



load_8bit: bool = False
base_model: str = "linhvu/decapoda-research-llama-7b-hf"
lora_weights: str = "/root/autodl-tmp/alpaca_lora/checkpoint-5"
prompt_template: str = "alpaca"  # The prompt template to use, will default to alpaca.
base_model = base_model or os.environ.get("BASE_MODEL", "")
assert base_model, "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

prompter = Prompter(prompt_template)
tokenizer = LlamaTokenizer.from_pretrained(base_model)
    
# 根据设备类型加载模型
model = load_model(base_model, lora_weights, load_8bit, device)

# 修复模型配置
model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
model.config.bos_token_id = 1
model.config.eos_token_id = 2

if not load_8bit:
    model.half()  # 修复某些用户的问题

model.eval()


if __name__ == "__main__":
    get_user_input_and_generate_text()
