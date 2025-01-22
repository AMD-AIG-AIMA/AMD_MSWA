import os

# import torch
from datasets import load_dataset
from fine_tune import tokenize_fn, DEFAULT_PAD_TOKEN, DEFAULT_BOS_TOKEN, DEFAULT_EOS_TOKEN, DEFAULT_UNK_TOKEN
import transformers
from functools import partial
import huggingface_hub
# import multiprocess.context as ctx


def create_tokenizer(model_name_or_path: str, model_max_length: int):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=True,
    )

    special_tokens_dict = dict()
    # special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    tokenizer.add_special_tokens(special_tokens_dict)

    return tokenizer


def tokenize_fn(tokenizer, example):
    context_length = tokenizer.model_max_length
    outputs = tokenizer(
        tokenizer.eos_token.join(example["text"]),
        truncation=False,
        return_tensors="pt",
        pad_to_multiple_of=context_length,
        padding=True,
    )
    return {"input_ids": outputs["input_ids"].view(-1, context_length)}


def prepare(model_name_or_path: str, cache_dir: str, model_max_length: int, num_proc: int):
    tokenizer = create_tokenizer(
        model_name_or_path=model_name_or_path,
        model_max_length=model_max_length)
    print(len(tokenizer))
    # return
    dataset = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", cache_dir=cache_dir)

    dataset = dataset.shuffle(seed = 127).map(
        partial(tokenize_fn, tokenizer),
        batched=True,
        batch_size=1000,
        # num_proc=16,
        writer_batch_size=1000,
        remove_columns=["text", "meta"])

    print(dataset)
    dataset.save_to_disk("./data/redpajama_1t_sample_4096_new")


def main():
    num_proc = int(os.cpu_count()//2)
    prepare(
        model_name_or_path="/group/ossdphi_algo_scratch_08/zichaoli/models/Llama-2-7b-hf",
        cache_dir="/group/ossdphi_algo_scratch_08/zichaoli/huggingface/datasets",
        model_max_length=4096,
        num_proc=num_proc)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # disable GPU
    # ctx._force_start_method('spawn')
    # torch.set_num_threads(1)
    main()