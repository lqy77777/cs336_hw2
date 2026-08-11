import os
from cs336_basics.bpe import train_bpe, Tokenizer
import time
import numpy as np
import json
from itertools import islice


def save_bpe(vocab, merges, special_tokens, output_path, meta_data = None):
    tokenizer_data = {
        "format_version": 1,
        "special_tokens": special_tokens,
        "vocab": {
            str(token_id): token_bytes.hex()
            for token_id, token_bytes in vocab.items()
        },
        #json的key必须是字符串，且不能保存bytes格式，需要用.hex()转化为16进制
        "merges": [
            [left.hex(), right.hex()]
            for left, right in merges
        ],
        "meta_data": meta_data or {}
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_data, f, indent=2)

def iter_documents(
    file,  #已经以文本模式打开的文件对象
    delimiter="<|endoftext|>",
    chunk_size=1024 * 1024,
):
    '''从一个很大的文本文件中分块读取内容，并按照 <|endoftext|> 将文件逐篇切分成文档。
    定义一个生成器函数。调用它时不会立即读取文件，只有开始遍历返回的生成器时，函数才真正执行。
    '''
    buffer = ""

    while chunk := file.read(chunk_size):
        buffer += chunk
        parts = buffer.split(delimiter)

        for document in parts[:-1]:
            yield document + delimiter

        buffer = parts[-1]

    if buffer:
        yield buffer

def write_token_ids(
    tokenizer,
    input_path,
    output_path,
    dtype,
    batch_size=1_000_000,
):
    """
    从大型文本文件中逐篇读取文档。
    使用 tokenizer 将文本转换成 token ID。
    每次收集固定数量的 token ID。
    将它们以 uint16 二进制格式分批写入磁盘。
    返回总 token 数量。
    """
    total_tokens = 0

    with open(input_path, "r", encoding="utf-8") as source:
        documents = iter_documents(source)  #生成器
        token_iterator = tokenizer.encode_iterable(documents)

        with open(output_path, "wb") as output:
            while True:
                token_batch = np.fromiter(
                    islice(token_iterator, batch_size),
                    dtype=dtype,
                )

                if token_batch.size == 0:
                    break

                token_batch.tofile(output)
                total_tokens += token_batch.size
    return total_tokens

def main():
    os.makedirs("tokens_id", exist_ok=True)

    vocab_size = 10000  #uint16可以表示0-65535
    special_tokens = ['<|endoftext|>']
    num_processes = 10
    token_dtype = np.uint16  #uint16可以表示0-65535

    input_path = 'data/TinyStoriesV2-GPT4-valid.txt'
    tokenizer_path = 'tokens_id/tinystories_valid_tokenizer.json'
    output_path = 'tokens_id/tinystories_valid_tokens.bin'


    t = time.perf_counter()
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens,num_processes)
    elapsed = time.perf_counter() - t
    print(f"Training time: {elapsed:.2f} seconds")

    tokenizer = Tokenizer(vocab, merges, special_tokens)
    t = time.perf_counter()
    total_tokens = write_token_ids(tokenizer,input_path,output_path,token_dtype)
    encode_time = time.perf_counter() - t
    print(f"Encoding time: {encode_time:.2f} seconds")
    print(f"Total tokens: {total_tokens:,}")

    metadata = {
        "dtype": np.dtype(token_dtype).name,
        "shape": [total_tokens],
        "vocab_size": len(vocab),
        "token_path": os.path.basename(output_path),
    }
    save_bpe(vocab,merges,special_tokens,tokenizer_path, metadata)
if __name__ == '__main__':
    main()