import os
from typing import BinaryIO
import regex as re
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from multiprocessing import Pool
import json
from functools import lru_cache

#gpt-2正则
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks
    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def count_chunk(input_path: str | os.PathLike, start, end, escaped):
    local_freq = Counter()
    with open(input_path, 'rb') as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore") #跳过无法解码的字节
        for para in re.split(escaped, chunk):
            for matched in re.finditer(PAT, para):
                #matched是一个正则匹配对象,.group()返回匹配到的字符串bytes对象
                pretoken = matched.group()
                #迭代一个 bytes 对象,取出来的每个元素是 int(0-255 之间的数值),不是 bytes
                key = tuple(bytes([i]) for i in pretoken.encode('utf-8'))
                local_freq[key] += 1
    return local_freq
        

def train_bpe(
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
        num_processes = 4,    #进程数
        **kwargs,
) -> tuple[dict[int,bytes], list[tuple[bytes, bytes]]]:
    #1.先创建初始词表(256+special tokens)
    vocab = {i : bytes([i]) for i in range(256)}    #bytes函数的用法
    for k,v in enumerate(special_tokens):
        vocab[256 + k] = v.encode('utf-8')
    index = 256 + len(special_tokens)   #记录下一个新词的索引

    #2.读取文件,确定分割点,然后pre-tokenization
    frequency = Counter()  #pre-tokenization后的frequecy_table
    #转义后的special_tokens
    escaped = '|'.join([re.escape(s) for s in special_tokens])
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    with Pool(num_processes) as pool: #多进程
        args = [(input_path, start, end, escaped) for start, end in zip(boundaries[:-1], boundaries[1:])]
        freqs = pool.starmap(count_chunk,args) #starmap和map有细微差别
    #每个进程返回一个局部的 Counter。所以 freqs 是一个列表，列表的每一个元素是一个counter
    for local_freq in freqs:
        frequency.update(local_freq)
    #3.merge
    #关键：创建pairs_to_frequency字典存储某个pair属于哪些words
    merges = []
    
    pairs = Counter()   #记录相邻字节出现的次数
    pairs_to_frequency = defaultdict(set)
    for key, count in frequency.items():
        if len(key) <= 1:   #特殊情形：单个字节无需处理
            continue
        for i in range(len(key)-1):
            pair = (key[i],key[i+1])   #相邻字节对
            pairs_to_frequency[pair].add(key)
            pairs[pair] += count
    while index < vocab_size:
        #次数打平的话返回字典序高的字节对  #对字典使用max函数得到的是key而不是整个字典
        max_pair = max(pairs, key = lambda x: (pairs[x], x))
        new_token = max_pair[0] + max_pair[1]  #两个字节拼接在一起 
        vocab[index] = new_token 
        merges.append(max_pair)
        #接下来是更新frequency
        #由于更新的过程需要修改key，所以我们先使用的是原本frequency key的副本
        for key in pairs_to_frequency[max_pair].copy():
            count = frequency[key]
            length = len(key)
            temp = []
            i = 0
            while i < length-1:
                pair = (key[i],key[i+1])
                pairs[pair] -= count
                if pair != max_pair:
                    temp.append(key[i])
                else:
                    temp.append(new_token)
                    i += 1
                    if i <= length - 2:
                        pairs[(key[i],key[i+1])] -= count
                if i == length - 2:
                    temp.append(key[i+1])
                i += 1
            for j in range(len(temp)-1):
                pair = (temp[j],temp[j+1])
                pairs[pair] += count
            new_key = tuple(temp)
            frequency[new_key] = count
            frequency.pop(key)
            if len(new_key) == 1:
                pass  #不可能含有pair，所以不必进行下述操作
            else:
                old_pairs = list(zip(key[:-1],key[1:]))
                new_pairs = list(zip(new_key[:-1],new_key[1:]))
                for old in old_pairs:
                    pairs_to_frequency[old].discard(key)
                    if pairs[old] == 0:   #删掉僵尸key
                        pairs.pop(old,0)
                for new in new_pairs:
                    pairs_to_frequency[new].discard(key)
                    pairs_to_frequency[new].add(new_key)
        pairs.pop(max_pair,0)
        if pairs == {}:
            break
        index += 1
         

    return (vocab,merges)

class Tokenizer():
    def __init__(
            self, 
            vocab: dict[int,bytes],
            merges: list[tuple[bytes,bytes]],
            special_tokens: list[str] = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.max = len(merges)
        self.special_tokens = None if special_tokens is None else sorted(special_tokens, key = len, reverse = True)
        self.reversed_vocab = {v : k for k,v in vocab.items()}
        #优先级，我们的方法是对每一个pre-token进行merge，先进行优先级最高的merge，再进行优先级低的merge
        self.rank = defaultdict(lambda: self.max)
        for i in range(len(merges)):
            self.rank[merges[i]] = i
        #括号会让被分割的东西也保留在split返回的列表里
        self.escaped = '(' + '|'.join([re.escape(s) for s in self.special_tokens]) + ')' if special_tokens is not None else None
    @classmethod
    def from_files(
        cls, 
        filepath: str
    ):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = {
            int(token_id): bytes.fromhex(token_hex)
            for token_id, token_hex
            in data["vocab"].items()
        }
        merges = [
            (bytes.fromhex(left), bytes.fromhex(right))
            for left, right in data["merges"]
        ]
        special_tokens = data.get("special_tokens", None)
        return cls(
            vocab=vocab,
            merges=merges,
            special_tokens=special_tokens,
        )
    #由于在merge的过程中，有大量重复的词，所以我们把他们存到缓存里，可以大大加快速度
    @lru_cache(maxsize=100_000)
    def _encode_pretoken(
        self,
        pretoken: str,
    ) -> tuple[int, ...]:
        keys = tuple(bytes([byte_id]) for byte_id in pretoken.encode('utf-8'))
        if len(keys) == 1:
            return (self.reversed_vocab[keys[0]],)
        while True:
            pairs = set(zip(keys[:-1], keys[1:]))
            ranks = {pair: self.rank[pair] for pair in pairs}
            min_pair = min(ranks, key=ranks.get)
            if ranks[min_pair] == self.max: #merge结束
                return tuple(self.reversed_vocab[key] for key in keys)
            temp = []
            new_token = min_pair[0] + min_pair[1]
            i = 0
            while i < len(keys) - 1:
                pair = (keys[i], keys[i + 1])
                if pair != min_pair:
                    temp.append(keys[i])
                else:
                    temp.append(new_token)
                    i += 1
                if i == len(keys) - 2:
                    temp.append(keys[i + 1])
                i += 1
            keys = tuple(temp)
            if len(keys) == 1:
                return (self.reversed_vocab[keys[0]],)
            
    def encode(self, text: str) -> list[int]:
        ids = []
        if self.special_tokens is not None:
            paras = re.split(self.escaped, text)
        else:
            paras = (text,)
        for para in paras:
            if self.special_tokens is not None and para in self.special_tokens:
                special_id = self.reversed_vocab[para.encode('utf-8')]
                ids.append(special_id)
                continue
            for matched in re.finditer(PAT, para):
                    pretoken = matched.group()
                    ids.extend(self._encode_pretoken(pretoken))
        return ids

    def encode_iterable(
            self, 
            iterable: Iterable[str]
    ) -> Iterator[int]:
        def out(inp):
            for string in inp:
                yield from self.encode(string)
        return out(iterable)
    
    def decode(self, ids: list[int]) -> str:
        text = []
        for token in ids:
            text.append(self.vocab[token])
        return b''.join(text).decode('utf-8',errors = 'replace')