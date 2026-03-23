import os
from typing import BinaryIO, Iterable, Iterator
from .pretokenization_example import find_chunk_boundaries
import regex as re
from collections import Counter
import multiprocessing as mp
import pickle
import cProfile
import pstats
import json
'''
1. 先把文本划分成一些pre token。统计pre token的出现频率
2. 统计pre token内部byte pair的出现频率
3. 把最高频的pair合成成一个token
'''
def UpdateCount(freCounter:Counter[tuple[bytes]], bpCounter:Counter):
    for text, times in freCounter.items():
        for i in range(0,len(text)-1):
            bpCounter[(text[i],text[i+1])]+=times
    return bpCounter

def UpdateFre(freCounter:Counter[tuple[bytes]], b1:bytes, b2:bytes)->Counter:
    newCounter = Counter()
    for tup, times in freCounter.items():
        newList = []
        i = 0
        while i<len(tup):
            if i<len(tup)-1 and tup[i]==b1 and tup[i+1]==b2:
                newList.append(b1+b2)
                i+=2
            else:
                newList.append(tup[i])
                i+=1
        newCounter[tuple(newList)] = times
    return newCounter

def PreTokenize(chunk:str, special_tokens: list[str])->Counter:
    # 按照speical_tones进行分割
    if special_tokens:
        escapeToks = [re.escape(tok) for tok in special_tokens]
        pattern = '|'.join(escapeToks)
        segs = re.split(pattern, chunk)
    else:
        segs = [chunk]
    counter = Counter()
    for seg in segs:
        if seg == "":
            continue
        #pre tokenize
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        matches = re.finditer(PAT, seg)
        for m in matches:
            text = m.group().encode('utf-8')
            # 很重要，把简单的字符串'abc'分成了('a','b','c')，有利于后续的合并操作
            t = tuple(bytes([i]) for i in text)
            counter[t]+=1
    return counter

def _worker(input_path:str, start, end, special_tokens)->Counter:
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        return PreTokenize(chunk, special_tokens)

def TrainBpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab: dict[int, bytes] = {}
    nextId = 0
    merges = []
    # 初始化vocab
    for i in range(0, 256):
        vocab[i] = bytes([i])
    nextId = 256
    for st in special_tokens:
        vocab[nextId] = st.encode()
        nextId+=1
    # train
    freCounter:Counter[tuple[bytes]] = Counter() # 逻辑上应该是bytes的counter，但是实际上是tuple of bytes的counter
    numProcess = os.cpu_count()
    
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, numProcess, b"<|endoftext|>")
    # multiprocessing 
    tasks = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        tasks.append((input_path, start, end, special_tokens))
    with mp.Pool(processes=numProcess) as pool:
        results = pool.starmap(_worker, tasks)
    # TODO:尝试异步获取结果
    for r in results:
        freCounter+=r
    while len(vocab.keys()) < vocab_size:
        bytePairCounter = Counter()
        # 拿出排行第一的pair
        UpdateCount(freCounter, bytePairCounter)
        bestPairItem = max(bytePairCounter.items(), key=lambda i: (i[1], i[0]))
        if not bestPairItem:
            break
        bestPair, bestFre = bestPairItem
        newWord = bestPair[0]+bestPair[1]
        vocab[nextId] = newWord
        nextId+=1
        merges.append(bestPair)
        freCounter = UpdateFre(freCounter, bestPair[0], bestPair[1])
        # print(f'vocab size = {len(vocab)}, freCounter={freCounter}')
    return vocab, merges

class MyTokenizer:
    def __init__(self, vocab:dict[int,bytes], merges:list[tuple[bytes,bytes]], specialTokens:list[str]=None):
        self.vocab = vocab
        self.vocab_inverse = {v:u for u,v in vocab.items()}
        self.specialTokens = sorted(specialTokens, key=lambda s: len(s), reverse=True) if specialTokens else []
        self.merges = {tup: self.vocab_inverse[tup[0]+tup[1]] for tup in merges}
    
    @classmethod
    def from_files(cls, vocab_path:str, merges_path:str, special_tokens:list[str]):
        with open(vocab_path, 'r') as f:
            data = json.load(f)
            vocab = {v:u.encode() for u,v in data.items()}
        with open(merges_path, 'r') as f:
            lines = f.readlines()
            merges = []
            for line in lines:
                vec = line.strip().split(' ')
                merges.append((vec[0].encode(), vec[1].encode()))
        return cls(vocab, merges, special_tokens)
    
    def PreTokenize(self, text:str)->list[bytes]:
        # 按照speical_tones进行分割
        if self.specialTokens:
            escapeToks = [re.escape(tok) for tok in self.specialTokens]
            pattern = f'({"|".join(escapeToks)})'
            segs = re.split(pattern, text)
        else:
            segs = [text]
        preTokens = []
        for seg in segs:
            if seg == "":
                continue
            if self.specialTokens and  seg in self.specialTokens:
                preTokens.append(seg.encode())
                continue
            #pre tokenize
            PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            matches = re.finditer(PAT, seg)
            for m in matches:
                text = m.group().encode('utf-8')
                preTokens.append(text)
        return preTokens
    
    def encode(self, text: str) -> list[int]:
        preTokens = self.PreTokenize(text)
        preTokensSplit = [list(bytes([i]) for i in token) for token in preTokens]
        # print(f'preTokens={preTokens}')
        ids = []
        for lst, s in zip(preTokensSplit, preTokens):
            if self.specialTokens and  s.decode() in self.specialTokens:
                print(f'{s} is a special token')
                ids.append(self.vocab_inverse[s])
                continue
            while len(lst)>1:
                pairs = [(lst[i], lst[i+1]) for i in range(len(lst)-1)]
                pairIds = [self.vocab_inverse.get(pair[0]+pair[1], None) for pair in pairs]
                if all([pid is None for pid in pairIds]):
                    break
                bestPair = min([(pair, pid) for pair, pid in zip(pairs, pairIds) if pid is not None], key=lambda i: i[1])
                i=0
                newList = []
                while i<len(lst):
                    if i<len(lst)-1 and (lst[i], lst[i+1])==bestPair[0]:
                        newList.append(self.vocab[bestPair[1]])
                        i+=2
                    else:
                        newList.append(lst[i])
                        i+=1
                lst = newList
            ids.extend([self.vocab_inverse[token] for token in lst])
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        return b''.join(self.vocab[i] for i in ids).decode(errors='replace')

        
if __name__ == '__main__':
    pass
