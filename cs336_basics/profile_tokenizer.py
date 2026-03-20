from .tokenizer import TrainBpe
if __name__ == '__main__':
    v, m = TrainBpe('/home/ljx123/cs336/assignment1-basics/data/TinyStories-valid.txt', 1000, ["<|endoftext|>"])