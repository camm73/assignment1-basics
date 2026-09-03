from cs336_basics.bpe_train import cam_train_bpe

# train_bpe_tinystories (a)
cam_train_bpe('../data/TinyStoriesV2-GPT4-train.txt', 10000, ['<|endoftext|>'])
