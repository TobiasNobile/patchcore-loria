from datasets import load_dataset

ds = load_dataset("flwrlabs/celeba")
print(ds)                     # train / valid / test
print(ds["train"][0].keys())