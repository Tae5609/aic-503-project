from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv("../dataset/train_label.csv")

all_text = " ".join(df["caption"].astype(str)).lower().split()
counter = Counter(all_text)

common = counter.most_common(20)
print("Common",common)
words, counts = zip(*common)

plt.figure()
plt.bar(words, counts)
plt.xticks(rotation=45)
plt.title("Top Word Frequency in Captions")

# SAVE
save_path = "./word_frequency.png"
plt.savefig(save_path, bbox_inches="tight", dpi=300)
plt.show()

print(f"Saved figure to: {save_path}")
