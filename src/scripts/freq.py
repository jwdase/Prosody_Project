import pandas as pd
import matplotlib.pyplot as plt

from language_detection.data.csv_tools.read import select_array, convert_to_list
from language_detection.data.audio import get_length

languages = ["en", "de", "it", "es"]

sizes = {}

for lang in languages:
    base = f"/om2/user/moshepol/prosody/data/raw_audio/{lang}/custom/length.csv"
    df = pd.read_csv(base)

    x = []

    for val in df["5.5 - 6.0"]:
        for link in convert_to_list(val):
            x.append(get_length(link))

    sizes[lang] = x.copy()

    print(f'Finished: {lang}')


datasets = [item for _, item in sizes.items()]
fig, axs = plt.subplots(2, 2, figsize=(15, 10))  

for i, lang in enumerate(languages):
    axs.flat[i].hist(sizes[lang], bins=20, color='skyblue', edgecolor='black')
    axs.flat[i].set_title(f"{lang.upper()} Length Distribution")
    axs.flat[i].set_xlabel("Duration (seconds)")
    axs.flat[i].set_ylabel("Count")

plt.tight_layout()
plt.savefig("histogram.pdf", format="pdf")
plt.close()

