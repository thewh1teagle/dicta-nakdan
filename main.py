"""
wget https://huggingface.co/datasets/thewh1teagle/hebright/resolve/main/knesset.txt.zip
"""
from transformers import AutoModel, AutoTokenizer
import torch
import time
from tqdm import tqdm
import re
import inspect

def remove_niqqud(text: str):
    return re.sub(r"[\u05B0-\u05C7]", "", text)


# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('dicta-il/dictabert-large-char-menaked')
print(tokenizer.model_max_length)
breakpoint()
model = AutoModel.from_pretrained('dicta-il/dictabert-large-char-menaked', trust_remote_code=True)

# Move model to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# Prepare your sentence
sentence = 'יואב היה בחור רגיל לגמרי – גר בדירה קטנה בתל אביב, עבד בתור בריסטה בבית קפה, ושיחק על גיטרה בסופי שבוע כדי לשכנע את עצמו שהוא אמן. כל עוד הקפה טוב – הוא היה מרוצה.'

# Tokenize once (if sentence doesn’t change)
inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True).to(device)
# Warmup (optional but helps for benchmarking)


total = 0
with open('knesset.txt') as fp:
    for line in fp:
        total += 1

BATCH_SIZE = 15
batch = []

with open('knesset.txt') as fp, open('knesset_niqqud.txt', 'w') as out:
    for line in tqdm(fp, total=total):
        line = line.strip()
        if not line:
            continue
        batch.append(line)
        if len(batch) >= BATCH_SIZE:
            try:
                results = model.predict(batch, tokenizer, mark_matres_lectionis = '') # later use | as separator
                # assert all lines without niqqud eaual to the original and show differences
                for i, (original, result) in enumerate(zip(batch, results)):
                    assert remove_niqqud(original) == remove_niqqud(result), f'{remove_niqqud(original)} != {remove_niqqud(result)}'
                out.writelines([r + '\n' for r in results])
                out.flush()
            except Exception as e:
                print(f'Batch error: {e}')
                out.writelines(['\n'] * len(batch))
            batch = []
            

    # Final batch (leftovers)
    if batch:
        try:
            results = model.predict(batch, tokenizer)
            out.writelines([r + '\n' for r in results])
        except Exception as e:
            print(f'Final batch error: {e}')
            out.writelines(['\n'] * len(batch))
    
    