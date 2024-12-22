import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import json
model = SentenceTransformer('all-MiniLM-L6-v2')
train_set_path = ''
test_set_path = ''
output_path = ''
k = 50
train_set = json.load(open(train_set_path))
test_set = json.load(open(test_set_path))

sentences = [line['question'] for line in train_set]
embeddings = model.encode(sentences)

dimension = embeddings.shape[1] 
index = faiss.IndexFlatL2(dimension) 
index.add(embeddings)  

sim_index = []
for line in tqdm(test_set):
    query = line['question']
    try:
        query_embedding = model.encode([query])
        distances, indices = index.search(query_embedding, k)
        sim_index.append({
            'question': query,
            'sim_indices': [int(idx) for idx in indices[0]],
            'sim_questions': [sentences[idx] for idx in indices[0]]
        })
    except:
        sim_index.append({
            'question': query,
            'sim_indices': [],
            'sim_questions': []
        })
    json.dump(sim_index, open(output_path, 'w'), indent=4)
