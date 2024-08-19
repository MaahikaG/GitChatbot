from pinecone import Pinecone
import re
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time

pc = Pinecone(api_key="64b34097-f3d5-4ee0-91d9-51ea02279463")
index = pc.Index("versionwise")

model = SentenceTransformer('all-MiniLM-L6-v2')


def process_pdf(file_path):
    # create a loader
    loader = PyPDFLoader(file_path)
    # load your data
    data = loader.load()
    print(f"Total pages loaded: {len(data)}")
    # Split your data up into smaller documents with Chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    documents = text_splitter.split_documents(data)
    print(f"Number of documents after splitting: {len(documents)}")
    # Convert Document objects into strings
    texts = [doc.page_content for doc in documents] 
    for i, text in enumerate(texts):  # Print the first 3 texts
        texts[i] = text.replace("\t", " ").replace("\n", " ")
    return texts

# Define a function to create embeddings
def create_embeddings(texts):
    embedding = model.encode(texts).tolist()
    return embedding


# Process a PDF and create embeddings
#file_path = "https://githubtraining.github.io/training-manual/book.pdf"  # Replace with your actual file path
file_path = "https://people.computing.clemson.edu/~jmarty/courses/commonCourseContent/common/progit.pdf"
texts = process_pdf(file_path)
embeddings = create_embeddings(texts)
ids = [f"{file_path}_chunk_{i}" for i in range(len(embeddings))]


print(f"Number of texts: {len(texts)}")
print(f"Number of embeddings: {len(embeddings)}")
print(f"Number of ids: {len(ids)}")

vectors = [
    {
        "id": id,
        "values": emb,
        "metadata": {"text": text}
    }
    for id, emb, text in zip(ids, embeddings, texts)
]

print(f"Number of vectors to upsert: {len(vectors)}")
for i in range(0, len(vectors), 100):
        batch = vectors[i:i + 100]
        index.upsert(vectors=batch, namespace="git_book2")
        # Debug: Confirm the batch upsert
        print(f"Upserted batch {i // 100 + 1} with {len(batch)} vectors")
        index_stats = index.describe_index_stats()
        print(index_stats)
        # Optional: Add a sleep to avoid rate limiting
        time.sleep(1)





