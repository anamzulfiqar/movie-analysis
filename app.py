import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
import gradio as gr
import zipfile
import os

# Set up the API key for Groq API
API_KEY = "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"
client = Groq(api_key=API_KEY)

# Initialize the embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Function to compress the dataset into a ZIP file
def compress_dataset(input_file):
    output_zip = input_file.replace('.csv', '.zip')  # Create output zip file name
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(input_file, os.path.basename(input_file))
    print(f"Compressed {input_file} into {output_zip}")
    return output_zip

# Compress the dataset
zip_file_path = compress_dataset("Hydra-Movie-Scrape - Hydra-Movie-Scrape.csv")

# Function to load data in chunks and compute embeddings
def load_data_and_compute_embeddings(zip_file_path, chunk_size=1000):
    embeddings = []  # List to hold embeddings
    summaries = []   # List to hold summaries

    # Extract the CSV file from the ZIP for reading
    with zipfile.ZipFile(zip_file_path, 'r') as zipf:
        with zipf.open(os.path.basename("Hydra-Movie-Scrape - Hydra-Movie-Scrape.csv")) as f:
            for chunk in pd.read_csv(f, chunksize=chunk_size):
                chunk_embeddings = chunk['Summary'].apply(
                    lambda x: embedding_model.encode(x).tolist() if pd.notnull(x) else None
                ).tolist()

                embeddings.extend(chunk_embeddings)
                summaries.extend(chunk['Summary'].tolist())

    # Create a DataFrame to store all summaries and embeddings
    df_embeddings = pd.DataFrame({'Summary': summaries, 'embedding': embeddings})
    return df_embeddings

# Load the dataset and compute embeddings
df_embeddings = load_data_and_compute_embeddings(zip_file_path)

# Function to retrieve similar summaries
def retrieve_similar_summaries(query, top_n=5):
    if df_embeddings is None:
        return []

    query_embedding = embedding_model.encode(query).tolist()
    similarities = cosine_similarity([query_embedding], df_embeddings['embedding'].tolist())
    
    top_indices = similarities[0].argsort()[-top_n:][::-1]
    return df_embeddings.iloc[top_indices]['Summary'].tolist()

# Function to generate a response with context
def generate_response_with_context(prompt, context):
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": f"{prompt}\nContext: {context}"}],
        model="llama3-8b-8192",
    )
    return chat_completion.choices[0].message.content

# Complete RAG function for movie analysis
def movie_analysis_rag(query):
    relevant_summaries = retrieve_similar_summaries(query)
    context = "\n".join(relevant_summaries)
    return generate_response_with_context(query, context)

# Gradio interface
def gradio_interface(query):
    return movie_analysis_rag(query)

iface = gr.Interface(
    fn=gradio_interface,
    inputs="text",
    outputs="text",
    title="Movie Analysis App",
    description="Enter a movie title or keyword to get a detailed analysis based on summaries."
)

# Launch the Gradio app
if __name__ == "__main__":
    iface.launch()
