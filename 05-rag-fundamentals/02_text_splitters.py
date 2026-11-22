"""
Module 05 - RAG Fundamentals: 02_text_splitters.py
RecursiveCharacterTextSplitter, chunk size, and overlap tuning.
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter

SAMPLE_TEXT = """
The Transformer is the first transduction model relying entirely on self-attention to compute 
representations of its input and output without using sequence-aligned RNNs or convolution.

In the Transformer, self-attention relates different positions of a single sequence in order 
to compute a representation of the sequence. Multi-head attention allows the model to jointly 
attend to information from different representation subspaces at different positions.

Positional encodings are added to the input embeddings at the bottoms of the encoder and 
decoder stacks. The positional encodings have the same dimension as the embeddings, so that 
the two can be summed.
"""

def main():
    print("=== Tuning RecursiveCharacterTextSplitter ===")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=160,
        chunk_overlap=30,
        separators=["\n\n", "\n", " ", ""],
        length_function=len
    )

    chunks = splitter.create_documents([SAMPLE_TEXT])

    print(f"Generated {len(chunks)} chunks from source text.\n")
    for idx, c in enumerate(chunks, 1):
        print(f"--- Chunk {idx} (Length: {len(c.page_content)} chars) ---")
        print(c.page_content.strip())

if __name__ == "__main__":
    main()
