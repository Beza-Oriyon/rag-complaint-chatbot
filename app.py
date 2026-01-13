import os
import chromadb
import gradio as gr
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_core.prompts import PromptTemplate

from langchain_huggingface import HuggingFacePipeline
# ─── Use absolute path (copy-paste your exact path) ───────────────────────────
db_path = r"C:\Users\bezaw\OneDrive\Desktop\10Acadamy-KAIM\RAG-Compliant-Chatbot\rag-complaint-chatbot\vector_store\chroma_db"

print(f"DEBUG: Using absolute path: {db_path}")
print("Folder exists?", os.path.exists(db_path))
if os.path.exists(db_path):
    print("Files in folder:", os.listdir(db_path))

client = chromadb.PersistentClient(path=db_path)

# Debug collections
print("DEBUG: Available collections:")
for c in client.list_collections():
    print(f" - {c.name} ({c.count()} vectors)")

# Force get_or_create (fixes most "not found" bugs)
embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

collection = client.get_or_create_collection(
    name="complaints_sample",
    embedding_function=embedding_function
)

print(f"Loaded collection '{collection.name}' with {collection.count()} vectors")

# ───────────────────────────────────────────────────────────────────────────────
#  PROMPT TEMPLATE
# ───────────────────────────────────────────────────────────────────────────────

prompt_template = PromptTemplate.from_template(
    """You are a financial analyst assistant for CrediTrust Financial. Your task is to answer questions about customer complaints using ONLY the provided context.

Be concise, insightful, and evidence-based. Cite specific examples from the context. If the context does not contain enough information, say: "I don't have enough information from the complaints to answer this."

Context:
{context}

Question: {question}

Answer:"""
)

# ───────────────────────────────────────────────────────────────────────────────
#  LLM (using GPT-2 as loaded in notebook)
# ───────────────────────────────────────────────────────────────────────────────

llm = HuggingFacePipeline.from_model_id(
    model_id="gpt2",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 300,
        "temperature": 0.7,
        "repetition_penalty": 1.2,
        "do_sample": True,
    },
    device=-1  # Force CPU
)

print("LLM (GPT-2) ready")

# ───────────────────────────────────────────────────────────────────────────────
#  RETRIEVAL FUNCTION
# ───────────────────────────────────────────────────────────────────────────────

def retrieve(question, k=6, product_filter=None):
    where = {"product_category": product_filter} if product_filter else None
    
    results = collection.query(
        query_texts=[question],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )
    
    context_chunks = []
    sources = []
    
    for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
        context_chunks.append(doc)
        sources.append(
            f"Complaint {meta['complaint_id']} ({meta['product_category']}) "
            f"- Chunk {meta['chunk_index']+1}/{meta['total_chunks']} "
            f"(distance: {dist:.4f})"
        )
    
    context = "\n\n".join(context_chunks)
    return context, sources

# ───────────────────────────────────────────────────────────────────────────────
#  MAIN RAG FUNCTION FOR GRADIO
# ───────────────────────────────────────────────────────────────────────────────

def rag_chat(question, product_filter):
    if not question.strip():
        return "Please enter a question.", ""
    
    filter_val = None if product_filter == "All Products" else product_filter
    
    try:
        context, sources = retrieve(question, k=6, product_filter=filter_val)
    except Exception as e:
        return f"Error during retrieval: {str(e)}", ""
    
    prompt = prompt_template.format(context=context, question=question)
    
    try:
        response = llm.invoke(prompt)
        answer = response.strip()
        
        # Clean up GPT-2 output a bit
        if prompt in answer:
            answer = answer.replace(prompt, "", 1).strip()
        if len(answer) > 1200:
            answer = answer[:1200] + "... (truncated)"
    except Exception as e:
        return f"Error generating answer: {str(e)}", ""
    
    sources_md = f"**Sources used ({len(sources)}):**\n\n" + "\n".join([f"• {s}" for s in sources])
    
    return answer, sources_md

# ───────────────────────────────────────────────────────────────────────────────
#  GRADIO INTERFACE
# ───────────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="CrediTrust Complaint Chatbot", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # CrediTrust Complaint Chatbot  
        Ask questions about customer complaints in **Credit Cards**, **Personal Loans**, **Savings Accounts**, and **Money Transfers**.
        
        **Examples**:
        - Why are customers unhappy with credit card late fees?
        - What are common complaints about Personal Loans?
        - Summarize fraud issues in Money Transfers
        """
    )
    
    with gr.Row():
        with gr.Column(scale=3):
            question_input = gr.Textbox(
                label="Your question",
                placeholder="e.g., Why are people complaining about late fees on credit cards?",
                lines=2
            )
            product_dropdown = gr.Dropdown(
                choices=["All Products", "Credit Cards", "Personal Loans", "Savings Accounts", "Money Transfers"],
                label="Filter by product (optional)",
                value="All Products"
            )
            submit_btn = gr.Button("Ask", variant="primary")
        
        with gr.Column(scale=4):
            answer_output = gr.Markdown(label="Answer")
    
    sources_output = gr.Markdown(label="Retrieved Sources (for verification)")

    def submit_fn(question, product):
        answer, sources_md = rag_chat(question, product)
        return answer, sources_md

    submit_btn.click(
        fn=submit_fn,
        inputs=[question_input, product_dropdown],
        outputs=[answer_output, sources_output]
    )

    gr.Examples(
        examples=[
            ["Why are customers unhappy with credit card late fees?", "Credit Cards"],
            ["What are common issues with Personal Loans?", "Personal Loans"],
            ["Summarize fraud complaints in Money Transfers", "Money Transfers"],
            ["Compare Savings Accounts and Credit Cards complaints", "All Products"]
        ],
        inputs=[question_input, product_dropdown],
        outputs=[answer_output, sources_output],
        label="Quick examples"
    )

if __name__ == "__main__":
    demo.launch(share=False)  # Change to True for temporary public link