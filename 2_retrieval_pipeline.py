import os

from langchain_chroma import Chroma
from langchain_aws import BedrockEmbeddings, ChatBedrockConverse
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv



load_dotenv()

persistent_directory = "db/chroma_db"

# Must match the model used in ingestion_pipeline.py, or the stored 1024-dim
# vectors cannot be compared against the query vector.
embedding_model = BedrockEmbeddings(
    model_id=os.getenv("BEDROCK_EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0"),
    region_name=os.getenv("AWS_REGION", "ap-south-1")
)

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}  
)

# Search for relevant documents
# query = "How much did Microsoft pay to acquire GitHub?"
query = "What was NVIDIA's first graphics accelerator called?"
# query = "Which company did NVIDIA acquire to enter the mobile processor market?"
# query = "What was Microsoft's first hardware product release?"
# query = "How much did Microsoft pay to acquire GitHub?"
# query = "In what year did Tesla begin production of the Roadster?"
# query = "Who succeeded Ze'ev Drori as CEO in October 2008?"
# query = "What was the name of the autonomous spaceport drone ship that achieved the first successful sea landing?"
# query = "What was the original name of Microsoft before it became Microsoft?"

retriever = db.as_retriever(search_kwargs={"k": 5})

# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={
#         "k": 5,
#         "score_threshold": 0.3  # Only return chunks with cosine similarity ≥ 0.3
#     }
# )

relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")
# Display results
print("--- Context ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")

#! this is the code for the answer genaration with LLM  : Start
# Combine the query and the relevant document contents
combined_input = f"""Based on the following documents, please answer this question: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""

# Create a Bedrock chat model. Most current models are only reachable through a
# cross-region inference profile, hence the "apac." prefix on the model id.
model = ChatBedrockConverse(
    model=os.getenv("BEDROCK_CHAT_MODEL", "apac.anthropic.claude-3-7-sonnet-20250219-v1:0"),
    region_name=os.getenv("AWS_REGION", "ap-south-1"),
    temperature=0,
    max_tokens=1024,
)

# Define the messages for the model
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]

# Invoke the model with the combined input
result = model.invoke(messages)

# Display the full result and content only
print("\n--- Generated Response ---")
# print("Full result:")
# print(result)
print("Content only:")
print(result.content)

#! this is the code for the answer genaration with LLM  : END

# Synthetic Questions: 

# 1. "What was NVIDIA's first graphics accelerator called?"
# 2. "Which company did NVIDIA acquire to enter the mobile processor market?"
# 3. "What was Microsoft's first hardware product release?"
# 4. "How much did Microsoft pay to acquire GitHub?"
# 5. "In what year did Tesla begin production of the Roadster?"
# 6. "Who succeeded Ze'ev Drori as CEO in October 2008?"
# 7. "What was the name of the autonomous spaceport drone ship that achieved the first successful sea landing?"
# 8. "What was the original name of Microsoft before it became Microsoft?"