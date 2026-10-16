import os
import pickle
from .rag import retrieve
from .model import generate_answer, analyze_image

VECTOR_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vectorstore")

# Load image->answer lookup if available
image_index = {}
image_index_path = os.path.join(VECTOR_DIR, "image_index.pkl")
if os.path.exists(image_index_path):
    with open(image_index_path, "rb") as f:
        image_index = pickle.load(f)

def rag_pipeline(query, image_file=None, image_filename=None):
    """Medical Research Q&A with image understanding and RAG retrieval"""
    try:
        # 1. If image is uploaded, build context from it
        image_context = None

        # Check dataset index first (exact filename match)
        if image_filename:
            fname = os.path.basename(image_filename)
            if fname in image_index:
                stored = image_index[fname]
                if stored.get("answer"):
                    image_context = stored["answer"]

        # If not in dataset index, use vision model to describe image
        if image_file is not None and image_context is None:
            image_description = analyze_image(image_file)
            if image_description:
                image_context = image_description

        # 2. Build the search query (combine image context + user query)
        search_query = query
        if image_context:
            search_query = f"{image_context}. {query}"

        # 3. Retrieve relevant medical documents
        docs = retrieve(search_query, k=8)

        # 4. Build the prompt based on what we have
        if image_context and docs:
            # Image + retrieved docs → full answer
            context = "\n\n---\n\n".join(docs)
            prompt = f"""You are a knowledgeable medical research assistant.

Image Finding: {image_context}

Relevant Medical Literature:
{context}

User Question: {query}

Using the image finding and the medical literature above, provide a clear, accurate, and helpful answer. 
Explain medical terms in simple language where possible."""

        elif image_context and not docs:
            # Image only → answer from image context alone
            prompt = f"""You are a knowledgeable medical research assistant.

Image Finding: {image_context}

User Question: {query}

Based on the image finding above, provide a clear, accurate, and helpful answer.
Explain medical terms in simple language where possible."""

        elif docs and not image_context:
            # No image → RAG only
            context = "\n\n---\n\n".join(docs)
            prompt = f"""You are a knowledgeable medical research assistant.

Relevant Medical Literature:
{context}

User Question: {query}

Using the medical literature above, provide a clear, accurate, and helpful answer.
Explain medical terms in simple language where possible."""

        else:
            # Nothing retrieved, nothing from image
            return (
                "❌ No relevant medical information found. Try:\n"
                "- Uploading a medical image\n"
                "- Asking about a specific condition or symptom\n"
                "- Rephrasing your question"
            )

        # 5. Generate the answer
        answer = generate_answer(prompt, query)

        # 6. Auto-add simple explanation for complex queries
        if any(word in query.lower() for word in ["explain", "meaning", "why", "what does", "interpret", "what is"]):
            explanation_prompt = f"""The following is a medical answer:

{answer}

Now rewrite this in simple, easy-to-understand language for a patient with no medical background. 
Keep it concise and reassuring."""
            simple = generate_answer(explanation_prompt, "Simple explanation")
            answer += f"\n\n---\n💡 **In Simple Terms:** {simple}"

        return answer

    except FileNotFoundError:
        return "❌ **Setup Required:** Run `python backend/ingest.py` first to build the vectorstore."
    except Exception as e:
        return f"❌ **Pipeline Error:** {str(e)}"