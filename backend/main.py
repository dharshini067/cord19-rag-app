from backend.rag import retrieve
from backend.model import generate_answer, analyze_image

def rag_pipeline(query, image_file=None):
    """Main RAG pipeline: analyze image, retrieve docs, and generate answer."""
    try:
        # Analyze image if provided
        image_description = None
        if image_file is not None:
            print("Analyzing uploaded image...")
            image_description = analyze_image(image_file)
            if image_description:
                print(f"Image analysis: {image_description}")
                # Combine image description with user query
                query_with_image = f"{image_description}. {query}"
            else:
                query_with_image = query
        else:
            query_with_image = query
        
        # Retrieve relevant documents
        docs = retrieve(query_with_image)
        
        if not docs:
            return "❌ No relevant documents found for your query. Try rephrasing your question."
        
        context = "\n".join(docs)
        
        # Generate answer
        answer = generate_answer(context, query_with_image)
        return answer
    
    except FileNotFoundError as e:
        return f"❌ **Data Error:** {str(e)}\n\nPlease run 'python backend/ingest.py' to generate the FAISS index."
    
    except RuntimeError as e:
        return f"❌ **Runtime Error:** {str(e)}"
    
    except Exception as e:
        return f"❌ **Unexpected Error:** {str(e)}"