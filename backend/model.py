from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import pipeline
import torch
from PIL import Image
import io

# -------------------------------
# Load LLM Model (upgraded to base)
# -------------------------------
print("Loading local LLM model...")
try:
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    llm_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
    llm_model.eval()
    print("Local LLM loaded successfully!")
except Exception as e:
    print(f"Error loading LLM model: {e}")
    tokenizer = None
    llm_model = None

# -------------------------------
# Load Image Model (medical-focused)
# -------------------------------
print("Loading image analysis model...")
try:
    image_classifier = pipeline(
        "image-classification",
        model="microsoft/resnet-50"
    )
    print("Image analysis model loaded successfully!")
except Exception as e:
    print(f"Warning: Image analysis model failed: {e}")
    image_classifier = None

# -------------------------------
# Image Analysis Function
# -------------------------------
def analyze_image(image_file):
    try:
        if image_classifier is None:
            return None

        if isinstance(image_file, bytes):
            image = Image.open(io.BytesIO(image_file))
        else:
            image = Image.open(image_file)

        if image.mode != "RGB":
            image = image.convert("RGB")

        results = image_classifier(image, top_k=3)
        labels = ", ".join([f"{r['label']}" for r in results])
        return labels

    except Exception as e:
        print(f"Image analysis error: {e}")
        return None

# -------------------------------
# Text Answer Generation (FIXED)
# -------------------------------
def generate_answer(context, question):
    if llm_model is None or tokenizer is None:
        return "Error: LLM model failed to load. Please restart the app."

    # Trim context but keep more room for the answer
    context = context[:1500]

    # Stronger prompt that forces detailed answers
    prompt = (
        f"You are a helpful medical research assistant. "
        f"Answer the following question in detail with at least 3 sentences. "
        f"Do not just repeat a label — explain it clearly.\n\n"
        f"Context: {context}\n\n"
        f"Question: {question}\n\n"
        f"Detailed Answer:"
    )

    try:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=768,       # increased from 512
            truncation=True
        )

        with torch.no_grad():
            outputs = llm_model.generate(
                inputs["input_ids"],
                max_new_tokens=300,   # increased from 200
                do_sample=True,       # allow more varied output
                temperature=0.7,      # creativity level
                top_p=0.9,
                num_beams=1,          # faster with do_sample=True
                early_stopping=False,
                repetition_penalty=1.3  # avoid repeating same phrase
            )

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        if not answer or len(answer) < 10:
            return f"{context} — No detailed answer could be generated. Please try rephrasing your question."

        return answer

    except Exception as e:
        return f"Error generating answer: {str(e)}"

# -------------------------------
# Post-processing
# -------------------------------
def postprocess_answer(question, answer):
    if any(word in question.lower() for word in ["meaning", "explain", "why", "interpret"]):
        simple_explanation = generate_answer(
            f"Explain the following in simple terms: {answer}",
            question
        )
        return answer + "\n\nSimple Explanation:\n" + simple_explanation
    return answer