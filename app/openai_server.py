from flask import Flask, request, jsonify
import base64
from openai import OpenAI
import io
import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

device = "cuda" if torch.cuda.is_available() else "cpu"

app = Flask(__name__)

def get_openai_client():
    try:
        client = OpenAI(
            api_key="AIzaSyAU3nOzJVdRZtnECKSyBmkiOttBbw5hF4o",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        return client
    except Exception as e:
        return None
    
from bs4 import BeautifulSoup
import requests

def scrape_pubmed(keyword):
    try:
        response = requests.get(f"https://pubmed.ncbi.nlm.nih.gov/?term={keyword}")
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        results = []
        for element in soup.select('.docsum-content'):
            title_tag = element.select_one('.docsum-title')
            title = title_tag.text.strip() if title_tag else ''
            link = f"https://pubmed.ncbi.nlm.nih.gov{title_tag['href']}" if title_tag and title_tag.has_attr('href') else ''
            authors = element.select_one('.full-authors').text.strip() if element.select_one('.full-authors') else ''
            snippet = element.select_one('.docsum-snippet').text.strip() if element.select_one('.docsum-snippet') else ''
            abstract = ''

            if link:
                try:
                    article_response = requests.get(link)
                    article_response.raise_for_status()
                    article_soup = BeautifulSoup(article_response.text, 'html.parser')
                    abstract_tag = article_soup.select_one('.abstract-content')
                    abstract = abstract_tag.text.strip() if abstract_tag else ''
                except Exception as e:
                    print(f"Error fetching abstract for {link}: {e}")

            results.append({
                'title': title,
                'link': link,
                'authors': authors,
                'snippet': snippet,
                'abstract': abstract
            })

        return results
    except Exception as e:
        print(f"Error scraping PubMed: {e}")
        return []
    
@app.route('/generate_rag', methods=['POST'])
def generate_rag():
    try:
        data = request.json
        entity_list = data.get('entity_list', [])
        query = data.get('query', '')
        all_articles = []

        for entity in entity_list:
            print(f"Scraping for entity: {entity}")
            articles = scrape_pubmed(entity)
            all_articles.extend(articles)
        
        # Build the context from abstracts
        context = "\n\n".join([article['abstract'] for article in all_articles if article['abstract']])

        if not context.strip():
            return jsonify({"rag_results": "No sufficient abstracts found to generate a response."})

        # Prepare citations
        citations = "\n".join([
            f"- {article.get('title', 'No Title')}: {article.get('link', 'No Link')}"
            for article in all_articles
        ])

        # Query Gemini for answer synthesis
        client = get_openai_client()
        if client is None:
            return jsonify({"error": "Failed to initialize OpenAI client"}), 500

        rag_prompt = f"""
Given the following context from scientific abstracts, answer the query below in 3-4 clear lines, medically accurate and concise.
Context:
{context}

Query:
{query}
"""

        response = client.chat.completions.create(
            model="gemini-1.5-pro",
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical expert providing short, clear, and medically precise answers from the given context."
                },
                {
                    "role": "user",
                    "content": rag_prompt
                }
            ]
        )

        answer = response.choices[0].message.content.strip()
        final_result = f"{answer} \n\n Citations:\n{citations}"

        return jsonify({"rag_results": final_result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
def load_clip_and_get_embedding(base64_string, model_path):
    # Decode base64 image to PIL
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    # Load pre-trained model
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    checkpoint = torch.load(model_path, map_location=device)

    # Handle state_dict structure
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # Load weights
    clip_model.load_state_dict(state_dict, strict=False)
    clip_model = clip_model.to(device).eval()

    # Define processor
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Preprocess image
    inputs = processor(images=image, return_tensors="pt").to(device)

    # Get image embedding
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    return image_features.cpu().numpy().squeeze()

def search_similar_image(query_embedding, triplet_csv_path, model_path):
    # Load triplets
    df = pd.read_csv(triplet_csv_path)

    image_embeddings = []
    image_ids = []

    for idx, row in df.iterrows():
        if row['predicate'] == "imageof":
            base64_img = row['subject']
            try:
                embedding = load_clip_and_get_embedding_from_base64(base64_img, model_path)
                image_embeddings.append(embedding)
                image_ids.append(base64_img)  # Or row['object'] if you want the node
            except Exception as e:
                print(f"Failed to process image at row {idx}: {e}")

    # Compute cosine similarity
    image_embeddings = np.array(image_embeddings)
    similarities = cosine_similarity([query_embedding], image_embeddings).squeeze()

    # Get the most similar image
    best_idx = similarities.argmax()
    return image_ids[best_idx], similarities[best_idx]


@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    try:
        # Get the request data
        data = request.json
        is_image_available = data.get("is_image_available")
        user_text = data.get('user_text', "Identify medical conditions or symptoms?")
        print(data,is_image_available)

        messages = [
            {
                "role": "system",
                "content": """
You are a dermatologist analyzing skin conditions with medical precision. Examine the image carefully and identify the most likely condition based on common patterns. Provide a clear, concise diagnosis in simple terms, stating whether it matches the user's suspicion, is a milder issue, or indicates something more serious. Ensure medical accuracy and avoid unnecessary details or extra text. Try to return the condition in a single sentence without any common terms like infection, inflammation, etc unless it is a common term for the condition.
                """
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                ],
            }
        ]
        
        model_path = "C:\Users\bhav1\Downloads\app\DermaBCK\epoch_20.pt"
        multimodal_graph = "C:\Users\bhav1\Downloads\app\DermaBCK\hybrid_mapping.csv"
        # If image is available, add image data to the messages
        if is_image_available:
            base64_image = data.get('image')
            messages[1]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })
        # Search similar image using clip embedding
        image_triplets = search_similar_image(clip_embeddings, multimodal_graph, model_path)

        # Add the image triplet result to the message content
        messages[1]["content"].append({
            "type": "text",
            "text": f"Most relevant image found in graph with similarity: {image_triplets[1]:.4f}"
        })

        print(data,is_image_available,messages)
        # Initialize OpenAI client
        client = get_openai_client()
        if client is None:
            return jsonify({"error": "Failed to initialize OpenAI client"}), 500

        # Make request to OpenAI
        response = client.chat.completions.create(
            model="gemini-1.5-pro",
            messages=messages,
        )
        print(data,is_image_available,messages,response)

        # Return the response
        return jsonify({"result": response.choices[0].message.content})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/validate_diagnosis', methods=['POST'])
def validate_diagnosis():
    try:
        # Get the request data
        data = request.json
        full_response = data.get('text', "")
        
        # Initialize OpenAI client
        client = get_openai_client()
        if client is None:
            return jsonify({"error": "Failed to initialize OpenAI client"}), 500
        
        # Make request to OpenAI using Gemini 2.0 Flash Lite
        response = client.chat.completions.create(
          model="gemini-2.0-flash-lite",
          messages=[
            {
              "role": "system",
              "content": """
Review the given response, remove unnecessary text like introductions or summaries, and refine it into a single, friendly yet professional paragraph. Ensure it clearly presents the likely diagnosis, key symptoms, risk factors, diagnostic methods, treatment, prevention, and patient care. Keep it concise, medically accurate, and engaging, avoiding redundancy or filler words.
When more than one disease is given to you, choose the most likely one and generate for it only. Try to start with phrases like "It is most likely..."  or similar phrases to introduce the condition naturally while maintaining a structured yet fluid response.
              """
            },
            {
              "role": "user",
              "content": f" {full_response}"
            }
          ]
        )
        
        # Return the response
        return jsonify({"result": response.choices[0].message.content})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001) 