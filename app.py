from flask import Flask, request, jsonify
from langchain.chains import LLMChain
from langchain_community.graphs.networkx_graph import NetworkxEntityGraph, KnowledgeTriple
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
import pandas as pd
import requests
import base64
import os

app = Flask(__name__)

# Initialize graph
graph = NetworkxEntityGraph()
df = pd.read_csv('filtered_dermavqa_nodes.csv', delimiter=',')
# df = pd.read_csv('graph.csv', delimiter='|')
triples = [KnowledgeTriple(row['node_1'], row['edge'], row['node_2']) for _, row in df.iterrows()]
for triple in triples:
    graph.add_triple(triple)

# Initialize LLM
llm = Ollama(model="llama3.1")

# Setup entity extraction chain
entity_prompt = PromptTemplate(
    input_variables=['input'],
    template="""
        Extract all medical entities from the following text. 
        Medical entities include disease name alone, without any other text. Do not include any other text in the output. Also do not include common terms like infection, inflammation, etc.
        Return the output as a single comma-separated list, or NONE if there is nothing to extract.
        DO NOT include any other text in the output.

        Begin!\n\n{input}\nOutput:
    """
)
entity_chain = LLMChain(llm=llm, prompt=entity_prompt)

# Setup QA chain
qa_prompt = PromptTemplate(
    input_variables=['context', 'question', 'user_query'],
    template="""
       You are a dermatology expert providing structured case descriptions for skin conditions. Given the provided knowledge triplets, generate a single, concise paragraph summarizing the condition, ensuring all key aspects are covered: Diagnosis (condition name and clinical features), Predisposing Factors (major risk factors), Diagnostic Methods (tests for confirmation), Treatment Approach (topical and systemic options), Prevention Strategies (recurrence prevention), and Patient Management Recommendations (long-term care, education, adherence). The response must be strictly one paragraph with no line breaks and a maximum of five lines, ensuring medical precision and professionalism. Summarize effectively without speculation or unnecessary details. Start with phrases like "It is likely..." or similar to introduce the condition naturally while maintaining a structured yet fluid response.
       Choose only one disease when multiple diseases are given and generate for it only.
       {context}\n{question}\n{user_query}
    """
)
qa_chain = LLMChain(llm=llm, prompt=qa_prompt)

# Function to save to CSV
def save_to_csv(image_name, query, generated_text):
    if not os.path.exists('generated_data.csv'):
        df = pd.DataFrame(columns=['id', 'imagepath', 'query', 'generated_text'])
        df.to_csv('generated_data.csv', index=False)
    df = pd.read_csv('generated_data.csv')
    next_id = len(df) + 1
    new_row = {'id': next_id, 'imagepath': image_name, 'query': query, 'generated_text': generated_text}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv('generated_data.csv', index=False)

@app.route('/query', methods=['POST'])
def process_image():
    data = request.json
    image_b64 = data.get('image')
    query = data.get('query', "Describe this image in detail from a dermatology point of view.")
    image_name = data.get('image_name', 'uploaded_image.png')

    # Determine if image is available
    is_image_available = bool(image_b64)
    print(data,is_image_available)

    # Step 1: Analyze image
    response = requests.post("http://localhost:5001/analyze_image", json={
        "image": image_b64,
        "user_text": query,
        "is_image_available": is_image_available
    })
    if response.status_code != 200:
        return jsonify({"error": "Image analysis failed"}), 500
    analysis_result = response.json().get("result", "")

    # Step 2: Extract entities
    extracted_entities = entity_chain.run(analysis_result)
    # Step 3: Process entities and build context
    if extracted_entities and extracted_entities.lower() != "none":
        entity_list = [entity.strip() for entity in extracted_entities.split(',')]
        print(entity_list)
        # Initialize context variable to store knowledge
        context = ""
            
        # Add entities to context
        context += "Extracted entities:\n"
        for entity in entity_list:
            context += f"- {entity}\n"
            
        # Use the entity list to query the knowledge graph
        is_knowledge_available = False
        for entity in entity_list:
            knowledge = graph.get_entity_knowledge(entity, depth=2)
            if knowledge:
                is_knowledge_available = True
                context += f"\nKnowledge about '{entity}':\n"
                context += str(knowledge) + "\n"
        if (is_knowledge_available == False):
            generated = requests.post(
                'http://localhost:5001/generate_rag',
                json={'entity_list': entity_list,
                        'user_text': query,}
            )
            if generated.status_code == 200:
                    generated_text_html = generated.json().get("rag_results").replace('\n', ' <br> ')
                    generated_text = str(generated_text_html)
        else:              
            generated_text = qa_chain.run({"context": context, "question": "", "user_query": query})
            # Step 5: Validate diagnosis
            try:
                validation_response = requests.post("http://localhost:5001/validate_diagnosis", json={
                    "text": generated_text
                })
                if validation_response.status_code == 200:
                    final_analysis = validation_response.json().get("result")
                else:
                    final_analysis = generated_text
            except:
                final_analysis = generated_text
    else:
        # If no entities were extracted, return the entity extraction result
        context = "No entities were extracted from the text."
        generated_text = "No specific dermatological condition could be identified from the image. Please consider providing a clearer image or consulting with a healthcare professional for an in-person assessment."
    

    final_analysis = generated_text
    # Save results
    save_to_csv(image_name, query, final_analysis)

    return jsonify({
        "initial_analysis": analysis_result,
        "extracted_entities": extracted_entities,
        "answer": final_analysis
    })

if __name__ == '__main__':
    app.run(port=5002)
