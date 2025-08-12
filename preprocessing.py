import json
import spacy
import pandas as pd
from scispacy.umls_linking import UmlsEntityLinker
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from autocorrect import Speller

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Load SciSpacy model  
nlp = spacy.load("en_core_sci_md")
nlp.add_pipe("scispacy_linker", last=True)

# File paths  
train_query_preprocessed_json_path = "train_query_preprocessed.json"
train_answers_json_path = "train_answersonly.json"
output_csv_path = "preprocessed_responses_with_scispacy.csv"

# Initialize tools  
lemmatizer = WordNetLemmatizer()
spell = Speller(lang='en')
stop_words = set(stopwords.words('english'))

# Relevant semantic types  
RELEVANT_SEMANTIC_TYPES = {'T047', 'T191', 'T033', 'T184'}

# Preprocess and enhance text  
def preprocess_and_enhance_text(text):
    doc = nlp(text)
    enhanced_text = text
    ner_scispacy = []

    for ent in doc.ents:
        if ent._.kb_ents:
            for umls_ent_id, _ in ent._.kb_ents:
                concept = nlp.get_pipe("scispacy_linker").kb.cui_to_entity[umls_ent_id]
                if any(sem_type in RELEVANT_SEMANTIC_TYPES for sem_type in concept.types):
                    scientific_name = concept.canonical_name
                    if ent.text.lower() != scientific_name.lower() or ent.text.lower() not in scientific_name.lower():
                        enhanced_text = enhanced_text.replace(ent.text, f"{ent.text} ({scientific_name})")
                    ner_scispacy.append((ent.text, "ENTITY"))
                    break  
    return enhanced_text.casefold(), ner_scispacy

# Load preprocessed queries  
with open(train_query_preprocessed_json_path, 'r', encoding='utf-8') as f:
    query_data = json.load(f)

# Extract relevant encounter_ids  
relevant_encounter_ids = {entry['encounter_id'] for entry in query_data}

# Load original responses  
with open(train_answers_json_path, 'r', encoding='utf-8') as f:
    answers_data = json.load(f)

# Process and save responses  
processed_entries = []

for entry in answers_data:
    encounter_id = entry['encounter_id']
    
    if encounter_id in relevant_encounter_ids:
        image_ids = entry.get('image_ids', [])
        responses = entry['responses']
        
        for response in responses:
            original_text = response['content_en']
            case_folded_text, ner_data = preprocess_and_enhance_text(original_text)

            # Remove special characters  
            no_special_chars = ''.join(e for e in case_folded_text if e.isalnum() or e.isspace())

            # Tokenization  
            tokens = word_tokenize(no_special_chars)

            # Remove stopwords  
            no_stopwords = [word for word in tokens if word.lower() not in stop_words]

            # Lemmatization  
            lemmatized = [lemmatizer.lemmatize(word) for word in no_stopwords]
            lemmatized_str = ' '.join(lemmatized)

            processed_entries.append({
                'id': encounter_id,
                'image_ids': ', '.join(image_ids),
                'response_text': original_text,
                'case_folding': case_folded_text,
                'no_special_chars': no_special_chars,
                'tokenized': str(tokens),
                'no_stopwords': str(no_stopwords),
                'lemmatized': str(lemmatized),
                'lemmatized_str': lemmatized_str,
                'ner_scispacy': str(ner_data)
            })

# Save to CSV  
df = pd.DataFrame(processed_entries)
df.to_csv(output_csv_path, index=False, encoding='utf-8')

print(f"Preprocessing complete. Saved to {output_csv_path}.")
