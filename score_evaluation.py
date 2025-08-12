import pandas as pd
from bert_score import score
import spacy

nlp = spacy.load("en_core_sci_sm")

df = pd.read_csv('response_evaluate.csv') 
generated_texts = df['graph_answer'].tolist()
reference_texts = df['response'].tolist()

def extract_entities(text):
    doc = nlp(text)
    return " ".join(ent.text for ent in doc.ents)

df['Generated_Entities'] = [extract_entities(text) for text in generated_texts]
df['Reference_Entities'] = [extract_entities(text) for text in reference_texts]

P, R, F1 = score(df['Generated_Entities'].tolist(), df['Reference_Entities'].tolist(), lang="en", rescale_with_baseline=False)

df['Precision'] = P.tolist()
df['Recall'] = R.tolist()
df['F1'] = F1.tolist()

print(df[['graph_answer', 'response', 'Generated_Entities', 'Reference_Entities', 'Precision', 'Recall', 'F1']])

average_precision = P.mean().item()
average_recall = R.mean().item()
average_f1 = F1.mean().item()

print("Average Precision:", average_precision)
print("Average Recall:", average_recall)
print("Average F1 Score:", average_f1)

df.to_csv('response_bert_scores_total.csv', index=False)
