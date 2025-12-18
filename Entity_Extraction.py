#%%

# 1. SET JAVA_HOME (Necessary for the kernel to find Java)
os.environ['JAVA_HOME'] = r'C:\Program Files\Java\jdk-25'

#give it more ram
os.environ['_JAVA_OPTIONS'] = '-Xmx14g'
print("JVM Memory Limit set via _JAVA_OPTIONS: 12GB")


#%%
import pandas as pd

#Upload all documents

train_set = pd.read_csv(
    r'C:\Users\samue\Downloads\INFORMATION RETRIEVAL PROJECT\CAQA_Train.csv',
    sep=';', #use ; as separator
    quotechar='"'  # use the standard double quote as the quote character
)
test_set = pd.read_csv(
    r'C:\Users\samue\Downloads\INFORMATION RETRIEVAL PROJECT\Test.csv',
    sep=';',
    quotechar='"' 
)

dev_set = pd.read_csv(
    r'C:\Users\samue\Downloads\INFORMATION RETRIEVAL PROJECT\CAQA_Dev.csv',
    sep=';',
    quotechar='"'  
)

df = pd.concat([train_set, test_set, dev_set], ignore_index=True)

# sanity check
print(f"Train shape: {train_set.shape}")
print(f"Test shape:  {test_set.shape}")
print(f"Dev shape:   {dev_set.shape}")
print(f"Combined shape: {df.shape}")
# %%
# extracting location

df["location"] = df["para_id"].str.extract(r'([a-zA-Z_]+)')
df["location"] = df["location"].str.replace("_", " ")
df["location"] = df["location"].str[:-1]
# I want all lowercase
df["location"] = df["location"].str.lower()


documents=df[["context","raw_ocr","publication_date","location","para_id"]]

documents['docno'] = documents["para_id"]
# %%
import pandas as pd
import re
import unicodedata

# CLEANING FUNCTION
def clean_historical_text(text):
    """
    Cleans text to fit standard American Newspaper format (ASCII only).
    1. Normalizes accents (à -> a, ç -> c).
    2. Removes symbols like @, %, £, €, and OCR noise.
    3. Collapses multiple spaces.
    """
    if not isinstance(text, str):
        return ""

    # we normalize the text (removing accents)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')

    # allowed characters
    text = re.sub(r'[^a-zA-Z0-9\s.,;:\'"\-?!]', '', text)

    # we collapse trailing whitespaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

documents['context_clean'] = documents['context'].astype(str).apply(clean_historical_text)



#%%

import spacy
from spacy.pipeline import EntityRuler
import pandas as pd
import os
from tqdm import tqdm
import multiprocessing

# we use the spacy large model to extract the entities
MODEL_NAME = "en_core_web_lg"

#we run this process in parallel to save time
N_PROCESS = 2 
BATCH_SIZE = 1000  # set batch size

# labels that we want to extract, considered relevant for the task
TARGET_LABELS = [
    "PERSON", "ORG", "DATE", "GPE", "FAC", 
    "CARDINAL", "LOC", "NORP", "PRODUCT", 
    "EVENT", "MISC", "LAW", "WORK_OF_ART","MONEY" 
]

# loading the model
print(f"Loading {MODEL_NAME}...")

try:
    nlp = spacy.load(MODEL_NAME, disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])
except OSError:
    # download model if not present
    os.system(f"python -m spacy download {MODEL_NAME}")
    nlp = spacy.load(MODEL_NAME, disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])

#spacy do not have any title/jobs type of entity, which may be useful for the task, here we try to extract those professions with a rule based approach
ruler = nlp.add_pipe("entity_ruler", before="ner")

title_patterns = [
    # the usual pattern is usually title + Naame , it can be in the short or long version
    {"label": "PERSON", "pattern": [
        {"TEXT": {"IN": ["Gen.", "Col.", "Maj.", "Capt.", "Hon.", "Gov.", "Pres.", "Sen.", "Rep.", "Dr.", "Rev.", "Prof.", "Mr.", "Mrs."]}}, 
        {"IS_TITLE": True} # the name following must be capitalized
    ]},
    # full word case
    {"label": "PERSON", "pattern": [
        {"TEXT": {"IN": ["General", "Colonel", "Major", "Captain", "Governor", "President", "Senator", "Representative", "Doctor", "Reverend", "Professor"]}}, 
        {"IS_TITLE": True}
    ]}
]
ruler.add_patterns(title_patterns)


# EXTRACT ENTITIES
def process_ner_optimized(df, text_col='context'):
    
    # fill the Nans if present (not present)
    data = df[text_col].fillna("").astype(str).tolist()
    
    final_results = {label: [] for label in TARGET_LABELS}
    
    print(f"starting NER on {len(data)} docs using {N_PROCESS} cores...")
    
    # start extracting
    doc_stream = nlp.pipe(data, n_process=N_PROCESS, batch_size=BATCH_SIZE)
    
    # progress bar
    for doc in tqdm(doc_stream, total=len(data), desc="Extracting Entities"):
        
        # temporary dict for current document
        doc_ents = {label: set() for label in TARGET_LABELS}
        
        for ent in doc.ents:
            if ent.label_ in TARGET_LABELS:
                doc_ents[ent.label_].add(ent.text.strip())
        
        # joint unique entities
        for label in TARGET_LABELS:
            final_results[label].append(", ".join(sorted(doc_ents[label])))

    # dataframe
    for label in TARGET_LABELS:
        df[label] = final_results[label]
        
    return df

#%%
if 'documents' in locals():
    
    documents = process_ner_optimized(documents)
    
    # Verify Titles are captured (Look at PERSON column)
    print("\n--- Extraction Complete. Check 'PERSON' column for titles ---")
    print(documents[['context', 'PERSON', 'ORG', 'DATE']].head().to_markdown(index=False))
else:
    print("Error: 'documents' DataFrame not found.")


# %%
documents.to_csv('final_documents_full_ner.csv')
# %%
documents.head()
# %%
import pandas as pd
documents=pd.read_csv(r'C:\Users\samue\Downloads\INFORMATION RETRIEVAL PROJECT\Scripts\final_documents_full_ner.csv')
documents.head()
# %%
documents.shape
documents.to_csv(r'C:\Users\samue\Downloads\INFORMATION RETRIEVAL PROJECT\Scripts\NER_DOCUMENTS.csv')
# %%
