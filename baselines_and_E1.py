#%%
import pandas as pd
print(pd.__version__)
print(pd.__file__)
BASE_PATH= r'C:\Users\samue\Downloads\INFORMATION RETRIEVAL PROJECT\indices'

##here we import all of our precious stuff
#%%
import os

##If we are running on our personal device, we need to initialize java.
os.environ['JAVA_HOME'] = r'C:\Program Files\Java\jdk-25'
os.environ['_JAVA_OPTIONS'] = '-Xmx14g'
print("JVM Memory Limit set via _JAVA_OPTIONS: 12GB")
import pyterrier as pt
#%%
import pandas as pd
dataframe_path=r'C:\Users\samue\Downloads\INFORMATION RETRIEVAL PROJECT\Scripts\NER_DOCUMENTS_CORRECTED.csv'
df=pd.read_csv(dataframe_path)
#%%
##Query and Qrels from the train set (we are experimenting like true scientists)
qrel_path=r'C:\Users\samue\Downloads\INFORMATION RETRIEVAL PROJECT\qrels.csv'
queries_path=r'C:\Users\samue\Downloads\INFORMATION RETRIEVAL PROJECT\queries.csv'
qrels= pd.read_csv(qrel_path)
queries= pd.read_csv(queries_path)
queries[['qid','query']]=queries[['query_id','question']]
#%%
##We need to pre-process our queries to avoid errors later,

def clean_query_text(q):
    if pd.isna(q): return ""
    # Replace any character that is NOT a letter, number, or space with a space
    q = re.sub(r'[^a-zA-Z0-9\s]', ' ', str(q))
    # Remove extra whitespace
    return " ".join(q.split())

# 2. Prepare Queries
# Apply cleaning
queries['query'] = queries['query'].apply(clean_query_text)

# ... previous imports and clean_query_text function ...

# 1. Apply cleaning
queries['query'] = queries['query'].apply(clean_query_text)
queries = queries[queries['query'].str.len() > 0]

# --- CRITICAL FIX STARTS HERE ---
# Force IDs to be strings in both dataframes , this is very important for pyterrier to read things correctly
queries['qid'] = queries['qid'].astype(str)
qrels['qid']   = qrels['qid'].astype(str)
qrels['docno'] = qrels['docno'].astype(str)

#%%
#BASELINE 1 N-GRAMS!
def to_ngrams(text, n=4):
    """
    Converts a string into a space-separated sequence of character n-grams.
    Example: "Tripoli", n=3 -> "Tri rip ipo pol oli"
    """
    if pd.isna(text) or text == "":
        return ""
    
    # Clean text: remove extra spaces, keep it simple
    text = str(text).strip()
    
    # Edge Case: If word is shorter than n (e.g. "To"), keep original
    if len(text) < n:
        return text
    
    # Generate n-grams
    # We slide a window of size 'n' across the text
    ngrams = [text[i : i+n] for i in range(len(text) - n + 1)]
    
    return " ".join(ngrams)

df['n-context'] =df['context'].fillna("").astype(str).apply(lambda x: to_ngrams(x, n=4))
df['n-context']= df['N-context'].apply(lambda x: " ".join(x) if isinstance(x, list) else x)
print(type(df['N-context'].iloc[0]))
#%%
fuzzy_query=queries.copy()
fuzzy_query['query']=fuzzy_query['query'].fillna("").astype(str).apply(lambda x: to_ngrams(x, n=4))
#%%
df['n-context'].head(1)

#%%

BASE_PATH = r"C:\Users\samue\Downloads\INFORMATION RETRIEVAL PROJECT\indices"
INDEX_FUZZY_PATH = os.path.join(BASE_PATH, "trial_fuzzy_index_5")

# 1) Delete the old (broken) index folder
if os.path.exists(INDEX_FUZZY_PATH):
    shutil.rmtree(INDEX_FUZZY_PATH)
df['docno']=df['docno'].astype(str)
# 2) Rebuild from scratch
metas = ["docno"]
text_fields_fuzzy = ['n-context']

docs = df[["docno", 'n-context']].to_dict("records")

indexer_fuzzy = pt.index.IterDictIndexer(
    INDEX_FUZZY_PATH,
    meta=metas,
    text_attrs=['n-context'],
    overwrite=True,          # important
)
indexref = indexer_fuzzy.index(docs)
index_fuzzy = pt.IndexFactory.of(indexref)

#%%
from pyterrier.measures import *
bm25_fuzzy= pt.BatchRetrieve(index_fuzzy, wmodel="BM25", metadata=["docno"])
bm25_fuzzy_results= bm25_fuzzy.transform(fuzzy_query[:100])
print(pt.Evaluate(bm25_fuzzy_results,qrels[:100],metrics=[R@1,R@10,"map"]))

#%%
BASE_PATH= r'C:\Users\samue\Downloads\INFORMATION RETRIEVAL PROJECT\indices'
df['docno'] = df['docno'].astype(str)
# 1. Define the columns you want to combine
cols = ["PERSON","ORG","GPE","EVENT","LOC","location","publication_date","NORP","MONEY","publication_date","location"]

# 2. Fill NaNs, convert to string, and join them across the row (axis=1)
df["entities"] = df[cols].fillna("").astype(str).agg(" ".join, axis=1)
INDEX_BASELINE_PATH = os.path.join(BASE_PATH, 'trial_baseline_index6') 
INDEX_TEST_PATH =os.path.join(BASE_PATH, 'trial_test_index10') 
INDEX_ENTITY_PATH =os.path.join(BASE_PATH, 'trial_entity_index') 
INDEX_ALL_PATH =os.path.join(BASE_PATH, 'trial_all_index') 

text_fields_base = [                            #here we specify which fields to index
    "context"
]
text_fields_correct = [                            #here we specify which fields to index
    "context_clean",
]
text_fields_entity = [                            #here we specify which fields to index
    "entities",
]
text_fields_all = [                            #here we specify which fields to index
    "context_corrected","entities","N-context"
]
#%%



metas=['docno']   #here we specify the meta data, fields that we need to carry on for later, but that will not influence the scoring
docs = df[[
    "docno",'context','context_clean','entities'
]].to_dict("records")
#%%
indexer_base = pt.index.IterDictIndexer(INDEX_BASELINE_PATH, meta=metas, text_attrs=text_fields_base)
indexref = indexer_base.index(docs)
index_baseline = pt.IndexFactory.of(indexref)

#%%
indexer_test = pt.index.IterDictIndexer(INDEX_TEST_PATH, meta=metas, text_attrs=text_fields_correct)
indexref = indexer_test.index(docs)
index_test = pt.IndexFactory.of(indexref)
#%%
indexer_entity = pt.index.IterDictIndexer(INDEX_ENTITY_PATH, meta=metas, text_attrs=text_fields_entity)
indexref = indexer_entity.index(docs)
index_entity = pt.IndexFactory.of(indexref)
#%%
indexer_all = pt.index.IterDictIndexer(INDEX_ALL_PATH, meta=metas, text_attrs=text_fields_all)
indexref = indexer_all.index(docs)
index_all = pt.IndexFactory.of(indexref)
# %%
import pandas as pd
import re
import pyterrier as pt
from pyterrier.measures import *

#%%
# 1. Define the Cleaner
def clean_query_text(q):
    if pd.isna(q): return ""
    # Replace any character that is NOT a letter, number, or space with a space
    q = re.sub(r'[^a-zA-Z0-9\s]', ' ', str(q))
    # Remove extra whitespace
    return " ".join(q.split())

# 2. Prepare Queries
# Apply cleaning
queries['query'] = queries['query'].apply(clean_query_text)

# ... previous imports and clean_query_text function ...

# 1. Apply cleaning
queries['query'] = queries['query'].apply(clean_query_text)
queries = queries[queries['query'].str.len() > 0]

# --- CRITICAL FIX STARTS HERE ---
# Force IDs to be strings in both dataframes
queries['qid'] = queries['qid'].astype(str)
qrels['qid']   = qrels['qid'].astype(str)
qrels['docno'] = qrels['docno'].astype(str)
# --------------------------------
#%%
# 2. Select exactly the first 1000 queries
selected_queries = queries[['qid', 'query']].head(1000)
selected_fuzzy_queries= selected_queries.copy()

#%%
# 3. Define Retrievers
bm25_default = pt.BatchRetrieve(index_baseline, wmodel="BM25", metadata=["docno"])
bm25_test    = pt.BatchRetrieve(index_test, wmodel="BM25", metadata=["docno"])
bm25_fuzzy    = pt.BatchRetrieve(index_fuzzy, wmodel="BM25", metadata=["docno"])
print(f"Running Experiment on {len(selected_queries)} queries...")
#%%
base_results=bm25_default.transform(selected_queries)
bm25_default_=pt.Transformer.from_df(base_results,uniform=True)
#%%
correct_results=bm25_test.transform(selected_queries)
bm25_test_=pt.Transformer.from_df(correct_results,uniform=True)
#%%
fuzzy_results=bm25_fuzzy.transform(selected_fuzzy_queries)
bm25_fuzzy_=pt.Transformer.from_df(fuzzy_results,uniform=True)
#%%
pl2_entity= pt.BatchRetrieve(index_entity,wmodel="PL2",metadata=["docno"])
entity_results=pl2_entity.transform(selected_queries)
pl2_entity_=pt.Transformer.from_df(entity_results,uniform=True)
#%%
bm25_all=pt.BatchRetrieve(index_all,wmodel="BM25",metadata=["docno"])
all_results= bm25_all.transform(queries[['qid','query']][:10000])
bm25_all_= pt.Transformer.from_df(all_results,uniform=True)
#%%

# 4. Run Experiment
final_results = pt.Experiment(
    [bm25_all_,magic],
    queries[['qid','query']][:10000],
    qrels, 
    eval_metrics=[R@1, MAP, R@10, R@100],
    baseline=0,
    names=["all","magic"]
)

print(final_results)
# %%
result_normal=bm25_all.transform(queries[['qid','query']][:10000])
#%%
result_edits=bm25_all.transform(gemini_queries[:10000])
#%%

magic_= 1.1*pt.Transformer.from_df(result_normal) + pt.Transformer.from_df(result_edits)
magic= magic_.transform(queries[['qid','query']][:10000])
# %%
print(pt.Evaluate(result_normal,qrels[:10000],metrics=[R@1,R@10,'map']))
print(pt.Evaluate(result_edits,qrels[:10000],metrics=[R@1,R@10,'map']))
#%%
print(pt.Evaluate(magic,qrels[:10000],metrics=[R@1,R@10,'map',R@100]))
# %%
queries[['qid','query']][:10000].to_csv('queries_to_weight_10000.csv', index=False)
# %%
gemini_queries=pd.read_csv(r'C:\Users\samue\Downloads\INFORMATION RETRIEVAL PROJECT\weighted_queries_phrase_boosted_10000.csv')
# %%
