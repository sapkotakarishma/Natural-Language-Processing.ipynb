# Natural-Language-Processing.ipynb
1. What is NLP?
     Natural Language Processing (NLP) is the technology that allows computers to "read" and "understand" human language. Since computers only understand numbers ($0$s and $1$s), NLP acts as a translator that turns messy human speech and text into structured data.

2. Common Uses
     Sentiment Analysis: Is a review positive or negative?
     Virtual Assistants: How Siri or Alexa understand your voice.
     Auto-Correct: Predicting what you meant to type.
     Machine Translation: Turning English into Spanish or French instantly.

3. The 5 Steps of an NLP Pipeline

   Step 1: Cleaning (Noise Reduction)
        Removing the "trash" from text like HTML tags, special characters (%, $, #), and extra spaces.
        
        Why: It prevents the computer from getting confused by irrelevant symbols.
   
   Step 2: Tokenization
       Splitting a long sentence into individual words called "tokens."
       
       Why: A computer processes words one by one, not as a whole paragraph.
   
   Step 3: Stopword Removal
       Filtering out common words like "the," "is," "at," and "which."
       
       Why: These words appear everywhere but don't carry unique meaning. Removing them saves processing power.
   
   Step 4: Lemmatization
       Changing words back to their "root" form (e.g., "running" becomes "run," "better" becomes "good").
       
       Why: It groups similar meanings together so the model recognizes they are the same concept.
   
   Step 5: Vectorization (The Math Step)
       Converting the cleaned words into numbers (Vectors).
       
       Why: Algorithms cannot "read" strings; they need a numerical matrix to perform calculations.


##### Example Code########
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize tools
     nltk.download('stopwords')
     
     nltk.download('wordnet')
     
     stop_words = set(stopwords.words('english'))
     
     lemmatizer = WordNetLemmatizer()

def nlp_pipeline(text):
    # 1. Cleaning: Keep only letters
    
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    
    # 2. Tokenization & 3. Stopword Removal
    
    tokens = [word for word in text.split() if word not in stop_words]
    
    # 4. Lemmatization: Bringing words to their root
    
    normalized = [lemmatizer.lemmatize(w) for w in tokens]
    
    return " ".join(normalized)

# --- Example Usage ---

          raw_data = ["I am loving this NLP project!", "The algorithms are running smoothly."]
          
          cleaned_data = [nlp_pipeline(doc) for doc in raw_data]

# 5. Vectorization: Turning text into a math matrix
          
          vectorizer = TfidfVectorizer()
          
          matrix = vectorizer.fit_transform(cleaned_data)
          
          print(f"Original: {raw_data[0]}")
          
          print(f"Processed: {cleaned_data[0]}")
          
          print(f"Matrix Shape: {matrix.shape}")
