# Exploring NLP capabilities as part of Bootcamp Study

import spacy
nlp = spacy.load('en_core_web_sm')

# Tokenised text
gardenpathSentences = [ "The man who whistles tunes pianos." \
                    "The complex houses married and single soldiers and their families.", \
                    "The old man the boat.", "Mary gave the child a Band-Aid", \
                    "That Jill is never here hurts", \
                    "The cotton clothing is made of grows in Mississippi" \
                    "Terry the dog wags it's tail", "The Mississipi is a river"]

gardenpath_sentences = ", ".join(gardenpathSentences)
doc = nlp(gardenpath_sentences)

for token in doc:
    print(token.text)

# Named Entity Recognition
for ent in doc.ents:
    print(f"{ent.text:{19}} {ent.label_}")

# Sentence categorisation
for token in doc:
    print(f"{token.text:{12}} {token.pos_:{6}} {token.dep_}")

print(spacy.explain("PERSON"))
print(spacy.explain("GPE"))

# PERSON - People, including fictional. It makes sense because it recognises people's nouns, 
# but gets confused with Terry (Non-garden path sentence, for testing purposes)
# GPE - countries, cities, states. The NER can't identify ambiguity beteween place and river.