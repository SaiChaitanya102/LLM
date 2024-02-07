from transformers import pipeline # importing pipeline from transformers library. The pipeline contains models of few NLP tasks

def bert_sentiment_analysis(text): # takes input as text and returns label and score
    sentiment_classifier = pipeline("sentiment-analysis") #Using Sentiment-analysis model from pipeline
    result = sentiment_classifier(text)[0] # to get highest accurated sentiment
    return result['label'], result['score']

text_to_analyze = "I absolutely loved the movie! It was fantastic."

sentiment_label, confidence_score = bert_sentiment_analysis(text_to_analyze)
print("Sentiment Label:", sentiment_label)
print("Confidence Score:", confidence_score)
