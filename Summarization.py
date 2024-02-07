from summarizer import Summarizer #This line imports the Summarizer class from the summarizer library. The Summarizer class provides an interface to perform text summarization using pre-trained models, particularly BERT-based models

def bert_summarize(text): #This function takes input as text and returns summary
    model = Summarizer() #loads the model
    summary = model(text) #invokes the model to generate a summary of the input text
    return summary

text_to_summarize = """
    Natural Language Processing (NLP) is the technology used to help machines to understand and learn text and language. With NLP data scientists aim to teach machines to understand what is said and written to make sense of the human language. It is used to apply machine learning algorithms to text and speech.
"""

summary = bert_summarize(text_to_summarize)
print(summary)
