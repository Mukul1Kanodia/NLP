from sklearn.feature_extraction.text import CountVectorizer

# Our two documents
corpus = [
    'The cat sat on the mat.',
    'The dog ate the cat.'
]

# Create the Bag-of-Words model
vectorizer = CountVectorizer()

# Generate the word-count matrix
X = vectorizer.fit_transform(corpus)

# See the vocabulary (the features)
print("Vocabulary: ", vectorizer.get_feature_names_out())
# Output: Vocabulary:  ['ate' 'cat' 'dog' 'mat' 'on' 'sat' 'the']
# Note: sklearn lowercases and sorts them automatically

# See the resulting vectors
print("\nBoW Vectors:")
print(X.toarray())
# Output:
# [[0 1 0 1 1 1 2]   <-- Vector for Document 1
#  [1 1 1 0 0 0 2]]  <-- Vector for Document 2