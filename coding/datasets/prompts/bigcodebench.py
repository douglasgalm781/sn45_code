DATA_SYNTH_PROMPT = """
Based on the following simple example, write more complex scenarios and invoke multiple Python libraries 
to solve each problem.
The written intent should align with a more specific and practical scenario, but should still be easy to 
do functional correctness assertion.
For each scenario, write a single Python function with the rewritten intent.
Please include requirements and terminal-based input-output examples in the function docstring.
The function should contain complex logic like if-else statements and loops.
You have to use more than three Python libraries for a scenario. Write imports and variable definitions 
outside the function.
Try to avoid using web APIs if possible.
If there are any constants (e.g. strings and numeric values) used in the functions, you need to declare 
them before the function.
If data is used, you need to provide sample data in the comment.
Try to return values for correctness assertion.
Each programming scenario and intent should be separated by a newline.
Generate two examples with two scenarios from the following simple example:
```python
def count_char(char, word):
    \"\"\"Counts the characters in word\"\"\"
    return word.count(char) # If you want to do it manually try a for loop
```


Scenario 1:
```python
import re
from collections import Counter
from itertools import chain
import pandas as pd
import numpy as np
import random
import string

# Constants
COMMON_WORDS = ["the", "be", "to", "of", "and", "a", "in", "that", "have", "I"]
THRESHOLD_FREQUENCY = 5


def analyze_text_corpus(corpus):
    \"\"\"
    Analyzes a list of text documents for word frequency, rare words, and token length statistics.
    
    Parameters:
        - corpus (List[str]): A list of text documents, where each document is a single string.
        
    Requirements:
        - re
        - collections
        - itertools
        - pandas
        - numpy
        - random
        - string

    Example:
    >>> corpus = [
    ...     "The quick brown fox jumps over the lazy dog.",
    ...     "To be or not to be, that is the question.",
    ...     "A journey of a thousand miles begins with a single step."
    ... ]
    >>> result = analyze_text_corpus(corpus)
    >>> print(result)
    {
        'most_common_words': [('the', 3), ('be', 2)],
        'rare_words': ['journey', 'thousand', 'begins'],
        'token_length_stats': {
            'mean': 4.0,
            'std_dev': 1.58,
            'median': 4
        }
    }
    
    Returns:
        dict: A dictionary containing the most common words, rare words, and token length statistics.
    \"\"\"
    
    # Tokenize and filter common words
    all_tokens = [re.findall(r'\b\w+\b', doc.lower()) for doc in corpus]
    flattened_tokens = list(chain.from_iterable(all_tokens))
    filtered_tokens = [word for word in flattened_tokens if word not in COMMON_WORDS]

    # Word frequency analysis
    word_counts = Counter(filtered_tokens)
    most_common_words = word_counts.most_common(5)
    rare_words = [word for word, count in word_counts.items() if count < THRESHOLD_FREQUENCY]
    
    # Token length analysis
    token_lengths = [len(token) for token in flattened_tokens]
    token_length_series = pd.Series(token_lengths)
    token_length_stats = {
        'mean': np.round(token_length_series.mean(), 2),
        'std_dev': np.round(token_length_series.std(), 2),
        'median': int(token_length_series.median())
    }
    
    return {
        'most_common_words': most_common_words,
        'rare_words': rare_words,
        'token_length_stats': token_length_stats
    }
```
Scenario 2:
```python
import re
from collections import Counter
from itertools import chain
import pandas as pd
import numpy as np
import random
import string
# Sample dataset for product data analysis
# Commented data format for input to function
# products = [
#     {"name": "Laptop", "price": 899.99, "category": "Electronics"},
#     {"name": "Book", "price": 14.99, "category": "Education"},
#     {"name": "Smartphone", "price": 699.99, "category": "Electronics"},
#     {"name": "Pen", "price": 1.99, "category": "Stationery"},
#     {"name": "Notebook", "price": 2.99, "category": "Stationery"},
#     {"name": "Headphones", "price": 199.99, "category": "Electronics"},
# ]

def product_category_statistics(products):
    \"\"\"
    Processes product information to analyze average prices, identify top categories,
    and group products by category based on price ranges.
    
    Parameters:
        - products (List[dict]): A list of dictionaries with keys 'name', 'price', and 'category'
        
    Requirements:
        - collections
        - pandas
        - numpy
        - random
        - string

    Example:
    >>> products = [
    ...     {"name": "Laptop", "price": 899.99, "category": "Electronics"},
    ...     {"name": "Book", "price": 14.99, "category": "Education"},
    ...     {"name": "Smartphone", "price": 699.99, "category": "Electronics"},
    ...     {"name": "Pen", "price": 1.99, "category": "Stationery"},
    ...     {"name": "Notebook", "price": 2.99, "category": "Stationery"},
    ...     {"name": "Headphones", "price": 199.99, "category": "Electronics"},
    ... ]
    >>> result = product_category_statistics(products)
    >>> print(result)
    {
        'average_price_by_category': {'Electronics': 599.99, 'Education': 14.99, 'Stationery': 2.49},
        'top_category': 'Electronics',
        'products_in_price_ranges': {
            'low': ['Pen', 'Notebook'],
            'mid': ['Book', 'Headphones'],
            'high': ['Smartphone', 'Laptop']
        }
    }
    
    Returns:
        dict: A dictionary containing average prices, the top category, and products grouped by price ranges.
    \"\"\"

    # DataFrame creation
    df = pd.DataFrame(products)

    # Average price by category
    avg_price_by_category = df.groupby("category")["price"].mean().round(2).to_dict()
    
    # Top category by product count
    category_counts = Counter(df['category'])
    top_category = category_counts.most_common(1)[0][0]

    # Price range grouping
    price_ranges = {'low': [], 'mid': [], 'high': []}
    for _, row in df.iterrows():
        if row["price"] < 10:
            price_ranges['low'].append(row["name"])
        elif 10 <= row["price"] < 100:
            price_ranges['mid'].append(row["name"])
        else:
            price_ranges['high'].append(row["name"])

    return {
        'average_price_by_category': avg_price_by_category,
        'top_category': top_category,
        'products_in_price_ranges': price_ranges
    }
```

Above is the illustration.

Generate five complex scenarios based on the following simple example:
"""
