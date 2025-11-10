# Vibe Matcher - Nexora AI Internship Assignment

## Overview
A professional fashion recommendation system that matches products based on vibe queries using semantic similarity and vector embeddings. Built for the Nexora AI Internship technical assessment.

## Features
- **Vibe-based Matching**: Understands fashion style preferences beyond keywords
- **Vector Similarity Search**: Uses cosine similarity for accurate recommendations
- **Robust Architecture**: Multiple fallback strategies for reliability
- **Professional Analytics**: Comprehensive performance evaluation and visualization
- **Production Ready**: Error handling, edge case management, and scalable design

## Technical Implementation
- **Data**: 8 fashion products with detailed descriptions and vibe tags
- **Embeddings**: TF-IDF + semantic enhancement with 384-dimensional vectors
- **Similarity**: Cosine similarity using scikit-learn
- **Evaluation**: 3 test queries with latency and accuracy metrics
- **Visualization**: Performance charts and similarity score analysis

## Assignment Requirements Met
✅ **Data Preparation**: 8 mock products with descriptions and vibe tags  
✅ **Embeddings**: Multiple embedding strategies with fallbacks  
✅ **Vector Search**: Cosine similarity with top-3 ranking  
✅ **Edge Cases**: Handles no matches and API failures  
✅ **Testing**: 3 queries with performance metrics  
✅ **Latency Plot**: Processing time visualization  
✅ **Reflection**: 5+ improvements and edge cases documented  

## Installation & Usage
```python
# Run in Google Colab
!git clone https://github.com/jaiswal-sarthak/vibe-matcher.git
%cd vibe-matcher

# Execute the main system
python main.py