# Vibe Matcher Pro - Advanced Fashion Recommendation System
# FIXED VERSION with Hugging Face API Integration & Diverse Results

import pandas as pd
import numpy as np
import requests
import time
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import timeit
import json
import re
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Professional styling
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("🚀 VIBE MATCHER PRO - Advanced AI Fashion Recommendation")
print("🔧 Hugging Face API Integration with Smart Fallbacks")

# =====================
# 1. ENHANCED DATA PREPARATION
# =====================

def create_enhanced_sample_data():
    """Create comprehensive fashion product data with rich metadata"""
    products = [
        {
            "id": "F001",
            "name": "Boho Festival Maxi Dress",
            "desc": "Flowy bohemian maxi dress with earthy tones, intricate floral patterns, and lightweight breathable fabric. Perfect for music festivals, summer outings, and beach vacations. Features elastic waist and side pockets.",
            "category": "Dresses",
            "price": 89.99,
            "vibes": ["boho", "festival", "flowy", "earthy", "bohemian", "summer"],
            "season": ["Spring", "Summer"],
            "occasion": ["Festival", "Beach", "Casual"]
        },
        {
            "id": "F002", 
            "name": "Urban Graphic Streetwear Hoodie",
            "desc": "Oversized urban hoodie with bold graphic prints, modern streetwear aesthetic, premium cotton blend. Perfect for city street style, casual outings, and layering. Features kangaroo pocket and adjustable drawstring.",
            "category": "Tops",
            "price": 65.50,
            "vibes": ["urban", "streetwear", "edgy", "comfortable", "modern", "trendy"],
            "season": ["Fall", "Winter"],
            "occasion": ["Casual", "Street", "Everyday"]
        },
        {
            "id": "F003",
            "name": "Classic Tailored Business Blazer",
            "desc": "Sophisticated navy blazer with sharp tailored lines, professional finish, and premium wool blend. Ideal for office wear, business meetings, and formal occasions. Features notch lapel and functional buttons.",
            "category": "Outerwear", 
            "price": 129.99,
            "vibes": ["professional", "elegant", "structured", "formal", "sophisticated"],
            "season": ["All Season"],
            "occasion": ["Office", "Formal", "Business"]
        },
        {
            "id": "F004",
            "name": "Cozy Cashmere Knit Sweater",
            "desc": "Luxurious chunky knit sweater in premium cashmere blend, exceptionally warm and soft. Perfect for cozy winter days, casual outings, and holiday gatherings. Features ribbed cuffs and crew neck.",
            "category": "Tops",
            "price": 149.99,
            "vibes": ["cozy", "comfortable", "warm", "casual", "luxurious", "soft"],
            "season": ["Fall", "Winter"],
            "occasion": ["Casual", "Holiday", "Everyday"]
        },
        {
            "id": "F005",
            "name": "Performance Athleisure Jogger Set",
            "desc": "Matching athletic set with advanced moisture-wicking fabric, sporty yet stylish design. Perfect for gym sessions, yoga, or casual streetwear. Features elastic waistband and breathable material.",
            "category": "Activewear",
            "price": 79.99,
            "vibes": ["athletic", "sporty", "comfortable", "modern", "active", "performance"],
            "season": ["All Season"],
            "occasion": ["Gym", "Casual", "Sports"]
        },
        {
            "id": "F006",
            "name": "Vintage Distressed Denim Jacket",
            "desc": "Classic denim jacket with authentic distressed finish and vintage wash. Versatile layering piece for retro-inspired outfits and casual looks. Features button-front and multiple pockets.",
            "category": "Outerwear",
            "price": 95.00,
            "vibes": ["vintage", "retro", "casual", "edgy", "classic", "timeless"],
            "season": ["Spring", "Fall"],
            "occasion": ["Casual", "Vintage", "Everyday"]
        },
        {
            "id": "F007",
            "name": "Elegant Silk Cocktail Dress",
            "desc": "Sophisticated silk cocktail dress with elegant cut, draped silhouette, and luxurious finish. Perfect for evening events, weddings, and special occasions. Features back zip and lining.",
            "category": "Dresses",
            "price": 199.99,
            "vibes": ["elegant", "sophisticated", "formal", "glamorous", "chic", "luxurious"],
            "season": ["All Season"],
            "occasion": ["Formal", "Evening", "Wedding"]
        },
        {
            "id": "F008",
            "name": "Minimalist Linen Button-Down Shirt",
            "desc": "Clean linen button-down shirt with minimalist design and breathable natural fabric. Ideal for smart casual occasions, office wear, and summer outings. Features button-down collar.",
            "category": "Tops",
            "price": 75.00,
            "vibes": ["minimalist", "clean", "casual", "sophisticated", "simple", "refined"],
            "season": ["Spring", "Summer"],
            "occasion": ["Casual", "Office", "Smart Casual"]
        },
        {
            "id": "F009",
            "name": "Sporty Running Shoes",
            "desc": "High-performance running shoes with advanced cushioning and breathable mesh. Perfect for athletic activities, gym workouts, and casual wear. Features rubber outsole and arch support.",
            "category": "Footwear",
            "price": 120.00,
            "vibes": ["athletic", "sporty", "performance", "comfortable", "active"],
            "season": ["All Season"],
            "occasion": ["Gym", "Sports", "Casual"]
        },
        {
            "id": "F010",
            "name": "Designer Leather Handbag",
            "desc": "Luxurious leather handbag with elegant design, multiple compartments, and gold-tone hardware. Perfect for formal events, office wear, and special occasions. Features adjustable strap.",
            "category": "Accessories",
            "price": 249.99,
            "vibes": ["elegant", "luxurious", "sophisticated", "formal", "chic"],
            "season": ["All Season"],
            "occasion": ["Formal", "Office", "Evening"]
        }
    ]
    
    return pd.DataFrame(products)

# Create enhanced dataset
df_products = create_enhanced_sample_data()
print("📦 ENHANCED PRODUCT CATALOG:")
print(f"Total Products: {len(df_products)}")
print(f"Categories: {df_products['category'].unique().tolist()}")
print(f"Price Range: ${df_products['price'].min()} - ${df_products['price'].max()}")
print("\n" + "="*80)

# =====================
# 2. HUGGING FACE API SERVICE WITH PROPER INTEGRATION
# =====================

class HuggingFaceEmbeddingService:
    """Professional Hugging Face API service with robust error handling"""
    
    def __init__(self):
        self.embedding_dim = 384
        self.current_provider = "HuggingFace"
        self.api_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
        self.headers = {"Authorization": "Bearer api_token"}
        self.fallback_used = False
        
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts using Hugging Face API"""
        print("🔮 Generating embeddings with Hugging Face API...")
        
        embeddings = []
        successful_embeddings = 0
        
        for i, text in enumerate(texts):
            print(f"   📥 Processing text {i+1}/{len(texts)}...")
            
            # Try Hugging Face API first
            api_embedding = self._get_single_embedding_api(text)
            
            if api_embedding is not None:
                embeddings.append(api_embedding)
                successful_embeddings += 1
                print(f"   ✅ Success with Hugging Face API")
            else:
                # Fallback to local embedding
                print(f"   ⚠️ API failed, using local fallback")
                local_embedding = self._get_local_embedding(text)
                embeddings.append(local_embedding)
                self.fallback_used = True
            
            # Rate limiting
            time.sleep(0.5)
        
        # Update provider info
        if self.fallback_used:
            self.current_provider = "Mixed (HuggingFace + Local)"
            print(f"🔧 Used mixed providers: {successful_embeddings} API, {len(texts)-successful_embeddings} local")
        else:
            print(f"🎉 All embeddings successfully generated with Hugging Face API!")
        
        return embeddings
    
    def _get_single_embedding_api(self, text: str, max_retries: int = 3) -> Optional[List[float]]:
        """Get single embedding using Hugging Face API with retries"""
        for attempt in range(max_retries):
            try:
                payload = {
                    "inputs": text,
                    "options": {"wait_for_model": True},
                    "parameters": {"truncation": True}
                }
                
                response = requests.post(
                    self.api_url, 
                    headers=self.headers, 
                    json=payload, 
                    timeout=30
                )
                
                if response.status_code == 200:
                    embedding = response.json()
                    if isinstance(embedding, list) and len(embedding) == self.embedding_dim:
                        return embedding
                    else:
                        print(f"      ⚠️ Invalid embedding format on attempt {attempt + 1}")
                
                elif response.status_code == 503:
                    wait_time = (attempt + 1) * 10
                    print(f"      ⏳ Model loading, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                    
                else:
                    print(f"      🔴 API Error {response.status_code} on attempt {attempt + 1}")
                    
            except requests.exceptions.Timeout:
                print(f"      ⏰ Timeout on attempt {attempt + 1}")
            except requests.exceptions.ConnectionError:
                print(f"      🔌 Connection error on attempt {attempt + 1}")
            except Exception as e:
                print(f"      ❌ Unexpected error on attempt {attempt + 1}: {str(e)[:100]}")
            
            if attempt < max_retries - 1:
                time.sleep(2)
        
        return None
    
    def _get_local_embedding(self, text: str) -> List[float]:
        """High-quality local fallback embedding"""
        # Create semantic-rich local embedding
        words = text.lower().split()
        
        # Enhanced feature extraction
        features = {}
        
        # Vibe-based features
        vibe_categories = {
            'boho': ['boho', 'bohemian', 'flowy', 'earthy', 'festival'],
            'urban': ['urban', 'streetwear', 'edgy', 'graphic', 'modern'],
            'professional': ['professional', 'tailored', 'office', 'business', 'formal'],
            'cozy': ['cozy', 'warm', 'soft', 'comfortable', 'knit'],
            'athletic': ['athletic', 'sporty', 'performance', 'active', 'gym'],
            'vintage': ['vintage', 'retro', 'classic', 'distressed', 'timeless'],
            'elegant': ['elegant', 'sophisticated', 'luxurious', 'chic', 'glamorous'],
            'minimalist': ['minimalist', 'clean', 'simple', 'refined', 'linen']
        }
        
        # Calculate vibe scores
        vibe_scores = []
        for vibe, keywords in vibe_categories.items():
            score = sum(1 for keyword in keywords if keyword in text.lower())
            vibe_scores.append(score / len(keywords))  # Normalize
        
        # Word frequency features
        word_freq = {}
        for word in words:
            if len(word) > 3:  # Only meaningful words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Create embedding vector
        embedding = np.zeros(self.embedding_dim)
        
        # Fill with vibe scores (first 8 dimensions)
        embedding[:8] = vibe_scores
        
        # Fill with word frequency features (next dimensions)
        for i, (word, freq) in enumerate(list(word_freq.items())[:self.embedding_dim-8]):
            embedding[8 + i] = freq * 0.1
        
        # Add some randomness for diversity (last few dimensions)
        embedding[-10:] = np.random.rand(10) * 0.1
        
        return embedding.tolist()
    
    def get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for a single query"""
        # Always try API first for queries
        api_embedding = self._get_single_embedding_api(query)
        if api_embedding is not None:
            return api_embedding
        else:
            print("   ⚠️ Query API failed, using local embedding")
            return self._get_local_embedding(query)

# Initialize Hugging Face service
embedding_service = HuggingFaceEmbeddingService()

# Generate embeddings
print("\n" + "="*80)
print("🚀 STARTING EMBEDDING GENERATION WITH HUGGING FACE API")
print("="*80)

product_embeddings = embedding_service.get_embeddings_batch(df_products['desc'].tolist())
df_products['embedding'] = product_embeddings

print(f"\n✅ EMBEDDING GENERATION COMPLETE!")
print(f"📊 Provider: {embedding_service.current_provider}")
print(f"📐 Dimension: {len(product_embeddings[0])}")
print(f"🎯 Total Embeddings: {len(product_embeddings)}")

# =====================
# 3. ADVANCED VIBE MATCHER WITH DIVERSITY ENHANCEMENT
# =====================

class AdvancedVibeMatcher:
    """Professional vibe matching system with diversity enhancement"""
    
    def __init__(self, products_df: pd.DataFrame):
        self.products_df = products_df
        self.embedding_matrix = np.array(products_df['embedding'].tolist())
        self.performance_log = []
        
    def get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for query using Hugging Face service"""
        return embedding_service.get_query_embedding(query)
    
    def find_similar_products(self, query: str, top_k: int = 3, 
                            similarity_threshold: float = 0.0,
                            max_price: float = None,
                            enhance_diversity: bool = True) -> List[Dict]:
        """Advanced similarity search with diversity enhancement"""
        
        start_time = time.time()
        
        # Get query embedding
        query_embedding = self.get_query_embedding(query)
        query_vector = np.array(query_embedding).reshape(1, -1)
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.embedding_matrix)[0]
        
        # Apply filters and get candidates
        candidate_indices = []
        for idx, similarity in enumerate(similarities):
            if similarity >= similarity_threshold:
                # Price filter
                if max_price and self.products_df.iloc[idx]['price'] > max_price:
                    continue
                candidate_indices.append((idx, similarity))
        
        # Sort by similarity
        candidate_indices.sort(key=lambda x: x[1], reverse=True)
        
        # Apply diversity enhancement
        if enhance_diversity and len(candidate_indices) > top_k:
            selected_indices = self._diversity_enhanced_selection(candidate_indices, top_k)
        else:
            selected_indices = [idx for idx, _ in candidate_indices[:top_k]]
        
        # Prepare results
        results = []
        for idx in selected_indices:
            product = self.products_df.iloc[idx]
            similarity_score = similarities[idx]
            
            results.append({
                'id': product['id'],
                'product_name': product['name'],
                'description': product['desc'],
                'category': product['category'],
                'price': product['price'],
                'vibes': product['vibes'],
                'season': product['season'],
                'occasion': product['occasion'],
                'similarity_score': round(similarity_score, 4),
                'match_quality': self._get_match_quality(similarity_score),
                'price_tier': self._get_price_tier(product['price'])
            })
        
        # Log performance
        processing_time = round((time.time() - start_time) * 1000, 2)
        self.performance_log.append({
            'query': query,
            'results_count': len(results),
            'processing_time_ms': processing_time,
            'top_score': results[0]['similarity_score'] if results else 0,
            'timestamp': time.time()
        })
        
        return results
    
    def _diversity_enhanced_selection(self, candidates: List[tuple], top_k: int) -> List[int]:
        """Select diverse results to avoid similar products"""
        selected_indices = []
        selected_categories = set()
        
        for idx, similarity in candidates:
            if len(selected_indices) >= top_k:
                break
                
            product_category = self.products_df.iloc[idx]['category']
            
            # If we don't have this category yet, or it's a very high similarity score
            if product_category not in selected_categories or similarity > 0.8:
                selected_indices.append(idx)
                selected_categories.add(product_category)
        
        # If we still need more results, add the highest similarity ones
        if len(selected_indices) < top_k:
            for idx, similarity in candidates:
                if idx not in selected_indices:
                    selected_indices.append(idx)
                    if len(selected_indices) >= top_k:
                        break
        
        return selected_indices
    
    def _get_match_quality(self, score: float) -> str:
        """Enhanced match quality assessment"""
        if score >= 0.8:
            return "Perfect Match 🎯"
        elif score >= 0.7:
            return "Excellent Match ⭐"
        elif score >= 0.6:
            return "Very Good Match 👍"
        elif score >= 0.5:
            return "Good Match ✅"
        elif score >= 0.4:
            return "Reasonable Match 👌"
        elif score >= 0.3:
            return "Fair Match 🤔"
        else:
            return "Weak Match 📉"
    
    def _get_price_tier(self, price: float) -> str:
        """Categorize price into tiers"""
        if price < 50:
            return "Budget 💰"
        elif price < 100:
            return "Moderate 💵"
        elif price < 150:
            return "Premium 💎"
        else:
            return "Luxury ✨"
    
    def display_enhanced_results(self, results: List[Dict], query: str):
        """Professional results display"""
        print(f"\n🎯 MATCH RESULTS FOR: '{query}'")
        print("=" * 80)
        
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['product_name']}")
            print(f"   📁 Category: {result['category']}")
            print(f"   💰 Price: ${result['price']} ({result['price_tier']})")
            print(f"   🏷️  Vibes: {', '.join(result['vibes'])}")
            print(f"   🎯 Match Score: {result['similarity_score']} - {result['match_quality']}")
            print(f"   🌟 Occasions: {', '.join(result['occasion'])}")
            print(f"   📅 Seasons: {', '.join(result['season'])}")
            print()
    
    def generate_analytics_report(self):
        """Generate comprehensive analytics report"""
        if not self.performance_log:
            return "No analytics data available yet."
        
        df_log = pd.DataFrame(self.performance_log)
        
        report = {
            'total_queries': len(df_log),
            'avg_processing_time': df_log['processing_time_ms'].mean(),
            'avg_similarity_score': df_log['top_score'].mean(),
            'success_rate': (df_log['results_count'] > 0).mean(),
            'performance_trend': df_log['processing_time_ms'].tolist()
        }
        
        return report

# Initialize advanced matcher
advanced_matcher = AdvancedVibeMatcher(df_products)

# =====================
# 4. COMPREHENSIVE TESTING WITH DIVERSE QUERIES
# =====================

print("\n" + "="*80)
print("🧪 COMPREHENSIVE TESTING WITH DIVERSE QUERIES")
print("="*80)

# Diverse test queries to demonstrate different matching
test_scenarios = [
    {
        "name": "Urban Street Style",
        "query": "streetwear urban graphic hoodie casual modern",
        "expected_category": "Tops",
        "max_price": 100
    },
    {
        "name": "Formal Evening Wear", 
        "query": "elegant sophisticated formal dress silk cocktail evening",
        "expected_category": "Dresses",
        "max_price": 250
    },
    {
        "name": "Cozy Winter Comfort",
        "query": "warm cozy comfortable sweater cashmere winter soft",
        "expected_category": "Tops",
        "max_price": 200
    },
    {
        "name": "Professional Business",
        "query": "professional office business tailored blazer formal",
        "expected_category": "Outerwear",
        "max_price": 150
    },
    {
        "name": "Athletic Performance",
        "query": "athletic sporty performance active gym running",
        "expected_category": "Activewear",
        "max_price": None
    },
    {
        "name": "Random Unrelated",
        "query": "completely unrelated technology terms here",
        "expected_category": None,
        "max_price": None
    }
]

# Run comprehensive evaluation
performance_metrics = []

print("\n🔍 RUNNING COMPREHENSIVE TEST SCENARIOS...")

for i, scenario in enumerate(test_scenarios, 1):
    print(f"\n{'='*60}")
    print(f"TEST {i}: {scenario['name']}")
    print(f"Query: '{scenario['query']}'")
    print(f"Filters: Max Price ${scenario['max_price'] if scenario['max_price'] else 'None'}")
    print('='*60)
    
    # Execute search
    results = advanced_matcher.find_similar_products(
        query=scenario['query'],
        top_k=3,
        similarity_threshold=0.0,
        max_price=scenario['max_price'],
        enhance_diversity=True
    )
    
    # Display results
    advanced_matcher.display_enhanced_results(results, scenario['query'])
    
    # Calculate metrics
    if results:
        metrics = {
            'scenario': scenario['name'],
            'query': scenario['query'],
            'top_score': results[0]['similarity_score'],
            'results_count': len(results),
            'unique_categories': len(set(r['category'] for r in results)),
            'avg_price': np.mean([r['price'] for r in results]),
            'category_match': any(r['category'] == scenario['expected_category'] for r in results) if scenario['expected_category'] else None,
            'processing_time': advanced_matcher.performance_log[-1]['processing_time_ms']
        }
    else:
        metrics = {
            'scenario': scenario['name'],
            'query': scenario['query'], 
            'top_score': 0,
            'results_count': 0,
            'unique_categories': 0,
            'avg_price': 0,
            'category_match': False,
            'processing_time': advanced_matcher.performance_log[-1]['processing_time_ms']
        }
    
    performance_metrics.append(metrics)

# =====================
# 5. PROFESSIONAL ANALYTICS DASHBOARD
# =====================

print("\n" + "="*80)
print("📊 PROFESSIONAL ANALYTICS DASHBOARD")
print("="*80)

# Convert metrics to DataFrame
df_metrics = pd.DataFrame(performance_metrics)

# Overall Performance Summary
print("\n📈 OVERALL PERFORMANCE SUMMARY")
print("="*50)
print(f"🏆 Average Similarity Score: {df_metrics['top_score'].mean():.3f}")
print(f"⚡ Average Processing Time: {df_metrics['processing_time'].mean():.2f}ms")
print(f"🎯 Successful Queries: {len([m for m in performance_metrics if m['results_count'] > 0])}/{len(performance_metrics)}")
print(f"🔄 Average Unique Categories per Query: {df_metrics['unique_categories'].mean():.1f}")
if df_metrics['category_match'].notna().any():
    print(f"📊 Category Match Accuracy: {df_metrics['category_match'].mean():.1%}")

# Enhanced visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Vibe Matcher Pro - Performance Analytics', fontsize=16, fontweight='bold')

# Plot 1: Similarity Scores
scenarios = [m['scenario'] for m in performance_metrics]
scores = [m['top_score'] for m in performance_metrics]
colors = ['green' if score > 0.6 else 'orange' if score > 0.4 else 'red' for score in scores]

bars = ax1.bar(scenarios, scores, color=colors, alpha=0.7)
ax1.set_title('Similarity Scores by Query Type', fontweight='bold')
ax1.set_ylabel('Similarity Score')
ax1.set_ylim(0, 1)
ax1.tick_params(axis='x', rotation=45)
for bar, score in zip(bars, scores):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{score:.3f}', 
             ha='center', va='bottom', fontweight='bold')

# Plot 2: Processing Time
times = [m['processing_time'] for m in performance_metrics]
ax2.bar(scenarios, times, color='skyblue', alpha=0.7)
ax2.set_title('Query Processing Time (ms)', fontweight='bold')
ax2.set_ylabel('Time (ms)')
ax2.tick_params(axis='x', rotation=45)
for i, time_val in enumerate(times):
    ax2.text(i, time_val + 5, f'{time_val}ms', ha='center', va='bottom')

# Plot 3: Results Diversity
unique_cats = [m['unique_categories'] for m in performance_metrics]
ax3.bar(scenarios, unique_cats, color='lightgreen', alpha=0.7)
ax3.set_title('Unique Categories per Query', fontweight='bold')
ax3.set_ylabel('Unique Categories')
ax3.tick_params(axis='x', rotation=45)
for i, cats in enumerate(unique_cats):
    ax3.text(i, cats + 0.1, str(cats), ha='center', va='bottom', fontweight='bold')

# Plot 4: Success Rate
success_rates = [1 if m['results_count'] > 0 else 0 for m in performance_metrics]
ax4.bar(scenarios, success_rates, color=['green' if rate else 'red' for rate in success_rates], alpha=0.7)
ax4.set_title('Query Success Rate', fontweight='bold')
ax4.set_ylabel('Success (1) / Failure (0)')
ax4.set_ylim(0, 1)
ax4.tick_params(axis='x', rotation=45)
for i, rate in enumerate(success_rates):
    ax4.text(i, rate + 0.05, 'Success' if rate else 'No Results', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# =====================
# 6. INTERACTIVE DEMO WITH USER INPUT
# =====================

print("\n" + "="*80)
print("🎮 INTERACTIVE VIBE MATCHER PRO DEMO")
print("="*80)

def safe_input(prompt):
    """Safe input function for various environments"""
    try:
        return input(prompt)
    except (EOFError, KeyboardInterrupt):
        return "quit"

def run_interactive_demo():
    """Interactive demo with user queries"""
    print("\n💡 ENHANCED FEATURES:")
    print("   • Hugging Face API embeddings")
    print("   • Diversity-enhanced results")
    print("   • Price filtering (add 'under $100' to query)")
    print("   • Real-time similarity scoring")
    
    print("\n🎯 SAMPLE QUERIES TO TRY:")
    sample_queries = [
        "urban streetwear under $100",
        "elegant formal dress for wedding", 
        "cozy winter sweater affordable",
        "professional office wear",
        "vintage denim jacket casual",
        "athletic sporty clothes",
        "minimalist clean style"
    ]
    
    for i, query in enumerate(sample_queries, 1):
        print(f"   {i}. '{query}'")
    
    print(f"\n{'='*60}")
    print("Starting Interactive Session...")
    print("="*60)
    
    query_count = 0
    max_queries = 8
    
    while query_count < max_queries:
        print(f"\n➡️  Enter your vibe query #{query_count + 1} (or 'quit' to exit):")
        
        user_query = safe_input("🎯 Your query: ").strip()
        
        if user_query.lower() in ['quit', 'exit', 'q', '']:
            print("\n👋 Thank you for using Vibe Matcher Pro!")
            break
        
        # Extract price filter
        max_price = None
        price_match = re.search(r'under \$?(\d+)', user_query.lower())
        if price_match:
            max_price = float(price_match.group(1))
            user_query = re.sub(r'under \$?\d+', '', user_query).strip()
            print(f"💰 Price filter applied: Under ${max_price}")
        
        print("🔄 Processing with Hugging Face API...")
        start_time = time.time()
        
        # Use diversity enhancement for better results
        results = advanced_matcher.find_similar_products(
            query=user_query,
            top_k=3,
            max_price=max_price,
            enhance_diversity=True
        )
        
        processing_time = round((time.time() - start_time) * 1000, 2)
        
        if results:
            advanced_matcher.display_enhanced_results(results, user_query)
            print(f"⚡ Processed in {processing_time}ms")
            print(f"🔧 Embedding Provider: {embedding_service.current_provider}")
            
            # Show detailed insights
            top_result = results[0]
            print("\n💡 MATCH ANALYSIS:")
            print(f"   • Best Match Score: {top_result['similarity_score']}")
            print(f"   • Categories Found: {len(set(r['category'] for r in results))}")
            
            if top_result['similarity_score'] > 0.5:
                query_words = set(user_query.lower().split())
                matching_vibes = set(top_result['vibes']).intersection(query_words)
                if matching_vibes:
                    print(f"   • Shared Vibes: {', '.join(matching_vibes)}")
        else:
            print("❌ No suitable matches found. Try different search terms!")
        
        query_count += 1
        
        if query_count < max_queries:
            print(f"\n💡 You have {max_queries - query_count} queries remaining.")
        else:
            print("\n🎉 Demo complete! Thanks for testing Vibe Matcher Pro!")

# Run interactive demo
run_interactive_demo()

# =====================
# 7. FINAL EXECUTIVE SUMMARY
# =====================

print("\n" + "="*80)
print("🎉 VIBE MATCHER PRO - EXECUTIVE SUMMARY")
print("="*80)

final_analytics = advanced_matcher.generate_analytics_report()

print(f"""
📊 PERFORMANCE HIGHLIGHTS:
• Embedding Provider: {embedding_service.current_provider}
• Total Products: {len(df_products)} diverse fashion items
• Average Processing Time: {final_analytics['avg_processing_time']:.1f}ms
• Query Success Rate: {final_analytics['success_rate']:.1%}
• Average Similarity Score: {final_analytics['avg_similarity_score']:.3f}

🚀 KEY INNOVATIONS:
1. **Hugging Face API Integration**: Real semantic understanding
2. **Diversity Enhancement**: Avoids similar product recommendations
3. **Robust Fallback System**: Works even when APIs are unavailable
4. **Professional Analytics**: Comprehensive performance monitoring
5. **Enhanced User Experience**: Interactive demo with insights

💼 BUSINESS VALUE:
• **Higher Conversion**: More relevant, diverse recommendations
• **Better UX**: Semantic understanding of fashion preferences
• **Scalable**: Ready for thousands of products
• **Cost-Effective**: Smart API usage with local fallbacks""")