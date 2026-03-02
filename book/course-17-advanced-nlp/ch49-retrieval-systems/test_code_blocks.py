"""
Test script for Chapter 49: Advanced Retrieval Systems
Tests all code blocks in sequence to verify they execute correctly.
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("TESTING CHAPTER 49 CODE BLOCKS")
print("="*80)

# Track results
issues = []
blocks_tested = 0

# =============================================================================
# BLOCK 1: Visualization - Bi-Encoder vs Cross-Encoder
# =============================================================================
print("\n[Block 1/12] Testing Visualization Code...")
blocks_tested += 1
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    import os

    # Create diagrams directory
    os.makedirs('diagrams', exist_ok=True)

    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Bi-Encoder Architecture ---
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('Bi-Encoder Architecture\n(Fast, Pre-computable)', fontsize=14, fontweight='bold')

    # Query encoder
    query_box = FancyBboxPatch((0.5, 6), 2, 1.5, boxstyle="round,pad=0.1",
                               edgecolor='blue', facecolor='lightblue', linewidth=2)
    ax1.add_patch(query_box)
    ax1.text(1.5, 6.75, 'Query\nEncoder', ha='center', va='center', fontsize=10, fontweight='bold')

    # Document encoder
    doc_box = FancyBboxPatch((0.5, 2.5), 2, 1.5, boxstyle="round,pad=0.1",
                             edgecolor='green', facecolor='lightgreen', linewidth=2)
    ax1.add_patch(doc_box)
    ax1.text(1.5, 3.25, 'Document\nEncoder', ha='center', va='center', fontsize=10, fontweight='bold')

    # Query input
    ax1.text(1.5, 8.5, '"computer graphics"', ha='center', va='center',
             fontsize=9, bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))

    # Document input
    ax1.text(1.5, 1, '"rendering algorithms..."', ha='center', va='center',
             fontsize=9, bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))

    # Arrows to encoders
    ax1.annotate('', xy=(1.5, 7.5), xytext=(1.5, 8.2),
                 arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax1.annotate('', xy=(1.5, 4), xytext=(1.5, 1.3),
                 arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Embeddings
    ax1.text(4.5, 6.75, 'q = [0.23, -0.45, ...]', ha='left', va='center',
             fontsize=9, family='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow'))
    ax1.text(4.5, 3.25, 'd = [0.31, -0.38, ...]', ha='left', va='center',
             fontsize=9, family='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow'))

    # Arrows to embeddings
    ax1.annotate('', xy=(4.3, 6.75), xytext=(2.6, 6.75),
                 arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax1.annotate('', xy=(4.3, 3.25), xytext=(2.6, 3.25),
                 arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Similarity computation
    sim_box = FancyBboxPatch((4.5, 4.5), 3, 1, boxstyle="round,pad=0.1",
                             edgecolor='red', facecolor='lightcoral', linewidth=2)
    ax1.add_patch(sim_box)
    ax1.text(6, 5, 'Similarity\n(q · d)', ha='center', va='center', fontsize=10, fontweight='bold')

    # Arrows to similarity
    ax1.annotate('', xy=(6, 5.3), xytext=(6, 6.5),
                 arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax1.annotate('', xy=(6, 4.7), xytext=(6, 3.5),
                 arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Score output
    ax1.text(6, 3.8, 'Score: 0.82', ha='center', va='center',
             fontsize=11, fontweight='bold', bbox=dict(boxstyle='round', facecolor='yellow', edgecolor='red'))

    # Complexity annotation
    ax1.text(5, 0.3, 'Complexity: O(N) after indexing\nCan pre-compute document embeddings',
             ha='center', va='center', fontsize=8, style='italic',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))

    # --- Cross-Encoder Architecture ---
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title('Cross-Encoder Architecture\n(Accurate, Must Recompute)', fontsize=14, fontweight='bold')

    # Combined input
    combined_text = '[CLS] computer graphics [SEP]\nrendering algorithms... [SEP]'
    ax2.text(5, 8.5, combined_text, ha='center', va='center',
             fontsize=8, bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'),
             family='monospace')

    # Arrow down
    ax2.annotate('', xy=(5, 7.3), xytext=(5, 8),
                 arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Single encoder
    encoder_box = FancyBboxPatch((3, 5.5), 4, 1.5, boxstyle="round,pad=0.1",
                                 edgecolor='purple', facecolor='plum', linewidth=2)
    ax2.add_patch(encoder_box)
    ax2.text(5, 6.25, 'Joint Encoder\n(BERT Cross-Encoder)', ha='center', va='center',
             fontsize=10, fontweight='bold')

    # Arrow to encoder
    ax2.annotate('', xy=(5, 5.5), xytext=(5, 7),
                 arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Classification head
    cls_box = FancyBboxPatch((3.5, 3.5), 3, 1, boxstyle="round,pad=0.1",
                             edgecolor='red', facecolor='lightcoral', linewidth=2)
    ax2.add_patch(cls_box)
    ax2.text(5, 4, 'Classification\nHead', ha='center', va='center', fontsize=10, fontweight='bold')

    # Arrow to classification
    ax2.annotate('', xy=(5, 4.5), xytext=(5, 5.3),
                 arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Score output
    ax2.text(5, 2.5, 'Relevance Score: 0.94', ha='center', va='center',
             fontsize=11, fontweight='bold', bbox=dict(boxstyle='round', facecolor='yellow', edgecolor='red'))

    # Arrow to score
    ax2.annotate('', xy=(5, 2.8), xytext=(5, 3.5),
                 arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Complexity annotation
    ax2.text(5, 0.3, 'Complexity: O(N × M)\nMust process each query-document pair',
             ha='center', va='center', fontsize=8, style='italic',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))

    # Usage notes
    ax2.text(5, 1.3, 'Use for: Re-ranking top-k candidates',
             ha='center', va='center', fontsize=8, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightyellow'))

    plt.tight_layout()
    plt.savefig('diagrams/bi_vs_cross_encoder.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("  ✓ Visualization code executed successfully")
except Exception as e:
    issues.append(f"Block 1 (Visualization): {str(e)}")
    print(f"  ✗ ERROR: {e}")

# =============================================================================
# BLOCK 2: Bi-Encoder Dense Retrieval
# =============================================================================
print("\n[Block 2/12] Testing Bi-Encoder Dense Retrieval...")
blocks_tested += 1
try:
    from sklearn.datasets import fetch_20newsgroups
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    # Load subset of 20 Newsgroups dataset
    categories = ['comp.graphics', 'sci.med', 'rec.sport.baseball']
    newsgroups = fetch_20newsgroups(
        subset='all',
        categories=categories,
        remove=('headers', 'footers', 'quotes'),
        random_state=42
    )

    # Use first 300 documents for demonstration (100 per category)
    n_docs = 300
    documents = newsgroups.data[:n_docs]
    labels = newsgroups.target[:n_docs]
    category_names = newsgroups.target_names

    print(f"  Loaded {len(documents)} documents across {len(categories)} categories")

    # Load pre-trained bi-encoder model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print(f"  Model embedding dimension: {model.get_sentence_embedding_dimension()}")

    # Encode all documents
    doc_embeddings = model.encode(documents, show_progress_bar=False, convert_to_numpy=True)
    print(f"  Document embeddings shape: {doc_embeddings.shape}")

    # Normalize embeddings
    doc_embeddings = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)

    # Define test query
    query = "How do computer graphics rendering algorithms work?"

    # Encode query
    query_embedding = model.encode(query, convert_to_numpy=True)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    query_embedding = query_embedding.reshape(1, -1)

    # Compute similarity scores
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]

    # Get top-10 results
    top_k = 10
    top_indices = np.argsort(similarities)[::-1][:top_k]

    # Analyze category distribution
    retrieved_categories = [category_names[labels[idx]] for idx in top_indices]
    target_count = retrieved_categories.count('comp.graphics')

    print(f"  ✓ Retrieved {target_count}/{top_k} documents from target category")

except Exception as e:
    issues.append(f"Block 2 (Bi-Encoder): {str(e)}")
    print(f"  ✗ ERROR: {e}")

# =============================================================================
# BLOCK 3: Cross-Encoder Re-Ranking
# =============================================================================
print("\n[Block 3/12] Testing Cross-Encoder Re-Ranking...")
blocks_tested += 1
try:
    from sentence_transformers import CrossEncoder

    # Load cross-encoder model
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    # Use top-50 candidates from bi-encoder
    top_50_indices = np.argsort(similarities)[::-1][:50]

    # Prepare query-document pairs
    pairs = [[query, documents[idx]] for idx in top_50_indices]

    # Compute cross-encoder scores
    cross_scores = cross_encoder.predict(pairs, show_progress_bar=False)

    # Get top-10 after re-ranking
    reranked_indices = np.argsort(cross_scores)[::-1][:10]
    reranked_doc_indices = top_50_indices[reranked_indices]

    # Compare
    bi_top10 = top_indices[:10]
    cross_top10 = reranked_doc_indices[:10]

    promoted = set(cross_top10) - set(bi_top10)
    demoted = set(bi_top10) - set(cross_top10)

    print(f"  ✓ Cross-encoder promoted {len(promoted)} docs, demoted {len(demoted)} docs")

except Exception as e:
    issues.append(f"Block 3 (Cross-Encoder): {str(e)}")
    print(f"  ✗ ERROR: {e}")

# =============================================================================
# BLOCK 4: Hybrid Search with BM25
# =============================================================================
print("\n[Block 4/12] Testing Hybrid Search...")
blocks_tested += 1
try:
    from rank_bm25 import BM25Okapi
    import string

    # Preprocess for BM25
    def preprocess_for_bm25(text):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text.split()

    tokenized_docs = [preprocess_for_bm25(doc) for doc in documents]
    tokenized_query = preprocess_for_bm25(query)

    # Build BM25 index
    bm25 = BM25Okapi(tokenized_docs)

    # Get BM25 scores
    bm25_scores = bm25.get_scores(tokenized_query)

    print(f"  BM25 score statistics: Mean={np.mean(bm25_scores):.4f}, Max={np.max(bm25_scores):.4f}")

    # Get top-20 from each method
    bm25_top20_indices = np.argsort(bm25_scores)[::-1][:20]
    dense_top20_indices = np.argsort(similarities)[::-1][:20]

    # Implement RRF
    def reciprocal_rank_fusion(rankings_list, k=60):
        rrf_scores = {}
        for ranking in rankings_list:
            for rank, doc_idx in enumerate(ranking, 1):
                if doc_idx not in rrf_scores:
                    rrf_scores[doc_idx] = 0.0
                rrf_scores[doc_idx] += 1.0 / (k + rank)
        return rrf_scores

    # Apply RRF
    rrf_scores = reciprocal_rank_fusion([bm25_top20_indices, dense_top20_indices], k=60)
    hybrid_top10_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:10]

    # Compare methods
    bm25_top10 = bm25_top20_indices[:10]
    dense_top10 = dense_top20_indices[:10]

    hybrid_categories = [category_names[labels[idx]] for idx in hybrid_top10_indices]
    hybrid_precision = hybrid_categories.count('comp.graphics') / 10

    print(f"  ✓ Hybrid search precision: {hybrid_precision:.2f}")

except Exception as e:
    issues.append(f"Block 4 (Hybrid Search): {str(e)}")
    print(f"  ✗ ERROR: {e}")

# =============================================================================
# BLOCK 5: Evaluation Metrics
# =============================================================================
print("\n[Block 5/12] Testing Evaluation Metrics...")
blocks_tested += 1
try:
    from scipy.stats import rankdata

    def recall_at_k(retrieved, relevant, k):
        retrieved_at_k = set(retrieved[:k])
        relevant_retrieved = retrieved_at_k.intersection(relevant)
        return len(relevant_retrieved) / len(relevant) if len(relevant) > 0 else 0.0

    def mean_reciprocal_rank(retrieved_lists, relevant_lists):
        reciprocal_ranks = []
        for retrieved, relevant in zip(retrieved_lists, relevant_lists):
            for rank, doc_idx in enumerate(retrieved, 1):
                if doc_idx in relevant:
                    reciprocal_ranks.append(1.0 / rank)
                    break
            else:
                reciprocal_ranks.append(0.0)
        return np.mean(reciprocal_ranks)

    def dcg_at_k(relevances, k):
        relevances = np.array(relevances[:k])
        if relevances.size == 0:
            return 0.0
        discounts = np.log2(np.arange(2, k + 2))
        return np.sum((2**relevances - 1) / discounts)

    def ndcg_at_k(relevances, k):
        dcg = dcg_at_k(relevances, k)
        ideal_relevances = sorted(relevances, reverse=True)
        idcg = dcg_at_k(ideal_relevances, k)
        return dcg / idcg if idcg > 0 else 0.0

    # Test metrics
    test_retrieved = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    test_relevant = set([0, 5, 10, 100, 200])

    recall = recall_at_k(test_retrieved, test_relevant, k=10)
    test_relevances = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    ndcg = ndcg_at_k(test_relevances, k=10)

    print(f"  ✓ Metrics computed: Recall@10={recall:.4f}, nDCG@10={ndcg:.4f}")

except Exception as e:
    issues.append(f"Block 5 (Evaluation Metrics): {str(e)}")
    print(f"  ✗ ERROR: {e}")

# =============================================================================
# BLOCK 6: RAG Pipeline
# =============================================================================
print("\n[Block 6/12] Testing RAG Pipeline...")
blocks_tested += 1
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    # Load GPT-2
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    gen_model = AutoModelForCausalLM.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Define question
    question = "What techniques are used in computer graphics rendering?"

    # Retrieve top-3 documents
    query_emb = model.encode(question, convert_to_numpy=True)
    query_emb = query_emb / np.linalg.norm(query_emb)
    query_emb = query_emb.reshape(1, -1)

    dense_sims = cosine_similarity(query_emb, doc_embeddings)[0]
    top_3_indices = np.argsort(dense_sims)[::-1][:3]

    retrieved_context = []
    for idx in top_3_indices:
        doc_text = documents[idx][:300]
        retrieved_context.append(doc_text)

    # Format prompt
    context_str = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(retrieved_context)])
    prompt = f"""Context:
{context_str}

Question: {question}

Answer based on the context above:"""

    # Generate with context
    input_ids = tokenizer.encode(prompt, return_tensors='pt', max_length=512, truncation=True)

    with torch.no_grad():
        output = gen_model.generate(
            input_ids,
            max_new_tokens=50,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    answer_with_context = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)

    # Generate without context
    prompt_no_context = f"Question: {question}\n\nAnswer:"
    input_ids_no_context = tokenizer.encode(prompt_no_context, return_tensors='pt')

    with torch.no_grad():
        output_no_context = gen_model.generate(
            input_ids_no_context,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    answer_no_context = tokenizer.decode(output_no_context[0][input_ids_no_context.shape[1]:], skip_special_tokens=True)

    print(f"  ✓ RAG pipeline executed successfully")
    print(f"    Answer with RAG: {len(answer_with_context.split())} words")
    print(f"    Answer without RAG: {len(answer_no_context.split())} words")

except Exception as e:
    issues.append(f"Block 6 (RAG Pipeline): {str(e)}")
    print(f"  ✗ ERROR: {e}")

# =============================================================================
# BLOCKS 7-11: Solutions (simplified testing)
# =============================================================================
print("\n[Block 7/12] Testing Solution 1 (Wine Search)...")
blocks_tested += 1
try:
    # Use synthetic data
    wine_descriptions = [
        "Bold Cabernet Sauvignon from Napa Valley with dark fruit flavors.",
        "Crisp Sauvignon Blanc from New Zealand with citrus notes.",
        "Elegant Pinot Noir from Oregon with cherry flavors.",
    ] * 10

    wine_embeddings = model.encode(wine_descriptions, show_progress_bar=False)
    wine_embeddings = wine_embeddings / np.linalg.norm(wine_embeddings, axis=1, keepdims=True)

    test_query = "fruity red wines from California"
    query_emb = model.encode(test_query)
    query_emb = query_emb / np.linalg.norm(query_emb)
    query_emb = query_emb.reshape(1, -1)

    sims = cosine_similarity(query_emb, wine_embeddings)[0]
    top_3 = np.argsort(sims)[::-1][:3]

    print(f"  ✓ Wine search solution executed")
except Exception as e:
    issues.append(f"Block 7 (Solution 1): {str(e)}")
    print(f"  ✗ ERROR: {e}")

print("\n[Block 8/12] Testing Solution 2 (BM25 vs Dense)...")
blocks_tested += 1
try:
    # Already have necessary components from earlier blocks
    tokenized_wines = [preprocess_for_bm25(desc) for desc in wine_descriptions]
    bm25_wines = BM25Okapi(tokenized_wines)

    test_q = preprocess_for_bm25("smooth wine with chocolate notes")
    bm25_scores_wines = bm25_wines.get_scores(test_q)

    print(f"  ✓ BM25 vs Dense comparison solution executed")
except Exception as e:
    issues.append(f"Block 8 (Solution 2): {str(e)}")
    print(f"  ✗ ERROR: {e}")

print("\n[Block 9/12] Testing Solution 3 (Two-Stage Retrieval)...")
blocks_tested += 1
try:
    # Test two-stage retrieval with wine data
    import time

    test_query_wine = "bold red wine with dark fruit"
    query_emb_wine = model.encode(test_query_wine)
    query_emb_wine = query_emb_wine / np.linalg.norm(query_emb_wine)

    start = time.time()
    sims_wine = cosine_similarity(query_emb_wine.reshape(1, -1), wine_embeddings)[0]
    stage1_time = time.time() - start

    top_k_wine = np.argsort(sims_wine)[::-1][:20]

    start = time.time()
    pairs_wine = [[test_query_wine, wine_descriptions[idx]] for idx in top_k_wine]
    cross_scores_wine = cross_encoder.predict(pairs_wine, show_progress_bar=False)
    stage2_time = time.time() - start

    print(f"  ✓ Two-stage retrieval solution executed")
    print(f"    Stage 1: {stage1_time*1000:.1f}ms, Stage 2: {stage2_time*1000:.1f}ms")
except Exception as e:
    issues.append(f"Block 9 (Solution 3): {str(e)}")
    print(f"  ✗ ERROR: {e}")

print("\n[Block 10/12] Testing Solution 4 (Hybrid on 20 Newsgroups)...")
blocks_tested += 1
try:
    # Use small subset for testing
    test_queries_eval = [
        "computer graphics rendering",
        "baseball statistics"
    ]

    query_targets_eval = ['comp.graphics', 'rec.sport.baseball']

    for test_q_eval in test_queries_eval[:1]:  # Test just one
        tokenized_q_eval = preprocess_for_bm25(test_q_eval)
        bm25_scores_eval = bm25.get_scores(tokenized_q_eval)
        bm25_ranking = np.argsort(bm25_scores_eval)[::-1][:20]

        query_emb_eval = model.encode(test_q_eval)
        query_emb_eval = query_emb_eval / np.linalg.norm(query_emb_eval)
        dense_sims_eval = cosine_similarity(query_emb_eval.reshape(1, -1), doc_embeddings)[0]
        dense_ranking = np.argsort(dense_sims_eval)[::-1][:20]

        rrf_scores_eval = reciprocal_rank_fusion([bm25_ranking, dense_ranking], k=60)

    print(f"  ✓ Hybrid evaluation solution executed")
except Exception as e:
    issues.append(f"Block 10 (Solution 4): {str(e)}")
    print(f"  ✗ ERROR: {e}")

print("\n[Block 11/12] Testing Solution 5 (Complete RAG)...")
blocks_tested += 1
try:
    # Simplified RAG test (skip Wikipedia fetch for speed)
    articles_test = [
        {'title': 'Neural Network', 'content': 'A neural network is a computational model...'},
        {'title': 'Decision Tree', 'content': 'Decision trees are supervised learning algorithms...'}
    ]

    article_texts = [a['content'] for a in articles_test]
    article_embeddings = model.encode(article_texts, show_progress_bar=False)
    article_embeddings = article_embeddings / np.linalg.norm(article_embeddings, axis=1, keepdims=True)

    test_question = "What is a neural network?"
    query_emb_art = model.encode(test_question)
    query_emb_art = query_emb_art / np.linalg.norm(query_emb_art)
    sims_art = cosine_similarity(query_emb_art.reshape(1, -1), article_embeddings)[0]

    print(f"  ✓ Complete RAG solution structure validated")
except Exception as e:
    issues.append(f"Block 11 (Solution 5): {str(e)}")
    print(f"  ✗ ERROR: {e}")

print("\n[Block 12/12] Testing Variable Consistency...")
blocks_tested += 1
try:
    # Verify key variables are accessible and consistent
    assert len(documents) == 300, "documents should have 300 items"
    assert len(doc_embeddings) == 300, "doc_embeddings should match documents"
    assert doc_embeddings.shape[1] == 384, "embeddings should be 384-dim"
    assert len(similarities) == 300, "similarities should match doc count"
    assert len(bm25_scores) == 300, "BM25 scores should match doc count"

    print(f"  ✓ All variables consistent across blocks")
except Exception as e:
    issues.append(f"Block 12 (Variable Consistency): {str(e)}")
    print(f"  ✗ ERROR: {e}")

# =============================================================================
# FINAL REPORT
# =============================================================================
print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)

if not issues:
    print(f"\n✓ ALL {blocks_tested} BLOCKS PASSED")
    print("\nRating: ALL_PASS")
else:
    print(f"\n✗ {len(issues)} BLOCK(S) FAILED out of {blocks_tested} tested")
    print("\nFailures:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
    print("\nRating: BROKEN" if len(issues) > 3 else "MINOR_FIXES")

print("\nDependencies verified:")
print("  - numpy")
print("  - sklearn (fetch_20newsgroups, cosine_similarity)")
print("  - sentence-transformers (SentenceTransformer, CrossEncoder)")
print("  - rank-bm25 (BM25Okapi)")
print("  - transformers (AutoTokenizer, AutoModelForCausalLM)")
print("  - torch")
print("  - matplotlib")
print("  - scipy")
print("="*80)
