"""
Generate all diagrams for Chapter 27: Classical NLP

This script generates all figures referenced in the content.md file.
All diagrams use a consistent color palette and styling.

Color Palette:
- Blue: #2196F3
- Green: #4CAF50
- Orange: #FF9800
- Red: #F44336
- Purple: #9C27B0
- Gray: #607D8B

Requirements:
- All figures are 150 DPI
- Max width: 800px (approx 5.33 inches at 150 DPI)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set consistent style
plt.style.use('default')
sns.set_palette("husl")

# Color palette
COLORS = {
    'blue': '#2196F3',
    'green': '#4CAF50',
    'orange': '#FF9800',
    'red': '#F44336',
    'purple': '#9C27B0',
    'gray': '#607D8B'
}

# Max width in inches at 150 DPI for 800px
MAX_WIDTH = 5.33

def print_progress(message):
    """Print progress message."""
    print(f"[INFO] {message}")

def print_success(message):
    """Print success message."""
    print(f"[SUCCESS] {message}")

def print_error(message):
    """Print error message."""
    print(f"[ERROR] {message}")


def generate_preprocessing_pipeline():
    """Generate preprocessing pipeline flowchart."""
    print_progress("Generating preprocessing_pipeline.png...")

    try:
        fig, ax = plt.subplots(figsize=(MAX_WIDTH, 7))
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 10)
        ax.axis('off')

        # Define stages
        stages = [
            ("Raw Text", "The movie wasn't AMAZING!!!", 1),
            ("Lowercase", "the movie wasn't amazing!!!", 2),
            ("Tokenize", "['the', 'movie', \"wasn't\", 'amazing', '!', '!', '!']", 3),
            ("Remove Punctuation", "['the', 'movie', \"wasn't\", 'amazing']", 4),
            ("Remove Stop Words", "['movie', \"wasn't\", 'amazing']", 5),
            ("Stem/Lemmatize", "['movie', 'be', 'amazing']", 6),
        ]

        y_pos = 8.5
        for i, (title, content, step) in enumerate(stages):
            # Draw box
            box = FancyBboxPatch((0.5, y_pos - 0.6), 13, 1,
                                 boxstyle="round,pad=0.1",
                                 edgecolor=COLORS['blue'],
                                 facecolor='#A3D9FF' if i % 2 == 0 else '#D4F1F4',
                                 linewidth=2)
            ax.add_patch(box)

            # Add text
            ax.text(1, y_pos - 0.1, f"Step {step}: {title}",
                    fontsize=10, fontweight='bold', va='center')
            ax.text(1, y_pos - 0.45, content,
                    fontsize=8, va='center', style='italic')

            # Draw arrow to next stage
            if i < len(stages) - 1:
                ax.arrow(7, y_pos - 0.7, 0, -0.4,
                        head_width=0.3, head_length=0.15,
                        fc=COLORS['blue'], ec=COLORS['blue'], linewidth=2)

            y_pos -= 1.5

        ax.text(7, 0.5, "Text Preprocessing Pipeline",
                fontsize=13, fontweight='bold', ha='center')

        plt.tight_layout()
        plt.savefig('preprocessing_pipeline.png', dpi=150, bbox_inches='tight')
        plt.close()

        print_success("preprocessing_pipeline.png created successfully")
        return True

    except Exception as e:
        print_error(f"Failed to generate preprocessing_pipeline.png: {str(e)}")
        return False


def generate_bow_matrix_heatmap():
    """Generate Bag-of-Words matrix heatmap."""
    print_progress("Generating bow_matrix_heatmap.png...")

    try:
        from sklearn.feature_extraction.text import CountVectorizer

        # Sample corpus
        corpus = [
            "The movie was excellent and entertaining",
            "Great movie with excellent acting",
            "The film was boring and disappointing",
            "Terrible movie with poor acting",
            "Excellent film with great story and acting"
        ]

        # Create Bag-of-Words vectorizer
        vectorizer = CountVectorizer(lowercase=True, token_pattern=r'\b\w+\b')
        X_bow = vectorizer.fit_transform(corpus)
        vocabulary = vectorizer.get_feature_names_out()
        X_bow_dense = X_bow.toarray()

        # Create DataFrame
        df_bow = pd.DataFrame(X_bow_dense,
                              columns=vocabulary,
                              index=[f"Doc {i+1}" for i in range(len(corpus))])

        # Visualize
        fig, ax = plt.subplots(figsize=(MAX_WIDTH, 4))
        sns.heatmap(df_bow, annot=True, fmt='d', cmap='YlOrRd',
                    cbar_kws={'label': 'Word Count'}, ax=ax)
        ax.set_title('Document-Term Matrix: Word Counts per Document',
                     fontsize=12, fontweight='bold', pad=15)
        ax.set_xlabel('Vocabulary Words', fontsize=10)
        ax.set_ylabel('Documents', fontsize=10)

        plt.tight_layout()
        plt.savefig('bow_matrix_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()

        print_success("bow_matrix_heatmap.png created successfully")
        return True

    except Exception as e:
        print_error(f"Failed to generate bow_matrix_heatmap.png: {str(e)}")
        return False


def generate_bow_vs_tfidf():
    """Generate BoW vs TF-IDF comparison."""
    print_progress("Generating bow_vs_tfidf.png...")

    try:
        from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

        # Corpus
        corpus = [
            "The movie was excellent and entertaining",
            "Great movie with excellent acting",
            "The film was boring and disappointing",
            "Terrible movie with poor acting",
            "Excellent film with great story and acting"
        ]

        # Vectorizers
        bow_vectorizer = CountVectorizer(lowercase=True, token_pattern=r'\b\w+\b')
        tfidf_vectorizer = TfidfVectorizer(lowercase=True, token_pattern=r'\b\w+\b')

        X_bow = bow_vectorizer.fit_transform(corpus)
        X_tfidf = tfidf_vectorizer.fit_transform(corpus)

        vocabulary = tfidf_vectorizer.get_feature_names_out()

        # Document 1 comparison
        doc_idx = 0
        doc_bow = X_bow[doc_idx].toarray().flatten()
        doc_tfidf = X_tfidf[doc_idx].toarray().flatten()

        # Create comparison DataFrame
        comparison_df = pd.DataFrame({
            'Word': vocabulary,
            'BoW Count': doc_bow,
            'TF-IDF Weight': doc_tfidf
        })
        comparison_df = comparison_df[comparison_df['BoW Count'] > 0].sort_values(
            'TF-IDF Weight', ascending=False
        )

        # Visualize
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(MAX_WIDTH, 4))

        # BoW plot
        words_in_doc = comparison_df['Word'].values
        bow_values = comparison_df['BoW Count'].values
        ax1.barh(words_in_doc, bow_values, color=COLORS['blue'])
        ax1.set_xlabel('Count', fontsize=10)
        ax1.set_title('Bag-of-Words: Raw Counts', fontsize=11, fontweight='bold')
        ax1.invert_yaxis()
        for i, v in enumerate(bow_values):
            ax1.text(v + 0.05, i, str(int(v)), va='center', fontsize=9)

        # TF-IDF plot
        tfidf_values = comparison_df['TF-IDF Weight'].values
        ax2.barh(words_in_doc, tfidf_values, color=COLORS['red'])
        ax2.set_xlabel('TF-IDF Weight', fontsize=10)
        ax2.set_title('TF-IDF: Weighted by Rarity', fontsize=11, fontweight='bold')
        ax2.invert_yaxis()
        for i, v in enumerate(tfidf_values):
            ax2.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)

        plt.suptitle('Document 1: "The movie was excellent and entertaining"',
                     fontsize=11, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('bow_vs_tfidf.png', dpi=150, bbox_inches='tight')
        plt.close()

        print_success("bow_vs_tfidf.png created successfully")
        return True

    except Exception as e:
        print_error(f"Failed to generate bow_vs_tfidf.png: {str(e)}")
        return False


def generate_ngram_extraction():
    """Generate n-gram extraction visualization."""
    print_progress("Generating ngram_extraction.png...")

    try:
        fig, ax = plt.subplots(figsize=(MAX_WIDTH, 7))
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 10)
        ax.axis('off')

        sentence = "The movie was not good"
        words = sentence.split()

        # Title
        ax.text(7, 9.5, "N-Gram Extraction: Capturing Word Order",
                fontsize=14, fontweight='bold', ha='center')

        # Original sentence
        y_pos = 8.5
        ax.text(1, y_pos, "Original:", fontsize=11, fontweight='bold')
        for i, word in enumerate(words):
            ax.add_patch(FancyBboxPatch((2 + i*2, y_pos-0.4), 1.8, 0.6,
                                        boxstyle="round,pad=0.05",
                                        edgecolor='black', facecolor='lightblue'))
            ax.text(2.9 + i*2, y_pos-0.1, word, ha='center', fontsize=9)

        # Unigrams
        y_pos = 6.5
        ax.text(1, y_pos, "Unigrams:", fontsize=11, fontweight='bold')
        for i, word in enumerate(words):
            ax.add_patch(FancyBboxPatch((2 + i*2, y_pos-0.4), 1.8, 0.6,
                                        boxstyle="round,pad=0.05",
                                        edgecolor=COLORS['green'],
                                        facecolor='lightgreen', linewidth=2))
            ax.text(2.9 + i*2, y_pos-0.1, word, ha='center', fontsize=8)

        # Bigrams
        y_pos = 4.5
        ax.text(1, y_pos, "Bigrams:", fontsize=11, fontweight='bold')
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        for i, bigram in enumerate(bigrams):
            width = 3.6
            ax.add_patch(FancyBboxPatch((2 + i*2.2, y_pos-0.4), width, 0.6,
                                        boxstyle="round,pad=0.05",
                                        edgecolor=COLORS['orange'],
                                        facecolor='lightyellow', linewidth=2))
            ax.text(2 + i*2.2 + width/2, y_pos-0.1, bigram, ha='center', fontsize=8)

        # Highlight "not good"
        ax.add_patch(FancyBboxPatch((8.4, y_pos-0.5), 3.6, 0.8,
                                    boxstyle="round,pad=0.1",
                                    edgecolor=COLORS['red'],
                                    facecolor='none', linewidth=3))
        ax.text(10.2, y_pos-1.2, '"not good" captured!', ha='center',
                fontsize=10, color=COLORS['red'], fontweight='bold')

        # Trigrams
        y_pos = 2.5
        ax.text(1, y_pos, "Trigrams:", fontsize=11, fontweight='bold')
        trigrams = [f"{words[i]} {words[i+1]} {words[i+2]}" for i in range(len(words)-2)]
        for i, trigram in enumerate(trigrams):
            width = 5.2
            ax.add_patch(FancyBboxPatch((2 + i*2.5, y_pos-0.4), width, 0.6,
                                        boxstyle="round,pad=0.05",
                                        edgecolor=COLORS['purple'],
                                        facecolor='lavender', linewidth=2))
            ax.text(2 + i*2.5 + width/2, y_pos-0.1, trigram, ha='center', fontsize=7)

        # Annotation
        ax.text(7, 0.8, "Bigrams capture local negation patterns that unigrams miss",
                fontsize=10, ha='center', style='italic', color='darkred')

        plt.tight_layout()
        plt.savefig('ngram_extraction.png', dpi=150, bbox_inches='tight')
        plt.close()

        print_success("ngram_extraction.png created successfully")
        return True

    except Exception as e:
        print_error(f"Failed to generate ngram_extraction.png: {str(e)}")
        return False


def generate_similarity_comparison():
    """Generate BoW vs TF-IDF similarity comparison heatmaps."""
    print_progress("Generating similarity_comparison.png...")

    try:
        from sklearn.datasets import fetch_20newsgroups
        from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        # Load dataset (3 categories)
        categories = ['sci.space', 'rec.sport.baseball', 'comp.graphics']
        newsgroups = fetch_20newsgroups(subset='all', categories=categories,
                                        remove=('headers', 'footers', 'quotes'),
                                        random_state=42)

        X = newsgroups.data
        y = newsgroups.target

        print_progress(f"  Loaded {len(X)} documents from {len(categories)} categories")

        # Vectorize with BoW and TF-IDF
        bow_vectorizer = CountVectorizer(max_features=5000, stop_words='english')
        tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

        print_progress("  Computing BoW vectors...")
        X_bow = bow_vectorizer.fit_transform(X)

        print_progress("  Computing TF-IDF vectors...")
        X_tfidf = tfidf_vectorizer.fit_transform(X)

        print_progress("  Computing similarity matrices...")
        cosine_sim_bow = cosine_similarity(X_bow)
        cosine_sim_tfidf = cosine_similarity(X_tfidf)

        # Visualize similarity heatmap for 20 random documents
        random_indices = np.random.RandomState(42).choice(len(X), 20, replace=False)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(MAX_WIDTH, 4))

        sns.heatmap(cosine_sim_bow[random_indices][:, random_indices],
                    ax=ax1, cmap='YlOrRd', cbar_kws={'label': 'Similarity'},
                    xticklabels=False, yticklabels=False)
        ax1.set_title('BoW Cosine Similarity', fontsize=11, fontweight='bold')
        ax1.set_xlabel('Document Index', fontsize=9)
        ax1.set_ylabel('Document Index', fontsize=9)

        sns.heatmap(cosine_sim_tfidf[random_indices][:, random_indices],
                    ax=ax2, cmap='YlOrRd', cbar_kws={'label': 'Similarity'},
                    xticklabels=False, yticklabels=False)
        ax2.set_title('TF-IDF Cosine Similarity', fontsize=11, fontweight='bold')
        ax2.set_xlabel('Document Index', fontsize=9)
        ax2.set_ylabel('Document Index', fontsize=9)

        plt.tight_layout()
        plt.savefig('similarity_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()

        print_success("similarity_comparison.png created successfully")
        return True

    except Exception as e:
        print_error(f"Failed to generate similarity_comparison.png: {str(e)}")
        return False


def main():
    """Main function to generate all diagrams."""
    print("\n" + "="*80)
    print("CLASSICAL NLP DIAGRAM GENERATION")
    print("="*80)
    print(f"Generating diagrams in current directory")
    print(f"Max width: 800px ({MAX_WIDTH:.2f} inches at 150 DPI)")
    print("="*80 + "\n")

    results = {}

    # Generate all diagrams
    diagrams = [
        ("preprocessing_pipeline.png", generate_preprocessing_pipeline),
        ("bow_matrix_heatmap.png", generate_bow_matrix_heatmap),
        ("bow_vs_tfidf.png", generate_bow_vs_tfidf),
        ("ngram_extraction.png", generate_ngram_extraction),
        ("similarity_comparison.png", generate_similarity_comparison),
    ]

    for diagram_name, generator_func in diagrams:
        try:
            success = generator_func()
            results[diagram_name] = success
        except Exception as e:
            print_error(f"Unexpected error generating {diagram_name}: {str(e)}")
            results[diagram_name] = False

    # Summary
    print("\n" + "="*80)
    print("GENERATION SUMMARY")
    print("="*80)

    successful = sum(1 for success in results.values() if success)
    total = len(results)

    for diagram_name, success in results.items():
        status = "✓" if success else "✗"
        print(f"{status} {diagram_name}")

    print("="*80)
    print(f"Successfully generated {successful}/{total} diagrams")
    print("="*80 + "\n")

    if successful == total:
        print_success("All diagrams generated successfully!")
        return 0
    else:
        print_error(f"Failed to generate {total - successful} diagram(s)")
        return 1


if __name__ == "__main__":
    import os
    import sys

    # Ensure we're in the right directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    sys.exit(main())
