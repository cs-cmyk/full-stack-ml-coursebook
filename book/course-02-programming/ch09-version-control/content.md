> **© 2026 Chirag Shinde. Licensed under CC BY-NC-SA 4.0.**
> See [LICENSE](../../LICENSE) for details.

---

# Chapter 9: Version Control with Git

## Why This Matters

Imagine spending weeks perfecting a machine learning model, only to accidentally overwrite the best version while experimenting with new hyperparameters. Or picture collaborating on a data analysis project where three teammates simultaneously edit the same preprocessing script, creating chaos. Version control with Git prevents these disasters and transforms how one works—tracking every experiment, enabling safe exploration, and turning code history into a searchable lab notebook. Whether working solo or on a team, Git is the professional data scientist's safety net and portfolio builder rolled into one indispensable tool.

## Intuition

Think of version control like writing a research paper with collaborators, but without the nightmare of `paper_draft1.docx`, `paper_draft2_john_edits.docx`, `paper_draft3_final.docx`, and `paper_draft3_final_ACTUALLY_FINAL.docx`. Every time someone makes a meaningful change, they create a labeled snapshot with a message: "Added results section" or "Fixed methods typo." The exact changes are visible—what changed, when, and by whom. If John's edits break something, the previous snapshot is just one command away.

Git works the same way for code. Instead of saving multiple copies of files with increasingly desperate names, a single set of files exists and snapshots called **commits** are taken whenever a meaningful milestone is reached. Each commit is like a save point in a video game—a moment to return to if the next experiment fails.

But Git goes further. It introduces **branches**, which are like parallel universes for a project. Want to try a risky new algorithm without breaking working code? Create a branch, experiment freely, and if it works, merge those changes back into the main timeline. If it fails, just delete the branch and return to the stable version. No mess, no lost work, no fear.

Here's a concrete analogy: imagine maintaining a collaborative cookbook with friends. The main branch is the official cookbook with tested recipes. When someone wants to add a new chocolate cake recipe, they create a feature branch to experiment. They test the recipe, adjust ingredients, and commit changes along the way. When satisfied, they submit a pull request: "I tested this chocolate cake recipe—please review and add to the main cookbook." Friends taste-test (code review), suggest improvements, and once approved, the recipe merges into the official book. The entire history is preserved: who added which recipes, when, and why. If a recipe turns out badly, tracing it back and fixing it is straightforward.

This is exactly how data science teams collaborate on models, pipelines, and analyses. One codebase, many experiments, clear history, fearless innovation.

## Formal Definition

**Version control** is a system that records changes to files over time, enabling recall of specific versions later. **Git** is a distributed version control system that tracks changes through a directed acyclic graph (DAG) of **commits**—immutable snapshots of project state.

Key mathematical and technical concepts:

- **Repository (repo)**: A project directory tracked by Git, containing a hidden `.git/` folder that stores the complete history as a DAG structure.

- **Commit**: A node in the commit graph, identified by a unique SHA-1 hash (40 hexadecimal characters, e.g., `a3f5c2d8e1b4...`). Each commit contains:
  - A pointer to a tree object (snapshot of all files)
  - Pointers to parent commit(s)
  - Author and committer metadata
  - Commit message
  - Timestamp

- **Branch**: A lightweight movable pointer to a commit. Creating a branch is O(1) and requires minimal storage (just the pointer).

- **HEAD**: A special pointer indicating current position in the commit graph. Typically points to a branch name, which points to a commit.

- **Working Directory → Staging Area → Repository**: Git's three-stage architecture where:
  1. Working Directory contains current files
  2. Staging Area (index) holds changes marked for the next commit
  3. Repository (`.git/` folder) stores all committed snapshots

- **Remote**: A version of the repository hosted on a server (e.g., GitHub), enabling distributed collaboration and backup.

The Git workflow can be expressed as a state machine with transitions:

```
Working Directory --[git add]--> Staging Area --[git commit]--> Repository --[git push]--> Remote
```

> **Key Concept:** Git transforms project history into a time-traveling graph where every experiment is preserved, every change is traceable, and collaborative chaos becomes structured teamwork.

## Visualization

[DIAGRAM: Three-Stage Git Workflow]

```
┌─────────────────────┐
│  Working Directory  │  (Current files - modified)
│   • model.py *      │
│   • data.csv        │
└─────────────────────┘
         │
         │ git add model.py
         ↓
┌─────────────────────┐
│   Staging Area      │  (Changes marked for commit)
│   • model.py ✓      │
└─────────────────────┘
         │
         │ git commit -m "message"
         ↓
┌─────────────────────┐
│   Repository        │  (Permanent history)
│   • Commit abc123   │
│   • Commit def456   │
│   • Commit 789xyz   │
└─────────────────────┘
         │
         │ git push
         ↓
┌─────────────────────┐
│  Remote (GitHub)    │  (Cloud backup + collaboration)
│   • origin/main     │
└─────────────────────┘
```

**Caption:** Git's three-stage workflow: modify files in the working directory, stage logical changes, commit snapshots to permanent history, and push to remote for backup and collaboration.

[DIAGRAM: Branching and Merging Visualization]

```
        C1 ← C2 ← C3 ←──────────── C6 (main)
               ↖              ↗
                C4 ← C5 (experiment-rf)

Legend:
C1: "Initial commit"
C2: "Add baseline model"
C4, C5: "Experiment with Random Forest"
C6: "Merge experiment-rf into main"
```

**Caption:** Branches enable parallel development. The experiment branch diverges from main at C2, develops independently (C4, C5), then merges back (C6). Main remains stable throughout experimentation.

## Examples

### Part 1: Initial Setup and Configuration

```python
# Complete Git Workflow for Data Science - First Repository
# This example shows the entire workflow from initialization to analysis

# SETUP: First, configure Git (run once on your system)
# In terminal:
# git config --global user.name "Your Name"
# git config --global user.email "you@email.com"

# Step 1: Create project directory and initialize Git
# In terminal:
# mkdir ds-iris-project
# cd ds-iris-project
# git init
# Output: Initialized empty Git repository in /path/to/ds-iris-project/.git/
```

**Configuration (Lines 3-6):** Before using Git, configure identity once per system. These settings attach name and email to every commit, creating an audit trail. This is essential for collaboration—teammates will see who made each change.

**Initialization (Lines 8-13):** The `git init` command creates a hidden `.git/` folder in the project directory. This folder is Git's database—it stores every commit, branch, and piece of history. The directory transforms from a regular folder into a **repository**—a time-traveling project folder.

### Part 2: Creating .gitignore

```python
# Step 2: Create .gitignore FIRST (prevent committing unwanted files)
# Create a file called .gitignore with these contents:
"""
# Data files
data/
*.csv
*.pkl

# Python
__pycache__/
*.pyc
.env
venv/

# Jupyter
.ipynb_checkpoints/

# System
.DS_Store
"""

# Step 3: Stage and commit .gitignore
# git add .gitignore
# git commit -m "Add .gitignore for data science project"
# Output: [main (root-commit) a1b2c3d] Add .gitignore for data science project
#  1 file changed, 15 insertions(+)
#  create mode 100644 .gitignore
```

**Creating .gitignore (Lines 1-19):** This is the most important step for data scientists, yet often forgotten. The `.gitignore` file tells Git which files to never track. Data files (often large), Python cache files (regenerated automatically), environment folders (user-specific), and system files (irrelevant noise) are excluded. Committing large datasets makes repositories slow and bloated—document data sources in the README instead.

**First commit (Lines 21-26):** `.gitignore` is staged with `git add` and committed with a clear message. This creates the repository's first snapshot. Notice the output: `[main (root-commit) a1b2c3d]` indicates being on the main branch, this is the root (first) commit, and `a1b2c3d` is the shortened commit hash (unique identifier).

### Part 3: Creating the Initial Analysis

```python
# Step 4: Create initial analysis script
# Save this as analysis.py
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Set random seed for reproducibility
np.random.seed(42)

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

print(f"Dataset shape: {X.shape}")
print(f"Classes: {iris.target_names}")
# Output:
# Dataset shape: (150, 4)
# Classes: ['setosa' 'versicolor' 'virginica']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train baseline model
model = LogisticRegression(max_iter=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Baseline accuracy: {accuracy:.3f}")
# Output:
# Baseline accuracy: 0.978
```

**Creating the analysis (Lines 1-36):** A complete, runnable data science script loads the Iris dataset, splits it into train/test sets, trains a logistic regression model, and reports accuracy. Setting `random_state=42` everywhere ensures reproducibility—crucial for scientific work. The script runs independently without external data files, making it perfect for version control.

### Part 4: Status Check and First Meaningful Commit

```python
# Step 5: Check status - see untracked file
# git status
# Output:
# On branch main
# Untracked files:
#   (use "git add <file>..." to include in what will be committed)
#         analysis.py
# nothing added to commit but untracked files present

# Step 6: Stage and commit the analysis
# git add analysis.py
# git commit -m "Add baseline Iris classification with logistic regression"
# Output: [main f4e5d6c] Add baseline Iris classification with logistic regression
#  1 file changed, 25 insertions(+)
#  create mode 100644 analysis.py
```

**Checking status (Lines 1-8):** The `git status` command shows that `analysis.py` is **untracked** (Git sees it but isn't tracking changes yet). This state persists until the file is explicitly added. Running `git status` constantly is recommended, especially when learning.

**First meaningful commit (Lines 10-14):** The analysis is staged and committed. The commit message follows best practices: it's descriptive, uses imperative mood ("Add" not "Added"), and explains what the code does. Future readers will appreciate clear messages when searching through history for "when did I add that baseline model?"

### Part 5: Making Improvements

```python
# Step 7: Improve the model (add feature scaling)
# Edit analysis.py to add these lines after the split:
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# (Rest of code remains the same)
# New output:
# Baseline accuracy: 0.978  (same or slightly better)

# Step 8: See what changed
# git diff analysis.py
# Output shows:
# +from sklearn.preprocessing import StandardScaler
# +
# +scaler = StandardScaler()
# +X_train = scaler.fit_transform(X_train)
# +X_test = scaler.transform(X_test)

# Step 9: Commit the improvement
# git add analysis.py
# git commit -m "Add feature scaling to improve model generalization"
# Output: [main 7g8h9i0] Add feature scaling to improve model generalization
#  1 file changed, 5 insertions(+)
```

**Making improvements (Lines 1-10):** The model is enhanced by adding feature scaling via `StandardScaler`. This is a separate logical change from the baseline model, so it deserves its own commit. Small, focused commits are better than giant "updated everything" commits—they're easier to understand, review, and potentially revert.

**Viewing changes (Lines 12-19):** The `git diff` command shows exactly what changed before staging. The output uses `+` for added lines and `-` for deleted lines (not shown here). This allows reviewing work before committing, catching accidental changes or debugging statements.

**Second commit (Lines 21-25):** The scaling improvement is committed separately. The history now tells a story: first a baseline was built, then it was improved with feature scaling. This narrative structure helps collaborators (and future readers) understand the project's evolution.

### Part 6: Viewing History and Branching

```python
# Step 10: View commit history
# git log --oneline
# Output:
# 7g8h9i0 Add feature scaling to improve model generalization
# f4e5d6c Add baseline Iris classification with logistic regression
# a1b2c3d Add .gitignore for data science project

# Step 11: Create branch for experiment
# git checkout -b experiment-random-forest
# Output: Switched to a new branch 'experiment-random-forest'
```

**Viewing history (Lines 1-6):** The `git log --oneline` command shows a condensed commit history. Each line has the short hash, commit message, and branch markers. This is the project's lab notebook—every experiment documented with precise timestamps and descriptions.

**Branching for experiments (Lines 8-10):** The `git checkout -b experiment-random-forest` command creates a new branch and switches to it in one operation. Branches are lightweight (just pointers to commits), so creating them is instantaneous and costs essentially nothing. This branch enables experimentation without risking the working code on main.

### Part 7: Experimenting on Branch

```python
# Step 12: Try Random Forest (edit analysis.py)
from sklearn.ensemble import RandomForestClassifier

# Replace LogisticRegression with:
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
# New output:
# Baseline accuracy: 0.956  (slightly worse for this small dataset)

# Step 13: Commit on experiment branch
# git add analysis.py
# git commit -m "Experiment with Random Forest classifier"
# Output: [experiment-random-forest j1k2l3m] Experiment with Random Forest classifier

# Step 14: Switch back to main - logistic regression is still there!
# git checkout main
# cat analysis.py  # Shows LogisticRegression version
```

**Experimenting freely (Lines 1-8):** On the experiment branch, logistic regression is replaced with Random Forest. Anything can be tried here—the main branch remains untouched. If Random Forest performed much better, this branch would merge into main. Since it's slightly worse on this small dataset, the branch could be kept for documentation or deleted.

**Branch switching (Lines 14-16):** Switching back to main with `git checkout main` updates working directory files to match the main branch. The `analysis.py` file now shows the logistic regression version again. This is Git's magic—switching between branches instantly changes the entire project state.

### Part 8: Visualizing Branch Structure

```python
# Step 15: View branch structure
# git log --oneline --graph --all
# Output:
# * j1k2l3m (experiment-random-forest) Experiment with Random Forest classifier
# * 7g8h9i0 (HEAD -> main) Add feature scaling to improve model generalization
# * f4e5d6c Add baseline Iris classification with logistic regression
# * a1b2c3d Add .gitignore for data science project

# Congratulations! You've created your first data science repository with Git.
```

**Visualizing branches (Lines 1-8):** The `git log --graph` command draws an ASCII graph showing branch structure. The experiment branch diverging from main after the scaling commit is visible. The `(HEAD -> main)` notation shows HEAD points to main, which points to commit `7g8h9i0`.

This workflow demonstrates Git's core value: **fearless experimentation**. Risky changes can be tried, knowing that returning to working code is always possible. Project history becomes a searchable record of every decision, making data science more reproducible and collaborative.

## Common Pitfalls

**1. Committing Large Datasets**

Beginners often commit CSV files, model binaries, or large datasets directly to Git, causing repositories to balloon to gigabytes in size. Git tracks every version of every file—a 500MB dataset committed five times means 2.5GB in the repository, even if the file is later deleted.

**Why it happens:** Without understanding `.gitignore`, it's natural to think "commit everything so nothing is lost." The `git add .` command makes it easy to accidentally stage all files in the directory.

**What to do instead:** Create `.gitignore` at project start before the first commit. For data files, commit a small sample for documentation (e.g., `data/sample_100_rows.csv`) and document where to obtain the full dataset in the README. Use dedicated tools like DVC (Data Version Control) or Git LFS (Large File Storage) for truly version-controlling large datasets, but for most projects, simply exclude data files and document their sources.

**Recovery:** If a large file has already been committed, use `git rm --cached large_file.csv` to remove it from Git tracking while keeping it locally. Then add `large_file.csv` to `.gitignore` and commit these changes. Note: this doesn't remove the file from history—it only prevents future tracking. For complete removal from history (needed if the file contains secrets or makes cloning impossible), use `git filter-branch` or BFG Repo-Cleaner, but these rewrite history and complicate collaboration.

**2. Vague Commit Messages**

Messages like "update code," "fixed stuff," "changes," or "asdf" provide zero information about what changed or why. Six months later, when finding when a bug was introduced becomes necessary, these messages are useless.

**Why it happens:** Commit messages feel like busywork when focused on coding. It's tempting to use the quickest message just to move on. Some developers treat commits as personal notes rather than professional documentation.

**What to do instead:** Write commit messages for future readers and collaborators. Follow the imperative mood convention: "Add feature" (not "Added feature" or "Adds feature"). Include what changed and why:

- **Good:** "Add feature engineering for age binning to improve model performance"
- **Better:** "Fix memory leak in data preprocessing loop by releasing references after each batch"
- **Best:** A multi-line message where the first line (subject) summarizes in ~50 characters, followed by a blank line and detailed explanation of what, why, and any side effects.

**Pro tip:** If a clear commit message cannot be written, the commit probably does too many things. Split it into smaller, focused commits using `git add -p` to stage partial changes.

**3. Merge Conflicts Panic**

When Git can't automatically merge changes because two branches modified the same lines, it marks the file with conflict markers: `<<<<<<<`, `=======`, `>>>>>>>`. Many beginners see these cryptic markers and panic, thinking they've broken something permanently or lost work.

**Why it happens:** Merge conflicts feel technical and scary when first encountered. The strange syntax looks like corrupted code, and error messages don't always clearly explain how to resolve conflicts.

**What to do instead:** Recognize that conflicts are normal and easily resolved—Git is simply asking which version to keep. Here's the process:

1. Run `git status` to see which files have conflicts
2. Open conflicted files and look for conflict markers:
   ```python
   <<<<<<< HEAD
   model = LogisticRegression(max_iter=200)  # Your changes on current branch
   =======
   model = RandomForestClassifier(n_estimators=100)  # Incoming changes from other branch
   >>>>>>> experiment-rf
   ```
3. Decide which version to keep (or combine both):
   - Keep yours: delete everything except the `HEAD` section
   - Keep theirs: delete everything except the section after `=======`
   - **Keep both:** delete markers and combine the code logically
4. Remove all conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`)
5. Ensure the code runs correctly
6. Stage the resolved file: `git add filename.py`
7. Complete the merge: `git commit -m "Resolve merge conflict by combining both models"`

**Remember:** Conflict markers are just text annotations. The file can be edited normally, run, tested, and once satisfied, staged and committed to complete the merge. No data is lost—both versions are preserved in history.

**Prevention:** Communicate with teammates about who's working on which files. Smaller, more frequent commits reduce conflict size. For Jupyter notebooks, conflicts are especially messy (JSON structure)—consider clearing outputs before committing to minimize conflicts.

## Practice

**Practice 1**

Initialize a Git repository and practice the basic workflow with a simple data analysis.

1. Create a new directory called `my-first-repo/`
2. Navigate into it and run `git init` to initialize Git
3. Configure Git with name and email (if not already done):
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "you@email.com"
   ```
4. Create a `.gitignore` file with basic Python and data science exclusions:
   ```
   __pycache__/
   *.pyc
   .ipynb_checkpoints/
   data/
   *.csv
   .DS_Store
   ```
5. Stage and commit `.gitignore` with message "Add .gitignore for Python data science"
6. Create a Python file `explore.py` that loads the Iris dataset and prints its shape and the first 5 samples
7. Check `git status`—is `explore.py` tracked or untracked?
8. Stage `explore.py` and commit with message "Add Iris data exploration"
9. Modify `explore.py` to also print basic statistics using `.describe()` on a pandas DataFrame
10. View changes with `git diff`
11. Commit the modification with message "Add statistical summary to exploration"
12. View commit history with `git log --oneline`

**Expected output:** Three commits should appear in the log. Running `git show` should display the most recent changes.

**Success criteria:** Repository contains clean commit history with descriptive messages, `.gitignore` prevents tracking unwanted files, and each Git command's purpose is clear.

---

**Practice 2**

Practice using branches to safely experiment with different machine learning algorithms, experiencing Git's true power for data science work.

**Setup:** Start with a repository containing a baseline classification model.

1. Create a new repository and initialize Git
2. Create `model.py` with a baseline logistic regression model for the Breast Cancer dataset:
   ```python
   from sklearn.datasets import load_breast_cancer
   from sklearn.model_selection import train_test_split
   from sklearn.linear_model import LogisticRegression
   from sklearn.metrics import accuracy_score
   import numpy as np

   np.random.seed(42)

   # Load data
   data = load_breast_cancer()
   X_train, X_test, y_train, y_test = train_test_split(
       data.data, data.target, test_size=0.3, random_state=42
   )

   # Baseline model
   model = LogisticRegression(max_iter=10000, random_state=42)
   model.fit(X_train, y_train)
   accuracy = accuracy_score(y_test, model.predict(X_test))
   print(f"Accuracy: {accuracy:.4f}")
   ```
3. Commit this baseline with message "Add logistic regression baseline"
4. Run the script and note the accuracy in a comment
5. Create a branch called `experiment-svm`: `git checkout -b experiment-svm`
6. Modify `model.py` to use `SVC` from `sklearn.svm` instead of LogisticRegression
7. Commit on the SVM branch: "Experiment with SVM classifier"
8. Run the script and note the SVM accuracy
9. Switch back to main: `git checkout main`
10. Verify that `model.py` shows logistic regression again (not SVM)
11. Create another branch: `git checkout -b experiment-random-forest`
12. Modify `model.py` to use `RandomForestClassifier`
13. Commit: "Experiment with Random Forest classifier"
14. Note the Random Forest accuracy
15. Compare all three accuracies—which model is best?
16. Merge the best-performing model into main:
    ```bash
    git checkout main
    git merge experiment-<best-model>
    ```
17. View the branch structure: `git log --graph --all --oneline`
18. Delete the experiment branches not needed: `git branch -d experiment-<name>`

**Bonus challenges:**
- Create a branch for hyperparameter tuning on the best model
- Try creating a merge conflict intentionally (modify same lines on two branches) and practice resolving it
- Use `git diff main..experiment-svm` to compare branches without switching

**Expected outcome:** Experience how branches enable risk-free experimentation. Main branch always contains working code while exploration happens freely on feature branches.

---

**Practice 3**

Build a portfolio-ready data science project with professional Git workflow, proper documentation, and GitHub hosting. This exercise simulates real-world professional development.

**Project:** Titanic survival prediction with complete exploratory data analysis, feature engineering, and modeling.

**Requirements:**

**Part 1: Local Setup**
1. Create project directory structure:
   ```
   titanic-survival-prediction/
   ├── data/
   ├── notebooks/
   ├── src/
   ├── models/
   ├── .gitignore
   ├── README.md
   └── requirements.txt
   ```
2. Initialize Git repository
3. Create comprehensive `.gitignore`:
   ```
   # Data
   data/
   *.csv

   # Models
   models/
   *.pkl

   # Python
   __pycache__/
   *.pyc
   .env
   venv/

   # Jupyter
   .ipynb_checkpoints/
   *.ipynb

   # System
   .DS_Store
   ```
4. Create `README.md` with:
   - Project title and description
   - Problem statement
   - Dataset source (Seaborn's Titanic dataset)
   - Setup instructions
   - (Results will be added later)
5. Create `requirements.txt`:
   ```
   numpy==1.24.3
   pandas==2.0.3
   scikit-learn==1.3.0
   seaborn==0.12.2
   matplotlib==3.7.2
   ```
6. Commit these foundational files: "Initialize project structure and documentation"

**Part 2: Development with Branches**
1. Create and switch to branch `feature-eda`:
   ```bash
   git checkout -b feature-eda
   ```
2. In `notebooks/`, create `01_exploratory_analysis.ipynb` with:
   - Load Titanic data from seaborn: `sns.load_dataset('titanic')`
   - Basic statistics and visualizations
   - Identification of missing values
   - Initial insights about survival patterns
3. **Important:** Clear all cell outputs before committing (Cell → All Output → Clear)
4. Commit: "Add exploratory data analysis notebook"
5. Merge back to main:
   ```bash
   git checkout main
   git merge feature-eda
   ```
6. Create branch `feature-engineering`:
   ```bash
   git checkout -b feature-engineering
   ```
7. Create `src/features.py` with functions to:
   - Handle missing values
   - Create age bins
   - Encode categorical variables
   - Create family size feature
8. Commit: "Add feature engineering functions"
9. Merge to main and delete branch:
   ```bash
   git checkout main
   git merge feature-engineering
   git branch -d feature-engineering
   ```
10. Create branch `model-training`:
    ```bash
    git checkout -b model-training
    ```
11. Create `src/train.py` that:
    - Loads and processes data using feature functions
    - Trains multiple models (LogisticRegression, RandomForest, GradientBoosting)
    - Compares performance with cross-validation
    - Saves best model (but this file is in .gitignore)
12. Commit: "Add model training pipeline with cross-validation"
13. Merge to main

**Part 3: GitHub Integration**
1. Create a new public repository on GitHub (don't initialize with README)
2. Connect the local repository to GitHub:
   ```bash
   git remote add origin https://github.com/yourusername/titanic-survival-prediction.git
   ```
3. Push the main branch:
   ```bash
   git push -u origin main
   ```
4. Verify on GitHub that:
   - README renders correctly
   - Code is browsable
   - `.gitignore` prevents data/ and models/ from being tracked
   - Commit history is visible
5. Update README with:
   - Model performance results
   - Sample predictions
   - Any visualizations (commit images as PNGs)
6. Commit and push:
   ```bash
   git add README.md
   git commit -m "Update README with model performance results"
   git push
   ```

**Part 4: Collaborative Workflow Practice**
1. Create a branch `improve-accuracy`:
   ```bash
   git checkout -b improve-accuracy
   ```
2. Make improvements (e.g., better feature engineering, hyperparameter tuning)
3. Commit changes
4. Push the branch to GitHub:
   ```bash
   git push -u origin improve-accuracy
   ```
5. On GitHub, create a Pull Request from `improve-accuracy` to `main`
6. Write a clear PR description explaining what changed and why
7. Review the PR (in professional setting, teammates would review)
8. Merge the PR on GitHub
9. Pull the changes back to local main:
   ```bash
   git checkout main
   git pull origin main
   ```
10. Delete the local and remote branches:
    ```bash
    git branch -d improve-accuracy
    git push origin --delete improve-accuracy
    ```

**Deliverables:**
- GitHub repository URL with:
  - ✅ Professional README with clear problem statement, setup instructions, and results
  - ✅ Clean commit history (minimum 8 meaningful commits)
  - ✅ Proper `.gitignore` (data and models excluded)
  - ✅ Working code that others can clone and run
  - ✅ Evidence of branching workflow (visible in commit graph)
  - ✅ At least one merged Pull Request

**Challenge questions for reflection:**
1. How would handling a large dataset (1GB+) that can't be excluded work? Research Git LFS and DVC.
2. If a teammate pushed conflicting changes to the same feature engineering function, how would resolution work?
3. What CI/CD tools could automate testing the model pipeline on every commit? Research GitHub Actions.
4. How could Git tags mark model versions deployed to production?

**Success criteria:** Repository looks professional enough to include on a resume or show to potential employers. Clean history tells the story of the development process.

## Solutions

**Solution 1**

```python
# Create directory and initialize
# In terminal:
# mkdir my-first-repo
# cd my-first-repo
# git init

# Configure Git
# git config --global user.name "Your Name"
# git config --global user.email "you@email.com"

# Create .gitignore
# Create file named .gitignore with:
"""
__pycache__/
*.pyc
.ipynb_checkpoints/
data/
*.csv
.DS_Store
"""

# Commit .gitignore
# git add .gitignore
# git commit -m "Add .gitignore for Python data science"

# Create explore.py
import numpy as np
from sklearn.datasets import load_iris
import pandas as pd

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

print(f"Dataset shape: {X.shape}")
print(f"First 5 samples:\n{X[:5]}")

# Check status
# git status
# Output: Untracked files: explore.py

# Stage and commit
# git add explore.py
# git commit -m "Add Iris data exploration"

# Modify explore.py to add statistics
df = pd.DataFrame(X, columns=iris.feature_names)
print(f"\nStatistical summary:\n{df.describe()}")

# View changes
# git diff

# Commit modification
# git add explore.py
# git commit -m "Add statistical summary to exploration"

# View history
# git log --oneline
# Output:
# abc123 Add statistical summary to exploration
# def456 Add Iris data exploration
# ghi789 Add .gitignore for Python data science
```

**Approach:** Start with `.gitignore` to prevent tracking unwanted files. Create small, focused commits with descriptive messages. Use `git status` and `git diff` frequently to understand current state before committing.

**Solution 2**

```python
# Initialize repository
# mkdir model-experiments
# cd model-experiments
# git init

# Create model.py with baseline
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

np.random.seed(42)

# Load data
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.3, random_state=42
)

# Baseline model
model = LogisticRegression(max_iter=10000, random_state=42)
model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Accuracy: {accuracy:.4f}")
# Output: Accuracy: 0.9766

# Commit baseline
# git add model.py
# git commit -m "Add logistic regression baseline"

# Create SVM branch and experiment
# git checkout -b experiment-svm

# Modify model.py - replace LogisticRegression with:
from sklearn.svm import SVC

model = SVC(random_state=42)
model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Accuracy: {accuracy:.4f}")
# Output: Accuracy: 0.9708 (slightly worse)

# Commit on SVM branch
# git add model.py
# git commit -m "Experiment with SVM classifier"

# Switch back to main
# git checkout main
# Verify model.py shows LogisticRegression

# Create Random Forest branch
# git checkout -b experiment-random-forest

# Modify model.py - replace with:
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Accuracy: {accuracy:.4f}")
# Output: Accuracy: 0.9649 (worst of three)

# Commit on RF branch
# git add model.py
# git commit -m "Experiment with Random Forest classifier"

# LogisticRegression performed best (0.9766), so merge it
# Already on main, which has the best model

# View branch structure
# git log --graph --all --oneline

# Delete experiment branches (optional)
# git branch -d experiment-svm
# git branch -d experiment-random-forest
```

**Approach:** Use branches to isolate each experiment. Main branch always contains working code. After testing all three models, the best-performing version (Logistic Regression) remains on main. Experiment branches can be kept for documentation or deleted. The key insight: branches enable parallel exploration without risk.

**Solution 3**

**Part 1: Local Setup**

```bash
# Create directory structure
mkdir titanic-survival-prediction
cd titanic-survival-prediction
mkdir data notebooks src models

# Initialize Git
git init

# Create .gitignore
cat > .gitignore << 'EOF'
# Data
data/
*.csv

# Models
models/
*.pkl

# Python
__pycache__/
*.pyc
.env
venv/

# Jupyter
.ipynb_checkpoints/
*.ipynb

# System
.DS_Store
EOF

# Create README.md
cat > README.md << 'EOF'
# Titanic Survival Prediction

## Problem Statement
Predict passenger survival on the Titanic using machine learning classification algorithms.

## Dataset
- Source: Seaborn's built-in Titanic dataset
- Load with: `sns.load_dataset('titanic')`

## Setup
```bash
pip install -r requirements.txt
```

## Results
(To be added after model training)
EOF

# Create requirements.txt
cat > requirements.txt << 'EOF'
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
seaborn==0.12.2
matplotlib==3.7.2
EOF

# Commit foundation
git add .gitignore README.md requirements.txt
git commit -m "Initialize project structure and documentation"
```

**Part 2: Development with Branches**

```bash
# Create EDA branch
git checkout -b feature-eda

# Create notebook (simplified content shown)
# In notebooks/01_exploratory_analysis.ipynb:
```

```python
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Load data
titanic = sns.load_dataset('titanic')

# Basic exploration
print(titanic.shape)
print(titanic.info())
print(titanic.describe())

# Survival rate
print(f"Survival rate: {titanic['survived'].mean():.2%}")

# Visualizations
sns.countplot(data=titanic, x='survived', hue='sex')
plt.title('Survival by Gender')
plt.show()

# Missing values
print(titanic.isnull().sum())
```

```bash
# Clear all outputs in Jupyter before committing!
# Commit EDA
git add notebooks/01_exploratory_analysis.ipynb
git commit -m "Add exploratory data analysis notebook"
git checkout main
git merge feature-eda

# Create feature engineering branch
git checkout -b feature-engineering
```

**src/features.py:**

```python
import pandas as pd
import numpy as np

def handle_missing_values(df):
    """Fill missing values appropriately."""
    df = df.copy()
    df['age'].fillna(df['age'].median(), inplace=True)
    df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)
    df.drop('deck', axis=1, inplace=True)  # Too many missing
    return df

def create_age_bins(df):
    """Create age categories."""
    df = df.copy()
    df['age_group'] = pd.cut(df['age'], bins=[0, 12, 18, 35, 60, 100],
                              labels=['child', 'teen', 'adult', 'middle', 'senior'])
    return df

def encode_categorical(df):
    """One-hot encode categorical variables."""
    df = df.copy()
    df = pd.get_dummies(df, columns=['sex', 'embarked', 'class'], drop_first=True)
    return df

def create_family_size(df):
    """Create family size feature."""
    df = df.copy()
    df['family_size'] = df['sibsp'] + df['parch'] + 1
    return df
```

```bash
# Commit feature engineering
git add src/features.py
git commit -m "Add feature engineering functions"
git checkout main
git merge feature-engineering
git branch -d feature-engineering

# Create model training branch
git checkout -b model-training
```

**src/train.py:**

```python
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from features import *

# Load data
titanic = sns.load_dataset('titanic')

# Feature engineering
titanic = handle_missing_values(titanic)
titanic = create_family_size(titanic)
titanic = create_age_bins(titanic)
titanic = encode_categorical(titanic)

# Select features
feature_cols = ['pclass', 'age', 'fare', 'family_size', 'sex_male']
X = titanic[feature_cols]
y = titanic['survived']

# Compare models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5)
    print(f"{name}: {scores.mean():.4f} (+/- {scores.std():.4f})")

# Output:
# Logistic Regression: 0.7845 (+/- 0.0283)
# Random Forest: 0.8113 (+/- 0.0245)
# Gradient Boosting: 0.8203 (+/- 0.0312)
```

```bash
# Commit model training
git add src/train.py
git commit -m "Add model training pipeline with cross-validation"
git checkout main
git merge model-training
git branch -d model-training
```

**Part 3 & 4: GitHub Integration**

```bash
# Create GitHub repo first via web interface, then:
git remote add origin https://github.com/yourusername/titanic-survival-prediction.git
git push -u origin main

# Update README with results
# (Edit README.md to add results section)
git add README.md
git commit -m "Update README with model performance results"
git push

# Create improvement branch
git checkout -b improve-accuracy
# (Make improvements - e.g., add polynomial features, tune hyperparameters)
git add src/train.py
git commit -m "Add hyperparameter tuning for GradientBoosting"
git push -u origin improve-accuracy

# Create PR on GitHub, review, merge
# Then pull changes
git checkout main
git pull origin main
git branch -d improve-accuracy
git push origin --delete improve-accuracy
```

**Approach:** Professional Git workflow mirrors industry practices: feature branches for isolated development, clear commit messages, proper `.gitignore`, comprehensive README, and GitHub integration. The repository becomes a portfolio piece demonstrating both technical skills and professional development practices.

**Challenge Answers:**
1. **Large datasets:** Use Git LFS for files up to ~5GB, or DVC for larger data with proper versioning
2. **Merge conflicts:** Use `git status` to identify conflicts, manually edit files to resolve, test code, then `git add` and `git commit`
3. **CI/CD:** GitHub Actions can run `pytest` on every commit, ensuring code quality
4. **Git tags:** Use `git tag v1.0.0` to mark production releases, enabling easy rollback

## Key Takeaways

- **Git is a safety net and lab notebook:** Every experiment is preserved, every change is traceable, and returning to working code is always possible. Use it from day one, even for solo projects.

- **The three-stage workflow is fundamental:** Working Directory → Staging Area → Repository. Stage logical units with `git add`, commit snapshots with clear messages using `git commit`, and push to remote with `git push` for backup and collaboration.

- **Branches enable fearless experimentation:** Create branches freely to try risky changes without touching stable code on main. Merge successful experiments, delete failures, and always keep main working.

- **What to commit matters for data science:** Commit code (`.py`), notebooks (with cleared outputs), documentation, and small reference data. Never commit large datasets, model binaries, credentials, or system files—use `.gitignore` to prevent this.

- **GitHub is a data science portfolio:** Every well-documented project with clean commit history demonstrates professional skills to potential employers. Make repositories public, write clear READMEs, and showcase end-to-end workflows.

- **Merge conflicts are normal, not disasters:** When Git shows conflict markers, it's asking which changes to keep. Edit the file to remove markers and combine code logically, then stage and commit to complete the merge.

- **Git commands are the daily toolkit:** Master these essentials and run `git status` constantly:
  ```bash
  git status              # Check current state (run this constantly)
  git add filename        # Stage specific file
  git commit -m "message" # Commit with descriptive message
  git log --oneline       # View commit history
  git diff                # See unstaged changes
  git branch              # List branches
  git checkout -b name    # Create and switch to new branch
  git merge branch-name   # Merge branch into current branch
  git push origin main    # Push commits to GitHub
  git pull origin main    # Pull updates from GitHub
  ```

---

## Glossary Additions

The following terms from this chapter should be added to the main glossary:

**Branch** — A lightweight movable pointer to a commit in Git, enabling parallel lines of development. Creating a branch is O(1) and allows safe experimentation without affecting the main codebase. First introduced in Section 9.

**Commit** — An immutable snapshot of a project at a point in time, identified by a unique SHA-1 hash. Each commit contains changes, metadata (author, timestamp), parent pointers, and a message. First introduced in Section 9.

**Git** — A distributed version control system that tracks changes to files through a directed acyclic graph of commits, enabling collaboration, experimentation, and complete project history. First introduced in Section 9.

**HEAD** — A special pointer in Git indicating current position in the commit graph. Typically points to a branch name, which points to a commit. In detached HEAD state, points directly to a commit. First introduced in Section 9.

**Merge conflict** — A situation where Git cannot automatically combine changes because two branches modified the same lines. Git marks the file with conflict markers (`<<<<<<<`, `=======`, `>>>>>>>>`), requiring manual resolution. First introduced in Section 9.

**Remote** — A version of a Git repository hosted on a server (e.g., GitHub, GitLab), enabling distributed collaboration, backup, and portfolio sharing. Typically named "origin" by convention. First introduced in Section 9.

**Repository (repo)** — A project directory tracked by Git, containing a hidden `.git/` folder that stores complete commit history, branches, and configuration. Can be local (on a computer) or remote (on a server). First introduced in Section 9.

**Staging area (index)** — The middle stage in Git's three-stage architecture where changes are marked for the next commit. Allows committing logical units rather than all changes at once. First introduced in Section 9.

**Version control** — A system that records changes to files over time, enabling recall of specific versions, tracking who changed what and when, collaborating safely, and recovering from mistakes. First introduced in Section 9.

**Working directory** — The current state of project files on disk, including any modifications made since the last commit. The first stage in Git's three-stage workflow. First introduced in Section 9.

**Next:** Chapter 10 covers debugging strategies and tools for efficiently identifying and fixing errors in data science code.
