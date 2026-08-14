# Meghtithi_Mitra_DMML_Assignment
# DMML_Assignments
DMML Assignments: Clustering and Semi-Supervised Learning

## Author

**Name:** Meghtithi Mitra
**Roll Number:** MDS202520

---

## Repository Contents

| Notebook | Topic |
|---|---|
| `Meghtithi_Assignment_2.ipynb` | K-Means Clustering with Jaccard Similarity |
| `Meghtithi_Assignment_3.ipynb` | Semi-Supervised Learning (Clustering + Classification) |

---

## Assignment 2: K-Means Clustering with Jaccard Similarity

### Objective
Implements K-Means clustering on three text document collections from the UCI Bag-of-Words repository — **KOS** (3,430 political blog posts), **NIPS** (1,500 ML research papers), and **Enron** (39,861 corporate emails) — using **Jaccard similarity** instead of Euclidean distance, since the data is high-dimensional and sparse.

### Approach
1. **Load & convert** — parse the `(docID, wordID, count)` UCI Bag-of-Words format into sparse binary matrices.
2. **EDA** — examine document length distributions, vocabulary size, and sparsity per dataset.
3. **Jaccard distance** — implement Jaccard similarity/distance for sparse binary vectors.
4. **K-Means with Jaccard distance** — custom implementation (standard random initialization for KOS and NIPS).
5. **Elbow method** — sweep K ∈ {2, 3, 5, 7, 10} and track inertia to select optimal K per dataset.
6. **K-Means++ for Enron** — random initialization produced non-monotonic inertia curves on Enron due to extreme sparsity (99.67%); switched to K-Means++ initialization to fix this.
7. **Cluster analysis** — inspect cluster sizes and top words per cluster to interpret themes.

### Results

| Dataset | Docs | Sparsity | Optimal K |
|---|---|---|---|
| KOS | 3,430 | 98.51% | 5 |
| NIPS | 1,500 | 95.99% | 7 |
| Enron | 39,861 | 99.67% | — (K-Means++) |

### Technologies Used
* Python (NumPy, SciPy sparse matrices, Matplotlib)
* Google Colab / Google Drive
* UCI Bag-of-Words dataset

---

## Assignment 3: Semi-Supervised Learning

### Objective
Tests whether smart selection of a *small* labeled subset — via clustering — can approach fully-labeled classifier performance, across two datasets and two classifiers.

### Part 1: Fashion MNIST — K-Means + MLP
1. Run K-Means on unlabeled, flattened (784-dim) image data for K ∈ {10, 20, 30, 50, 75, 100}.
2. For each cluster, select the single image closest to the centroid as the representative sample.
3. Train an MLP (Flatten → Dense(300, ReLU) → Dense(100, ReLU) → Dense(10, Softmax)) using only these K labeled samples.
4. Compare against two baselines: MLP trained on the full labeled training set, and MLP trained on 50 randomly chosen labels.
5. Analyze the best-performing K via confusion matrix and per-class classification report.

### Part 2: Overhead MNIST — K-Means + SVM
Same clustering-based sample-selection methodology, applied to the Overhead MNIST dataset with an SVM classifier instead of an MLP. Includes baseline comparison, K-sweep, and analysis of which samples were selected by clustering.

### Technologies Used
* Python (NumPy, Pandas, Matplotlib)
* TensorFlow / Keras (MLP)
* scikit-learn (K-Means, SVM)
* Google Colab / Google Drive

---

## How to Run

Both notebooks were developed in Google Colab and mount Google Drive to access datasets. To run:

1. Open the notebook in Google Colab.
2. Update the dataset paths under the Drive-mounting cell to point to your own copy of the data.
3. Run all cells top to bottom.

---

## 🎯 Summary

* **Assignment 2** demonstrates that distance metric choice matters for sparse text data, and that initialization strategy (K-Means++ vs. random) matters for highly sparse, uneven datasets like Enron.
* **Assignment 3** demonstrates that clustering-based sample selection for labeling can substantially reduce labeling cost while approaching fully-labeled accuracy, across both a neural network (MLP) and a classical classifier (SVM).
