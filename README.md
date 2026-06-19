# 🎓 Course Recommendation System — IBM ML Capstone (Coursera)

> **Capstone project** for the [IBM Machine Learning Professional Certificate](https://www.coursera.org/professional-certificates/ibm-machine-learning) on Coursera.
> Builds an end-to-end personalized course recommender system using multiple ML paradigms — from content-based filtering to deep neural embeddings.

---

## 📌 Project Overview

Online learning platforms host thousands of courses, making it increasingly difficult for learners to discover content that matches their interests and skill level. This project tackles that problem by designing, implementing, and evaluating several **recommendation system algorithms** on real Coursera enrollment and rating data.

The pipeline progresses from exploratory analysis all the way through collaborative filtering with neural network embeddings, with each technique building on the previous one.

---

## 🗂️ Repository Structure

```
Course-recommendation-coursera-ibm-ml-capstone/
│
├── README.md
│
└── ibm-ml-capstone/
    │
    ├── Course_Work/                          # Lab notebooks (guided IBM course labs)
│   ├── 01_lab_jupyter_eda.ipynb                        # Exploratory Data Analysis
│   ├── 02_lab_jupyter_fe_bow_solution.ipynb            # Feature Engineering: Bag of Words
│   ├── 03_lab_jupyter_content_user_profile.ipynb       # Content-Based: User Profiles
│   ├── 04_lab_jupyter_content_course_similarity.ipynb  # Content-Based: Course Similarity
│   ├── 05_lab_jupyter_content_clustering.ipynb         # Content-Based: K-Means Clustering
│   ├── 06_lab_jupyter_cf_knn.ipynb                     # Collaborative Filtering: KNN
│   ├── 07_lab_jupyter_cf_nmf.ipynb                     # Collaborative Filtering: NMF
│   ├── 08_lab_jupyter_cf_ann.ipynb                     # Collaborative Filtering: Neural Networks
│   ├── 09_lab_jupyter_cf_regression_w_embeddings.ipynb # CF + Regression w/ Embeddings
│   ├── 10_lab_jupyter_cf_classification_w_embeddings.ipynb # CF + Classification w/ Embeddings
│   ├── course_ratings.csv                              # Course ratings dataset
│   └── profile_rs_results.csv                         # User profile recommender results
│
├── Some_Personal_Notebooks/              # Independent implementations & experiments
│   ├── 01_EDA.ipynb                               # Personal EDA exploration
│   ├── 02_ContentBased_UserProfile.ipynb          # Custom user profile implementation
│   ├── 03_ContentBased_CourseSimilarity.ipynb     # Custom course similarity
│   ├── 04_ContentBased_Clustering.ipynb           # Custom clustering experiments
│   ├── 05_CF_KNN.ipynb                            # KNN CF using Surprise library
│   ├── 06_CF_NMF.ipynb                            # NMF collaborative filtering
│   ├── 07_CF_NeuralEmbeddings_Comparison.ipynb    # Neural embedding model comparison
│   ├── courses_bowed.csv                          # Bag-of-Words encoded course features
│   ├── cf_model_comparison.png                    # CF model performance comparison chart
│   ├── enrollment_distribution.png               # Enrollment distribution plot
│   ├── genre_distribution.png                    # Course genre distribution plot
│   ├── knn_comparison.png                        # KNN similarity metric comparison
│   ├── nmf_factors.png                           # NMF latent factor visualization
│   ├── nn_training_curve.png                     # Neural network training curve
│   ├── pca_elbow.png                             # PCA elbow plot for clustering
│   └── wordcloud.png                             # Word cloud of course titles
│
└── Report_And_Presentation/
    ├── Machine Learning Capstone Project Report.pdf   # Full project report
    └── ML_Capstone_Project_Presentation.pptx         # Project presentation slides
```

---

## 🧠 Techniques Implemented

### 1. Exploratory Data Analysis (EDA)

- Word cloud visualization of course titles to identify key topics
- Summary statistics and distributions of course genres
- Enrollment patterns across courses and users
- Identification of most popular courses by enrollment count

### 2. Feature Engineering

- **Bag of Words (BoW)** encoding on course titles and descriptions
- **Genre-based binary vectors** to represent course content
- **User profile vectors** built from weighted genre ratings

### 3. Content-Based Filtering

| Method             | Description                                                                                                          |
| ------------------ | -------------------------------------------------------------------------------------------------------------------- |
| User Profile       | Builds a genre preference vector per user from their enrollment history                                              |
| Course Similarity  | Computes cosine similarity between course genre vectors to recommend similar courses                                 |
| K-Means Clustering | Groups users by learning interests using PCA-reduced profile vectors; recommends popular courses within each cluster |

### 4. Collaborative Filtering (CF)

| Method                                 | Key Details                                                                                                              |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **KNN**                          | User-based and item-based CF using MSD and cosine similarity (via Surprise library)                                      |
| **NMF**                          | Decomposes the user-item interaction matrix into latent user/item factor matrices                                        |
| **Neural Network Embeddings**    | Trains a Keras embedding model (16-dim) with dot-product interaction; binary classification setup with negative sampling |
| **Regression w/ Embeddings**     | Uses extracted embedding vectors as features for Ridge regression to predict numerical ratings                           |
| **Classification w/ Embeddings** | Uses embeddings as features for Logistic Regression / Random Forest to predict rating mode                               |

---

## 📊 Key Results

- **Content-based clustering** effectively grouped users by domain interest (ML, Cloud, Web Dev, etc.), enabling group-level recommendations.
- **NMF** provided compact latent representations and competitive RMSE with significantly reduced memory compared to KNN.
- **Neural embedding models** learned rich latent features in an end-to-end manner, with training curves showing stable convergence.
- **Regression/Classification with embeddings** demonstrated that pre-trained embeddings can serve as strong input features for downstream supervised models.

Refer to the [project report](Report_And_Presentation/Machine%20Learning%20Capstone%20Project%20Report.pdf) for detailed metrics and model comparisons.

---

## 🛠️ Tech Stack

| Category                | Libraries / Tools                                                       |
| ----------------------- | ----------------------------------------------------------------------- |
| Data Manipulation       | `pandas`, `numpy`                                                   |
| Visualization           | `matplotlib`, `seaborn`, `wordcloud`                              |
| Machine Learning        | `scikit-learn` (KMeans, PCA, Ridge, LogisticRegression, RandomForest) |
| Collaborative Filtering | `scikit-surprise` (KNNBasic, NMF)                                     |
| Deep Learning           | `tensorflow` / `keras`                                              |
| Environment             | Jupyter Notebook, Python 3.x                                            |

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn wordcloud scikit-learn scikit-surprise tensorflow
```

### Running the Notebooks

1. Clone this repository:

   ```bash
   git clone https://github.com/shivanshbtr/ibm-ml-capstone.git
   cd ibm-ml-capstone
   ```
2. Launch Jupyter:

   ```bash
   jupyter notebook
   ```
3. Navigate through the notebooks in order — `Course_Work/` for the guided labs or `Some_Personal_Notebooks/` for independent implementations.

> **Note:** Datasets are loaded directly from IBM Cloud Object Storage URLs within the notebooks. No local dataset download is required (except `course_ratings.csv` and `profile_rs_results.csv` already included in `Course_Work/`).

---

## 📄 Report & Presentation

- 📝 [Full Project Report (PDF)](Report_And_Presentation/Machine%20Learning%20Capstone%20Project%20Report.pdf)
- 📊 [Project Presentation (PPTX)](Report_And_Presentation/ML_Capstone_Project_Presentation.pptx)

---

## 🏅 Certificate

This project was completed as part of the **IBM Machine Learning Professional Certificate** on Coursera.
Course: *Machine Learning Capstone* — [IBM ML321EN](https://www.coursera.org/learn/machine-learning-capstone)

---

*Made with curiosity, caffeine, and a lot of `model.fit()` calls ☕*
