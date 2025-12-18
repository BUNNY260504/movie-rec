# 🎬 Movie Recommender System

## 📖 Overview

A **content-based movie recommendation system** that suggests films based on similarity in genres, keywords, cast, crew, and plot. Built with Streamlit and powered by machine learning, it provides personalized recommendations with rich metadata from TMDB API.

### Key Highlights

- 🎯 **Content-Based Filtering** using NLP and cosine similarity
- 🔴 **Real-Time Data** from TMDB API (posters, trailers, cast, ratings)
- ⚡ **Fast Recommendations** with pre-computed similarity matrix
- 📊 **4,800+ Movies** in the catalog
- 🎨 **Interactive UI** with trending movies, random suggestions, and viewing history

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Movie Search** | Search from 4,800+ movies and get instant recommendations |
| **Surprise Me** | Random movie discovery with full details |
| **Trending Movies** | Weekly trending films from TMDB |
| **Rich Metadata** | Cast, crew, budget, revenue, ratings, runtime, trailers |
| **Viewing History** | Track and revisit recently viewed movies |
| **Responsive Design** | Mobile-friendly interface |

---


**What you can do:**
- Search through 4,800+ movies
- Get 5 similar movie recommendations instantly
- View detailed information (cast, crew, budget, ratings, trailers)
- Discover trending movies weekly
- Get random movie suggestions


---

## 🏗️ Architecture

### System Overview

```
┌─────────────────┐
│   User Input    │
│  (Movie Title)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Streamlit App  │
│   (Frontend)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Recommender    │
│    Engine       │
│ (Cosine Sim.)   │
└────────┬────────┘
         │
         ├───────────────┬───────────────┐
         ▼               ▼               ▼
┌──────────────┐  ┌──────────┐  ┌──────────────┐
│ Similarity   │  │ TMDB API │  │ Local Cache  │
│ Matrix (pkl) │  │ (Live)   │  │ (Session)    │
└──────────────┘  └──────────┘  └──────────────┘
```

### Recommendation Algorithm

1. **Text Vectorization**: Convert movie features (genres, keywords, cast, crew, overview) into vectors using CountVectorizer (5000 features)
2. **Similarity Computation**: Calculate cosine similarity between all movie pairs (4806 × 4806 matrix)
3. **Recommendation**: For a given movie, retrieve top 5 most similar movies based on cosine similarity scores

**Cosine Similarity Formula:**
```
similarity(A, B) = (A · B) / (||A|| × ||B||)
```


---

## 🔧 Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

### Core Dependencies

| Category | Technologies |
|----------|-------------|
| **Framework** | Streamlit |
| **ML/NLP** | scikit-learn, NLTK (PorterStemmer) |
| **Data Processing** | Pandas, NumPy, Pickle |
| **API** | TMDB API, Requests |
| **Deployment** | Streamlit Cloud |


---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/hk-kumawat/Movie-Recommender-System.git
cd Movie-Recommender-System

# Install dependencies
pip install -r requirements.txt

# Set up TMDB API key (see below)
mkdir .streamlit
echo '[tmdb]\napi_key = "YOUR_API_KEY"' > .streamlit/secrets.toml

# Run the application
streamlit run app.py
```

**Access the app at:** `http://localhost:8501`


---

## ⚙️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- TMDB API key ([Get one here](https://www.themoviedb.org/settings/api))

### Step-by-Step Setup

**1. Clone the Repository**
```bash
git clone https://github.com/hk-kumawat/Movie-Recommender-System.git
cd Movie-Recommender-System
```

**2. Create Virtual Environment** (Recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Configure TMDB API Key**

Create `.streamlit/secrets.toml`:
```toml
[tmdb]
api_key = "your_tmdb_api_key_here"
```

**How to get TMDB API Key:**
1. Sign up at [themoviedb.org](https://www.themoviedb.org/)
2. Go to Settings → API
3. Request API Key (select "Developer")
4. Copy your API key

**5. Run the Application**
```bash
streamlit run app.py
```

### Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| API Key Error | Check `.streamlit/secrets.toml` format |
| Port Already in Use | Use `streamlit run app.py --server.port 8502` |
| NLTK Data Missing | Run `python -m nltk.downloader punkt stopwords` |



---

## 📊 Dataset

**Source:** [TMDb 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) (Kaggle)

### Dataset Details

| File | Records | Description |
|------|---------|-------------|
| `tmdb_5000_movies.csv` | 4,803 | Movie metadata (title, overview, genres, keywords, budget, revenue) |
| `tmdb_5000_credits.csv` | 4,803 | Cast and crew information |

**Key Statistics:**
- **Movies:** 4,806 (after preprocessing)
- **Features:** 5,000 (CountVectorizer)
- **Genres:** 20 unique genres
- **Time Period:** 1916-2017

### Data Processing Pipeline

```
Raw Data
    ↓
Merge movies + credits
    ↓
Extract features (genres, keywords, cast, crew, overview)
    ↓
Text preprocessing (lowercase, remove spaces)
    ↓
Stemming (PorterStemmer)
    ↓
Combine into "tags" column
    ↓
Vectorize (CountVectorizer, max_features=5000)
    ↓
Compute cosine similarity matrix (4806 × 4806)
    ↓
Save model (movie_list.pkl, similarity.pkl)
```


---

## 🧠 Model Training

The recommendation model is trained using a content-based filtering approach. Here's how it works:

### Training Process

**1. Data Collection & Preprocessing**
```python
# Load datasets
movies = pd.read_csv('Dataset/tmdb_5000_movies.csv')
credits = pd.read_csv('Dataset/tmdb_5000_credits.csv')

# Merge on title
movies = movies.merge(credits, on='title')

# Extract relevant features
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
```

**2. Feature Engineering**
```python
# Extract top 3 cast members
def convert_cast(text):
    return [actor['name'] for actor in ast.literal_eval(text)[:3]]

# Extract director from crew
def fetch_director(text):
    for person in ast.literal_eval(text):
        if person['job'] == 'Director':
            return [person['name']]
    return []

# Apply transformations
movies['cast'] = movies['cast'].apply(convert_cast)
movies['crew'] = movies['crew'].apply(fetch_director)
movies['genres'] = movies['genres'].apply(lambda x: [genre['name'] for genre in ast.literal_eval(x)])
movies['keywords'] = movies['keywords'].apply(lambda x: [kw['name'] for kw in ast.literal_eval(x)])
```

**3. Text Processing**
```python
# Combine all features into tags
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# Convert to string and lowercase
movies['tags'] = movies['tags'].apply(lambda x: ' '.join(x).lower())

# Apply stemming
from nltk.stem import PorterStemmer
ps = PorterStemmer()
movies['tags'] = movies['tags'].apply(lambda x: ' '.join([ps.stem(word) for word in x.split()]))
```

**4. Vectorization & Similarity Computation**
```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Create count vectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()

# Compute cosine similarity matrix
similarity = cosine_similarity(vectors)

# Save models
pickle.dump(movies, open('model_files/movie_list.pkl', 'wb'))
pickle.dump(similarity, open('model_files/similarity.pkl', 'wb'))
```

### Model Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `max_features` | 5000 | Maximum vocabulary size for CountVectorizer |
| `stop_words` | 'english' | Remove common English words |
| `similarity_metric` | Cosine Similarity | Measure of similarity between vectors |
| `top_n_recommendations` | 5 | Number of recommendations to return |

### Training Environment

- **Notebook:** `Movie Recommender System.ipynb`
- **Training Time:** ~2 minutes (on standard CPU)
- **Model Size:** 184 MB (similarity matrix)
- **Libraries:** scikit-learn, NLTK, Pandas, NumPy


---

## 📁 Project Structure

```
Movie-Recommender-System/
│
├── app.py                          # Main Streamlit application
├── Movie Recommender System.ipynb  # Data preprocessing & model training
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore rules
├── LICENSE                         # MIT License
│
├── Dataset/                        # Raw movie data
│   ├── tmdb_5000_movies.csv
│   └── tmdb_5000_credits.csv
│
├── model_files/                    # Trained models
│   ├── movie_list.pkl             # Movie data (4806 movies)
│   └── similarity.pkl             # Cosine similarity matrix (4806×4806)
│
└── .streamlit/                     # Configuration (not in repo)
    └── secrets.toml               # TMDB API key
```


---

## 📈 Performance

### Model Metrics

| Metric | Value |
|--------|-------|
| **Movies in Catalog** | 4,806 |
| **Feature Dimensions** | 5,000 |
| **Similarity Matrix Size** | 4,806 × 4,806 |
| **Average Recommendation Time** | <2 seconds |
| **Model Size** | 184 MB (similarity.pkl) |

### System Performance

- **API Response Time:** ~1.2s (TMDB)
- **Recommendation Generation:** ~0.8s
- **Memory Usage:** ~500MB
- **Concurrent Users:** 100+


---

## 🎯 How to Use

### Web Application

1. **Search Mode:** Select a movie from the dropdown and click "Show Details & Recommendations"
2. **Surprise Mode:** Click "Surprise Me!" for a random movie suggestion
3. **Trending:** View weekly trending movies at the top
4. **History:** Access recently viewed movies from the sidebar

### API Integration

```python
import pickle
import pandas as pd

# Load models
movies = pickle.load(open('model_files/movie_list.pkl', 'rb'))
similarity = pickle.load(open('model_files/similarity.pkl', 'rb'))

# Get recommendations
def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommendations = []
    for i in distances[1:6]:
        recommendations.append(movies.iloc[i[0]].title)
    return recommendations

# Example
print(recommend('Avatar'))
# Output: ['Guardians of the Galaxy', 'Star Wars', 'Star Trek', ...]
```


---

## 📚 API Reference

### Core Functions

#### `recommend(movie_title: str) -> list`

Returns top 5 similar movies based on content similarity.

**Parameters:**
- `movie_title` (str): Title of the movie (must exist in dataset)

**Returns:**
- List of dictionaries containing recommended movies with poster URLs and trailers

**Example:**
```python
recommendations = recommend('The Dark Knight')
# Returns: [
#   {'title': 'The Dark Knight Rises', 'poster': '...', 'trailer': '...'},
#   {'title': 'Batman Begins', 'poster': '...', 'trailer': '...'},
#   ...
# ]
```

#### `get_movie_details(movie_id: int) -> dict`

Fetches comprehensive movie information from TMDB API.

**Parameters:**
- `movie_id` (int): TMDB movie ID

**Returns:**
- Dictionary with rating, cast, crew, budget, revenue, genres, etc.

**Example:**
```python
details = get_movie_details(19995)  # Avatar
# Returns: {
#   'rating': 7.2,
#   'cast': [...],
#   'director': 'James Cameron',
#   'budget': '$237,000,000',
#   ...
# }
```

#### `get_trending_movies() -> list`

Gets current trending movies from TMDB API.

**Returns:**
- List of top 5 trending movies with posters and IDs

#### `fetch_poster(movie_id: int) -> str`

Fetches movie poster URL from TMDB API.

**Parameters:**
- `movie_id` (int): TMDB movie ID

**Returns:**
- Full URL to movie poster (500px width)

#### `fetch_trailer(movie_id: int) -> str`

Fetches YouTube trailer URL from TMDB API.

**Parameters:**
- `movie_id` (int): TMDB movie ID

**Returns:**
- YouTube URL to official trailer (if available)

### Configuration

**Environment Variables:**
```python
TMDB_API_KEY = st.secrets["tmdb"]["api_key"]  # From .streamlit/secrets.toml
```

**Session State:**
```python
st.session_state.history        # Recently viewed movies (list of IDs)
st.session_state.mode           # Current mode: 'search' or 'surprise'
st.session_state.selected_movie # Currently selected movie title
```
