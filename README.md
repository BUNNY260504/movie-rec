# movie-rec

 🎬 Movie Recommender System

A content‑based movie recommender web app that helps you discover films using TMDb metadata and similarity scores. Users can search for movies, get similar recommendations, explore trending titles, and view rich details like posters, overviews, ratings, genres, and cast information in a clean UI.

***

## Features

- Search any movie by title and instantly load its details from TMDb (poster, overview, rating, genres, etc.).
- Get top N similar movies using a pre‑computed similarity matrix (content‑based filtering with cosine similarity).
- Optional “Surprise Me” / random pick from the catalog to help when the user is undecided.
- Support for trending or popular movies using TMDb endpoints.
- Interactive web UI (likely Streamlit/Flask) with simple controls and card‑based movie display.

***

## Tech Stack

| Layer          | Technologies                                                                 |
|----------------|------------------------------------------------------------------------------|
| Language       | Python 3.x                                                                   |
| Data / ML      | Pandas, NumPy, scikit‑learn (cosine similarity, vectorization)  |
| Web Framework  | Streamlit or Flask for the front‑end web app              |
| External API   | TMDb API for movie metadata, posters, and images           |
| Data Storage   | Local CSV / pickle files for movies metadata and similarity matrix = |
| Environment    | Conda/virtualenv, requirements.txt for dependencies          |


***
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

## How It Works

1. **Data ingestion and preprocessing**  
   - Load movie metadata (title, genres, overview, cast, crew, etc.) from TMDb/Kaggle and clean missing values.
   - Create a combined text feature (e.g., genres + overview + cast) for each movie to represent its content.

2. **Feature engineering & similarity**  
   - Vectorize the combined text feature using techniques like CountVectorizer or TF‑IDF.
   - Compute pairwise cosine similarity between movies and store the resulting similarity matrix in a pickle file.

3. **Recommendation logic**  
   - For a selected movie, look up its index, retrieve the similarity scores from the matrix, and sort movies by descending similarity.
   - Return top N similar movies (excluding the selected movie) along with their IDs.

4. **TMDb integration & UI**  
   - Use TMDb movie IDs to fetch live details (poster path, overview, vote_average, genres, release date, cast) via the TMDb API.
   - Render recommendations with posters and details in the web app, including options like “Search”, “Recommend”, and “Surprise Me”.

***

## Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/BUNNY260504/Movie-Recommender-System.git
cd Movie-Recommender-System
```


### 2. Create and activate a virtual environment

Using conda:

```bash
conda create -n movies python=3.10 -y
conda activate movies
```


Or with venv:

```bash
python -m venv venv
source venv/bin/activate    # Linux / macOS
venv\Scripts\activate       # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```


### 4. Set your TMDb API key

1. Create a TMDb account and generate an API key from your TMDb dashboard.
2. Create a `.env` file in the project root (or set environment variables directly):

```env
TMDB_API_KEY=your_tmdb_api_key_here
```

3. Ensure your code reads this key (e.g., using `python-dotenv` or `os.environ`).

### 5. Prepare model artifacts (if not already included)

If `movies.pkl` and `similarity.pkl` are not committed, run the preprocessing notebook/script:

```bash
python scripts/build_similarity_matrix.py
```

This script should: load raw data, engineer features, compute similarity, and save `*.pkl`.

***

## Usage Guide

### Run the web application

If it is a Streamlit app:

```bash
streamlit run app.py
```


Then open the URL shown in the terminal (usually `http://localhost:8501` for Streamlit) in your browser.

### In‑app workflow

- Use the search box to type a movie title. Autocomplete or exact match will select the movie from your dataset.
- Click the “Recommend” button to get top similar movies along with posters, ratings, and overviews.
- Optionally use “Surprise Me” to get a random movie suggestion, or switch to “Trending” to see currently popular titles from TMDb.

***

## Model and Data Details

- **Recommendation type**: Content‑based filtering (no explicit user ratings needed).[1][3]
- **Features used**: Typically a combination of genres, overview, cast, and crew fields; you can adjust this in `recommender.py`.
- **Similarity metric**: Cosine similarity on vectorized text features, stored in a pre‑computed matrix for fast lookup.
- **Data source**: TMDb or a Kaggle TMDb dataset (e.g., TMDB 5000 Movies) for offline metadata, plus TMDb API calls for live posters and details.




