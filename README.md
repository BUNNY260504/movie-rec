

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
