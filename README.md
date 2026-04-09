# 🎵 Spotify Explorer

Une application Streamlit interactive pour explorer, recommander et visualiser des chansons à partir de données Spotify ! Ajoute des filtres par humeur, découvre la popularité des titres, crée une playlist personnalisée et explore des tendances musicales.

---

## 🚀 Fonctionnalités principales

- **Accueil** : Stats rapides, top 20 titres/popularité/artistes, et découverte aléatoire.
- **Recherche** : Recherche par chanson ou artiste avec filtre humeur et aperçu audio (YouTube/MP3).
- **Analytics** : Histogrammes, corrélations, scatter plots, filtrage avancé.
- **Trends** : Évolution de la musique et des tendances audio par année.
- **Recommandations** : Suggestions par similarité ou par humeur personnalisée (mood).
- **Ma Playlist** : Ajoute, retire, exporte tes morceaux préférés. Visualisation directe de ta playlist.

---

## 🎭 Système de filtre par humeur

| Humeur      | Description                                                  |
|-------------|-------------------------------------------------------------|
| 😊 Happy    | high valence (>=0.6) & énergie (>=0.4)                       |
| 😢 Sad      | valence (<=0.4) & énergie (<=0.5)                            |
| 😌 Chill    | valence (<=0.6) & énergie (<=0.5) & danceability (>= 0.3)    |
| ⚡ Energetic| énergie (>= 0.7) & tempo (>=120)                             |

---

## 📦 Prérequis & Installation

```bash
pip install streamlit pandas numpy plotly scikit-learn
```

---

## 🗂 Fichiers attendus

- **app.py** (ton code python Streamlit principal)
- **output.csv** (jeu de données audio Spotify, avec les colonnes : name, artists, release_date, year, duration_ms, valence, energy, ...)

---

## 🏁 Démarrage rapide

```bash
streamlit run app.py
```

Ouvre dans ton navigateur, à l’adresse http://localhost:8501

---

## 📊 Caractéristiques audio

| Feature           | Explication                                        |
|-------------------|----------------------------------------------------|
| valence           | Positivité/humeur du morceau (0 triste, 1 joyeux)  |
| energy            | Intensité et énergie globale du morceau (0-1)      |
| danceability      | Adapté à la danse ? (0-1)                         |
| acousticness      | Acoustique ou électronique (0-1)                   |
| instrumentalness  | Instrumental (0-1)                                 |
| liveness          | Présence d’un public live (0-1)                    |
| speechiness       | Présence de paroles parlées (0-1)                  |
| tempo             | Vitesse en BPM                                     |
| popularity        | Score de popularité sur Spotify (0-100)            |

---

## 🎨 Astuce : Personnalisation

- Les couleurs/dark mode sont adaptables dans le bloc CSS du fichier python.
- La playlist reste mémorisée via `st.session_state`.

---

## 🤝 Contribuer

Pull requests bienvenues ! N’hésite pas à ouvrir des issues pour signaler un bug ou proposer une amélioration.

---

**Bon remix avec Spotify Explorer ! 🎶**
