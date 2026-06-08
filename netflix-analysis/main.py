import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("images", exist_ok=True)

base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, "netflix_titles.csv")
df = pd.read_csv(csv_path)

df['director'] = df['director'].fillna("Unknown")
df['cast'] = df['cast'].fillna("Unknown")
df['country'] = df['country'].fillna("Unknown")

type_counts = df['type'].value_counts()
print("Film vs Dizi:\n", type_counts)

plt.figure(figsize=(6,4))
sns.countplot(data=df, x='type')
plt.title("Netflix Content Types")
plt.tight_layout()
plt.savefig("images/content_types.png")
plt.close()

country_counts = df.groupby('country')['title'].count().sort_values(ascending=False)

print("\nTop Countries:\n")
print(country_counts.head(10))

top10_countries = country_counts.head(10)

plt.figure(figsize=(10,5))
top10_countries.plot(kind='bar')
plt.title("Top 10 Countries by Content Count")
plt.ylabel("Content Count")
plt.tight_layout()
plt.savefig("images/top_countries.png")
plt.close()

df['genre'] = df['listed_in'].apply(lambda x: x.split(",")[0])

genre_counts = df.groupby('genre')['title'].count().sort_values(ascending=False)

print("\nTop Genres:\n")
print(genre_counts.head(10))

top_genres = genre_counts.head(10)

plt.figure(figsize=(10,5))
top_genres.plot(kind='bar')
plt.title("Top Genres")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("images/top_genres.png")
plt.close()

df['duration_int'] = df['duration'].str.extract(r'(\d+)').astype(float)

print("\nAverage Duration:")
print(np.mean(df['duration_int']))


def content_score(row):
    score = 0

    if "Drama" in str(row['listed_in']):
        score += 2

    if row['type'] == "Movie":
        score += 1

    if row['duration_int'] > 90:
        score += 2

    return score

df['score'] = df.apply(content_score, axis=1)


def segment(score):
    if score >= 4:
        return "High"
    elif score >= 2:
        return "Medium"
    else:
        return "Low"

df['segment'] = df['score'].apply(segment)

print("\nSegment Distribution:\n")
print(df['segment'].value_counts())

plt.figure(figsize=(6,4))
sns.countplot(data=df, x='segment')
plt.title("Content Segmentation")
plt.tight_layout()
plt.savefig("images/segment_distribution.png")
plt.close()


top_content = df.sort_values('score', ascending=False)

print("\nTop Content:\n")
print(top_content[['title', 'score']].head(10))


matrix = df[['duration_int']].dropna().values
result = np.dot(matrix.T, matrix)

print("\nMatrix Result:\n")
print(result)

print("\nGrafikler 'images' klasörüne başarıyla kaydedildi.")