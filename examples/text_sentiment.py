"""Upload a text classifier, then predict and explain one review.

A scikit-learn Pipeline (TF-IDF, then logistic regression) is trained on 60 short reviews written below, so
nothing is downloaded. Kaimei reads the pipeline exactly, words and all, and the model then takes text.

    pip install kaimei scikit-learn
    export KAIMEI_API_KEY=km_live_...          # optional: export KAIMEI_BASE_URL=http://localhost:8000
    python examples/text_sentiment.py
"""

from __future__ import annotations

import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from kaimei import Kaimei

# Step 1. A small review corpus: 30 positive and 30 negative, in the file itself.
POSITIVE = [
    "a wonderful film with a warm and moving story",
    "brilliant acting and a clever script",
    "i loved every minute of it",
    "a joy to watch from start to finish",
    "beautiful music and gorgeous images",
    "funny, sharp and full of heart",
    "the best film i have seen this year",
    "a great cast and a satisfying ending",
    "charming, clever and wonderfully acted",
    "an excellent and touching drama",
    "superb direction and a gripping plot",
    "delightful from the first scene to the last",
    "a fresh, lively and entertaining comedy",
    "moving and beautifully made",
    "the leads are terrific and the dialogue sparkles",
    "i would happily see it again",
    "a smart, warm and memorable film",
    "stunning photography and a great score",
    "genuinely funny and surprisingly emotional",
    "a masterpiece of quiet storytelling",
    "lovely performances and a wonderful pace",
    "thrilling, inventive and great fun",
    "a heartfelt story told with skill",
    "pure delight, i recommend it",
    "wonderful characters and a clever twist",
    "an impressive and rewarding film",
    "so charming that i smiled all the way through",
    "great writing and superb acting",
    "an uplifting, beautiful and funny story",
    "excellent from beginning to end",
]
NEGATIVE = [
    "a dull film with a weak and confusing story",
    "terrible acting and a lazy script",
    "i hated every minute of it",
    "a bore to watch from start to finish",
    "ugly images and awful music",
    "unfunny, flat and empty of heart",
    "the worst film i have seen this year",
    "a poor cast and a disappointing ending",
    "clumsy, tedious and badly acted",
    "a boring and silly drama",
    "weak direction and a predictable plot",
    "dreadful from the first scene to the last",
    "a stale, slow and tiresome comedy",
    "messy and badly made",
    "the leads are wooden and the dialogue is painful",
    "i would never see it again",
    "a stupid, cold and forgettable film",
    "dull photography and a poor score",
    "painfully unfunny and strangely cold",
    "a mess of loud and pointless scenes",
    "terrible performances and a slow pace",
    "boring, predictable and a waste of time",
    "a hollow story told without skill",
    "pure misery, i do not recommend it",
    "awful characters and a silly twist",
    "a disappointing and tiresome film",
    "so tedious that i looked at my watch all the way through",
    "poor writing and bad acting",
    "a depressing, ugly and slow story",
    "terrible from beginning to end",
]
texts = POSITIVE + NEGATIVE
labels = [1] * len(POSITIVE) + [0] * len(NEGATIVE)

# Step 2. Train the pipeline. Nothing here is Kaimei's.
pipeline = Pipeline([("tfidf", TfidfVectorizer()), ("model", LogisticRegression(C=10.0))])
pipeline.fit(texts, labels)
print(f"The pipeline was trained on {len(texts)} reviews and scores {pipeline.score(texts, labels):.0%} on them.")

# Step 3. Connect to Kaimei.
client = Kaimei(
    api_key=os.environ["KAIMEI_API_KEY"],
    base_url=os.environ.get("KAIMEI_BASE_URL") or "https://api.kaimei.ai",
)

# Step 4. Upload it. The reviews are given as X so the reading is checked on your own texts.
model = client.models.upload(
    pipeline,
    name="Review sentiment",
    X=texts,
    labels=["negative", "positive"],
)
print(f"The model is registered as {model.id}.")

# Step 5. Ask Kaimei how it read the model: exactly, and whether it runs the model itself.
structure = client.models.structure(model.id, detail="summary")
print(
    f"Kaimei's reading is {structure['fidelity']} (check: {structure['verification']}); "
    f"it {'runs' if structure['runnable'] else 'only draws'} the model."
)

# Step 6. One prediction. The feature is the text itself.
review = {"text": "a clever and moving film, but the ending was weak"}
prediction = client.models.predict(model.id, input=review)
output = prediction.output
print(f"Kaimei predicts {output['label']} ({output['probability']:.1%}) for {review['text']!r}.")
print(f"The pipeline itself says {pipeline.predict_proba([review['text']])[0].round(4).tolist()} (negative, positive).")

# Step 7. One explanation: which words moved the score.
explanation = client.explanations.create(model=model.id, input=review, method="shapley")
print("The words that mattered most, with their attribution toward the predicted class:")
for part in explanation.top_features[:5]:
    print(f"  {part['value']:<10} {part['attribution']:+.3f}")
