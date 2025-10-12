from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import tensorflow as tf

# Pequeño clasificador NN de categorías semánticas por palabra (inglés)
# - Extrae features de caracteres (frecuencia, sufijos/prefijos, métricas simples)
# - Entrena sobre un conjunto curado de semillas por categoría
# - Devuelve categoría para orientar generación de oraciones

CATEGORIES = [
    'animal', 'container', 'tool', 'vehicle', 'furniture', 'garment', 'food',
    'instrument', 'device', 'body_part', 'plant', 'place', 'abstract'
]

SEEDS: Dict[str, List[str]] = {
    'animal': ['cat','dog','horse','bird','fish','elephant','lion','tiger','monkey','sheep','goat','wolf','rabbit'],
    'container': ['bottle','box','jar','can','barrel','basket','bin','bucket','crate','tank','vessel'],
    'tool': ['hammer','screwdriver','wrench','chisel','saw','pliers','drill','shovel','axe','trowel'],
    'vehicle': ['car','bus','train','bicycle','truck','motorcycle','airplane','boat','ship','tram','scooter'],
    'furniture': ['chair','table','sofa','couch','desk','bed','wardrobe','cabinet','stool','bench'],
    'garment': ['shirt','pants','trousers','skirt','dress','jacket','coat','sweater','hat','gloves','socks','shoes'],
    'food': ['bread','apple','banana','cheese','butter','egg','tomato','potato','rice','pasta','meat','salad'],
    'instrument': ['guitar','piano','violin','trumpet','drums','flute','saxophone','cello','clarinet'],
    'device': ['phone','laptop','computer','camera','printer','router','tablet','keyboard','monitor'],
    'body_part': ['hand','arm','leg','foot','head','eye','ear','nose','mouth','finger','knee','back'],
    'plant': ['tree','flower','grass','bush','leaf','root','seed','fern','moss'],
    'place': ['house','school','hospital','office','kitchen','garden','library','park','airport','station'],
    'abstract': ['freedom','happiness','idea','truth','knowledge','time','energy','information','power']
}

SUFFIXES = ['tion','ment','ness','ing','er','or','al','ist','ity','able','less','ful','ous','ive']
PREFIXES = ['re','un','pre','trans','micro','auto','anti','post','sub','inter','over','under']

@dataclass
class NNClassifier:
    model: tf.keras.Model
    labels: List[str]

_classifier: NNClassifier | None = None


def _char_features(word: str) -> np.ndarray:
    w = word.lower()
    letters = 'abcdefghijklmnopqrstuvwxyz'
    counts = np.zeros(len(letters), dtype=np.float32)
    for ch in w:
        idx = letters.find(ch)
        if idx >= 0:
            counts[idx] += 1.0
    if len(w) > 0:
        counts /= len(w)
    # métricas simples
    vowels = set('aeiou')
    vowel_ratio = sum(1 for ch in w if ch in vowels) / max(1, len(w))
    length_norm = min(len(w), 20) / 20.0
    has_dash = 1.0 if '-' in w else 0.0

    # sufijos / prefijos
    suf = np.array([1.0 if w.endswith(s) else 0.0 for s in SUFFIXES], dtype=np.float32)
    pre = np.array([1.0 if w.startswith(p) else 0.0 for p in PREFIXES], dtype=np.float32)

    return np.concatenate([counts, [vowel_ratio, length_norm, has_dash], suf, pre], axis=0)


def _build_dataset() -> Tuple[np.ndarray, np.ndarray]:
    X: List[np.ndarray] = []
    y: List[int] = []
    for ci, cat in enumerate(CATEGORIES):
        for w in SEEDS.get(cat, []):
            X.append(_char_features(w))
            y.append(ci)
    X = np.stack(X, axis=0)
    y = tf.keras.utils.to_categorical(np.array(y, dtype=np.int32), num_classes=len(CATEGORIES))
    return X, y


def _build_model(input_dim: int, output_dim: int) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(output_dim, activation='softmax'),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def get_category_classifier() -> NNClassifier:
    global _classifier
    if _classifier is not None:
        return _classifier

    # Construir dataset pequeño
    X, y = _build_dataset()
    input_dim = X.shape[1]
    output_dim = y.shape[1]

    # Construir y entrenar modelo (rápido)
    model = _build_model(input_dim, output_dim)
    model.fit(X, y, epochs=80, batch_size=16, verbose=0, validation_split=0.1)

    _classifier = NNClassifier(model=model, labels=CATEGORIES)
    return _classifier


def predict_category(word: str) -> Tuple[str, float]:
    clf = get_category_classifier()
    x = _char_features(word)
    x = np.expand_dims(x, axis=0)
    probs = clf.model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    return clf.labels[idx], float(probs[idx])
