from __future__ import annotations
from typing import Optional, Tuple, List
import re
import nltk
from nltk.corpus import wordnet as wn

# Asegurar recursos
try:
    wn.synsets('test')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

# Palabras clave que apuntan a categorías
CATEGORY_KEYWORDS = {
    'animal': {
        'animal','mammal','canine','dog','hound','feline','cat','bird','fish','reptile','amphibian','insect'
    },
    'device': {
        'device','controller','remote','remote_control','gamepad','joystick','handset','transmitter','receiver','camera','computer','laptop','phone','keyboard','monitor','mouse','printer','router','tablet','headset','earbuds'
    },
    'container': {'container','bottle','box','jar','can','barrel','basket','bin','bucket','crate','tank','vessel'},
    'tool': {'tool','hammer','screwdriver','wrench','chisel','saw','pliers','drill','shovel','axe','trowel'},
    'vehicle': {'vehicle','car','bus','train','bicycle','truck','motorcycle','airplane','boat','ship','tram','scooter'},
    'furniture': {'chair','table','sofa','couch','desk','bed','wardrobe','cabinet','stool','bench'},
    'garment': {'shirt','pants','trousers','skirt','dress','jacket','coat','sweater','hat','gloves','socks','shoes'},
    'food': {'food','bread','apple','banana','cheese','butter','egg','tomato','potato','rice','pasta','meat','salad'},
    'instrument': {'instrument','guitar','piano','violin','trumpet','drums','flute','saxophone','cello','clarinet'},
    'body_part': {'hand','arm','leg','foot','head','eye','ear','nose','mouth','finger','knee','back'},
    'plant': {'tree','flower','grass','bush','leaf','root','seed','fern','moss'},
    'place': {'house','school','hospital','office','kitchen','garden','library','park','airport','station'},
    'abstract': {'freedom','happiness','idea','truth','knowledge','time','energy','information','power'}
}

DOG_HYPERNYM_KEYS = {'dog.n.01','hound.n.01','canine.n.02','domestic_dog.n.01'}

REMOTE_ALIASES = [
    'remote control','remote-controller','remote','tv remote','controller','handset','gamepad','joystick'
]

WORD_RE = re.compile(r"[a-zA-Z]+(?: [a-zA-Z]+)*")


def _normalize(text: str) -> str:
    return re.sub(r"[_-]+", " ", text.lower()).strip()


def _tokenize(text: str) -> List[str]:
    return _normalize(text).split()


def _synsets_noun(word: str):
    return [s for s in wn.synsets(word) if s.pos() == 'n']


def _hypernym_lemmas(s) -> List[str]:
    names = []
    for h in s.closure(lambda x: x.hypernyms()):
        names.extend([l.name().lower() for l in h.lemmas()])
    return names


def _contains_any(names: List[str], pool: set) -> bool:
    for n in names:
        if n in pool:
            return True
    return False


def _map_by_hypernym(word: str) -> Optional[str]:
    for s in _synsets_noun(word):
        names = set(_hypernym_lemmas(s))
        # animal detection (dog breeds etc.)
        if any(key in {h.name() for h in s.closure(lambda x: x.hypernyms())} for key in DOG_HYPERNYM_KEYS):
            return 'animal'
        for cat, keywords in CATEGORY_KEYWORDS.items():
            if _contains_any(list(names), keywords):
                return cat
    return None


def _map_by_keyword(word: str) -> Optional[str]:
    w = _normalize(word)
    # direct matches / aliases for remote
    for alias in REMOTE_ALIASES:
        if alias in w:
            return 'device'
    tokens = set(_tokenize(w))
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if tokens & keywords:
            return cat
    return None


def _canonical(word: str) -> str:
    w = _normalize(word)
    # special casing for remote control
    for alias in REMOTE_ALIASES:
        if alias in w:
            return 'remote control'
    return w


def is_dog_breed(word: str) -> bool:
    for s in _synsets_noun(word):
        chain = {h.name() for h in s.closure(lambda x: x.hypernyms())}
        if chain & DOG_HYPERNYM_KEYS:
            return True
    return False


def map_category(word: str) -> Tuple[Optional[str], str, float]:
    w = _normalize(word)
    if not w:
        return None, w, 0.0

    # 1) Try hypernym-based
    cat = _map_by_hypernym(w)
    if cat:
        return cat, _canonical(w), 0.9

    # 2) Keyword-based
    cat2 = _map_by_keyword(w)
    if cat2:
        return cat2, _canonical(w), 0.75

    # 3) Heuristics for plural/singular
    if w.endswith('s') and len(w) > 3:
        cat3 = _map_by_hypernym(w[:-1]) or _map_by_keyword(w[:-1])
        if cat3:
            return cat3, _canonical(w[:-1]), 0.7

    return None, _canonical(w), 0.0
