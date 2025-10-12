import random
from typing import List

# Intentamos cargar NLTK WordNet; si no está, se descarga en tiempo de ejecución
import nltk
from nltk.corpus import wordnet as wn
from models.example_nn import predict_category
from utils.taxonomy_mapper import map_category, is_dog_breed


def ensure_nltk():
  try:
    wn.synsets('test')
  except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')


def _indefinite_article(word: str) -> str:
  if not word:
    return 'a'
  w = word.lower()
  exceptions_an = ['honest', 'hour', 'honor', 'heir', 'herb']
  for e in exceptions_an:
    if w.startswith(e):
      return 'an'
  if w.startswith(('uni', 'use', 'user', 'euro', 'one')):
    return 'a'
  return 'an' if w[0] in 'aeiou' else 'a'


def _pick_synonyms(word: str, pos=None, limit=6) -> List[str]:
  seen = set()
  out = []
  for s in wn.synsets(word):
    if pos and s.pos() != pos:
      continue
    for l in s.lemmas():
      n = l.name().replace('_', ' ')
      if n.lower() != word.lower() and n not in seen:
        seen.add(n)
        out.append(n)
        if len(out) >= limit:
          return out
  return out


def _pick_hypernyms(word: str, pos=None, limit=5) -> List[str]:
  out = []
  for s in wn.synsets(word):
    if pos and s.pos() != pos:
      continue
    for h in s.hypernyms():
      name = h.lemmas()[0].name().replace('_', ' ')
      if name not in out:
        out.append(name)
        if len(out) >= limit:
          return out
  return out


def _related_context_terms(word: str) -> List[str]:
  # Recolecta sinónimos e hiperónimos ligeros
  syns = _pick_synonyms(word, limit=6)
  hypers = _pick_hypernyms(word, limit=6)
  return syns[:3] + hypers[:3]


def _templates(word: str) -> List[str]:
  w = word
  art = _indefinite_article(w)
  rel = _related_context_terms(w)
  r1 = rel[0] if len(rel) > 0 else w
  r2 = rel[1] if len(rel) > 1 else 'context'
  r3 = rel[2] if len(rel) > 2 else 'design'

  return [
    f"After {art} {w} was placed near the focal point, it subtly anchored the composition and contrasted with the surrounding {r2}.",
    f"Because {art} {w} had been engineered with durability in mind, professionals frequently preferred it over similar {r1}s during field operations.",
    f"Although some considered {art} {w} ordinary, its nuanced features revealed a deliberate {r3} philosophy oriented toward long-term reliability.",
    f"If {art} {w} were removed from the workflow, the sequence of steps would have to be redesigned to preserve clarity and safety.",
    f"By aligning {art} {w} with complementary elements, the system established a rhythm that streamlined maintenance and improved traceability.",
    f"Given that {art} {w} often appears in technical documentation, overlooking its specification could introduce inconsistencies downstream.",
    f"When {art} {w} interacts with {r1}s under constrained conditions, small parameter changes can dramatically influence performance.",
    f"Since {art} {w} bridges form and function, it frequently becomes the subject of closer inspection during quality audits.",
    f"If one were to compare {art} {w} with its predecessors, subtle yet consequential improvements in ergonomics would quickly emerge.",
    f"In narratives depicting iterative design, the {w} frequently serves as a tangible example of how constraints shape innovative outcomes.",
  ]


def _category_templates(category: str, w: str) -> List[str]:
  art = _indefinite_article(w)
  if category == 'animal':
    return [
      f"Although the {w} appeared cautious at first, it adapted quickly to unfamiliar surroundings and established a reliable foraging pattern.",
      f"Field observations suggest that {art} {w} modifies its behavior when ambient noise rises above a measurable threshold.",
      f"Because the {w} relies on subtle environmental cues, minor habitat disruptions can cascade into significant changes in movement.",
      f"When researchers introduced a novel stimulus, the {w} exhibited a clear preference hierarchy consistent with previous ethological studies.",
    ]
  if category == 'container':
    return [
      f"By sealing the {w} properly, technicians preserved the sample’s integrity across multiple temperature cycles.",
      f"The {w}, labeled with a tamper-evident indicator, ensured that trace contamination was readily detectable.",
      f"Because the {w} resists impact stress, it is routinely selected for transporting sensitive components.",
      f"After calibrating volumetric markings, the {w} supported precise dosing without additional measuring tools.",
    ]
  if category == 'tool':
    return [
      f"If the {w} is aligned correctly before use, torque is distributed evenly and reduces the risk of fastener damage.",
      f"Technicians prefer the {w} for confined spaces, where handle geometry and leverage are especially critical.",
      f"Because the {w} maintains edge retention under repeated load, productivity remains stable throughout extended shifts.",
      f"After routine maintenance, the {w} consistently met safety benchmarks during post-inspection trials.",
    ]
  if category == 'vehicle':
    return [
      f"Because the {w} optimizes fuel consumption at moderate speeds, route planning often prioritizes steady acceleration.",
      f"In adverse weather, the {w} adjusts traction control parameters to keep lateral slip within acceptable bounds.",
      f"When the {w} approaches an intersection, sensor fusion aggregates camera and radar data to refine its stopping distance.",
      f"After updating the firmware, the {w} demonstrated improved lane-keeping in low-contrast conditions.",
    ]
  if category == 'furniture':
    return [
      f"Because the {w} supports ergonomic posture, users report less strain during prolonged tasks.",
      f"When arranged against natural light, the {w} reduces glare and improves perceived workspace clarity.",
      f"After reinforcing the joints, the {w} withstood cyclical loading without measurable looseness.",
      f"The {w}, paired with adjustable accessories, adapts to diverse anthropometric profiles.",
    ]
  if category == 'garment':
    return [
      f"Because the {w} regulates moisture effectively, comfort remains high during sustained activity.",
      f"When layered beneath an outer shell, the {w} preserves heat without restricting range of motion.",
      f"After multiple wash cycles, the {w} retained its shape and color better than comparable fabrics.",
      f"The {w} incorporates flat seams to minimize abrasion along high-friction contact points.",
    ]
  if category == 'food':
    return [
      f"When the {w} is prepared at lower temperatures, its natural flavors remain more pronounced and balanced.",
      f"Because the {w} combines well with acidic components, chefs use it to create contrast in otherwise rich dishes.",
      f"After proper storage, the {w} maintains texture and aroma without requiring additional preservatives.",
      f"The {w} pairs well with herbs that amplify its subtle sweetness without overpowering the palate.",
    ]
  if category == 'instrument':
    return [
      f"Because the {w} responds linearly to pressure changes, students learn dynamic control more predictably.",
      f"When the {w} is tuned carefully, harmonic overtones align and produce a more coherent timbre.",
      f"After adjusting the bridge, the {w} exhibited improved sustain and articulation across midrange frequencies.",
      f"The {w} benefits from regular maintenance to stabilize intonation during seasonal shifts.",
    ]
  if category == 'device':
    return [
      f"Because the {w} caches frequently accessed data, application latency drops under typical workloads.",
      f"When paired with secure authentication, the {w} reduces the risk of unauthorized access in shared environments.",
      f"After a firmware patch, the {w} handled concurrent connections without observable throughput degradation.",
      f"The {w} uses power-saving modes to extend runtime during off-peak usage.",
    ]
  if category == 'body_part':
    return [
      f"Because the {w} bears repetitive load, targeted conditioning helps prevent overuse injuries.",
      f"When the {w} is aligned correctly, gait efficiency improves and compensatory strain decreases.",
      f"After rehabilitation, the {w} regained functional range with minimal residual stiffness.",
      f"The {w} benefits from coordinated muscle activation to stabilize adjacent joints.",
    ]
  if category == 'plant':
    return [
      f"Because the {w} thrives in partial shade, ground cover remains uniform throughout seasonal transitions.",
      f"When irrigation is moderated, the {w} develops deeper roots and resists drought stress more effectively.",
      f"After pruning, the {w} allocated resources toward new growth and improved canopy density.",
      f"The {w} supports pollinator activity when clustered near complementary species.",
    ]
  if category == 'place':
    return [
      f"Because the {w} centralizes resources, foot traffic patterns become more predictable during peak hours.",
      f"When the {w} integrates clear signage, wayfinding accuracy improves markedly for first-time visitors.",
      f"After retrofitting accessibility features, the {w} accommodated a broader range of users.",
      f"The {w} benefits from natural ventilation strategies that reduce energy consumption.",
    ]
  # abstract y fallback
  return [
    f"Because {w} influences decision-making, stakeholders often reframe objectives to clarify measurable outcomes.",
    f"When teams articulate {w} precisely, ambiguity decreases and cross-functional alignment improves.",
    f"After revisiting the definition of {w}, the project timeline reflected more realistic constraints.",
    f"The concept of {w} interacts with organizational culture in subtle yet consequential ways.",
  ]


def generate_examples(word: str, count: int = 6) -> List[str]:
  ensure_nltk()
  w = (word or '').strip()
  if not w:
    return []

  # 1) Mapear con taxonomía robusta (WordNet + heurísticas)
  mapped_cat, canonical, conf = map_category(w)
  w = canonical or w

  # 2) Fallback a NN ligera si no se pudo mapear
  if not mapped_cat:
    mapped_cat, prob = predict_category(w)

  # 3) Tomar plantillas específicas + genéricas
  pool = _category_templates(mapped_cat or 'abstract', w) + _templates(w)

  # 3) Barajar y seleccionar sin repetidos
  random.shuffle(pool)
  out = []
  seen = set()
  for s in pool:
    if s not in seen and w.lower() in s.lower():
      out.append(s)
      seen.add(s)
    if len(out) >= count:
      break

  # 4) Si faltan, variantes con sinónimos del dominio seleccionado
  if len(out) < count:
    syns = _pick_synonyms(w, limit=10)
    base = f"While the {w} remains central to the task, its interaction with {{syn}} under strict constraints reveals use cases rarely discussed explicitly."
    for syn in syns:
      s = base.format(syn=syn)
      if s not in seen:
        out.append(s)
        seen.add(s)
      if len(out) >= count:
        break
  return out
