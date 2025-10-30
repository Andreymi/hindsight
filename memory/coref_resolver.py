"""
Coreference resolution for memory units.

Ensures every memory unit is self-contained by replacing pronouns
with their actual referents.
"""
import spacy
from typing import List, Dict, Optional
from fastcoref import FCoref
import threading


def get_nlp():
    """Get or load spaCy model."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        raise Exception("spaCy model not found. Run: uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl")


# Global fastcoref model instance (singleton pattern)
_fastcoref_model = None
_fastcoref_lock = threading.Lock()


def get_fastcoref_model():
    """Get or load FastCoref model (singleton pattern)."""
    global _fastcoref_model
    if _fastcoref_model is None:
        with _fastcoref_lock:
            if _fastcoref_model is None:
                # Use CPU by default, can be configured with device='cuda:0' for GPU
                _fastcoref_model = FCoref(device='cpu')
    return _fastcoref_model


def resolve_pronouns_in_text(text: str, context_sentences: List[str] = None) -> str:
    """
    Resolve pronouns to their referents to make text self-contained.

    Strategy:
    1. Identify pronouns in the text
    2. Look for named entities in the same sentence or previous sentences
    3. Replace pronouns with the most likely referent based on:
       - Gender agreement
       - Number agreement (singular/plural)
       - Proximity (closer entities more likely)

    Args:
        text: The sentence to resolve
        context_sentences: Previous sentences for context (optional)

    Returns:
        Text with pronouns resolved
    """
    nlp = get_nlp()

    # Parse the target sentence
    doc = nlp(text)

    # Collect all sentences for context
    all_text = text
    if context_sentences:
        # Add previous sentences for context
        all_text = " ".join(context_sentences) + " " + text

    full_doc = nlp(all_text)

    # Extract entities with their positions
    entities = []
    for ent in full_doc.ents:
        if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT']:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
            })

    # Check if sentence already has a named entity subject
    has_named_subject = False
    for token in doc:
        if token.dep_ in ['nsubj', 'nsubjpass'] and token.pos_ == 'PROPN':
            has_named_subject = True
            break

    # Find pronouns and anaphoric references that need resolution
    pronouns_to_replace = []

    for token in doc:
        # Handle pronouns (he, she, it, they)
        if token.pos_ == 'PRON' and token.dep_ in ['nsubj', 'nsubjpass']:
            # Subject pronouns that need resolution
            pron_lower = token.text.lower()

            # Skip if sentence already has a named subject earlier
            if has_named_subject and any(
                t.dep_ in ['nsubj', 'nsubjpass'] and t.pos_ == 'PROPN' and t.i < token.i
                for t in doc
            ):
                continue

            # Skip if it's already a proper name or demonstrative
            if pron_lower in ['i', 'you', 'we', 'this', 'that', 'these', 'those']:
                continue

            # Find the best entity to replace it with
            referent = find_best_referent(
                pronoun=token,
                entities=entities,
                doc=full_doc
            )

            if referent:
                pronouns_to_replace.append({
                    'pronoun': token,
                    'referent': referent,
                    'start': token.idx,
                    'end': token.idx + len(token.text)
                })

        # Handle definite noun phrases (e.g., "The project")
        elif token.text.lower() == 'the' and token.head.pos_ == 'NOUN':
            # Check if this "the X" phrase is a subject
            if token.head.dep_ in ['nsubj', 'nsubjpass']:
                # Try to find what "the X" refers to
                noun = token.head.text
                # Look for indefinite mentions earlier ("a project", "an organization")
                for ent_token in reversed(list(full_doc)):
                    if ent_token.text.lower() == noun.lower():
                        # Found a matching noun - check if it has indefinite article
                        if any(child.text.lower() in ['a', 'an'] for child in ent_token.children):
                            # Replace "the project" with "the Python project" or similar
                            # Get the full noun phrase
                            descriptors = []
                            for child in ent_token.children:
                                if child.pos_ in ['ADJ', 'PROPN', 'NOUN'] and child.i < ent_token.i:
                                    descriptors.append(child.text)

                            if descriptors:
                                full_phrase = ' '.join(descriptors) + ' ' + noun
                                # Calculate span to replace
                                span_start = token.idx
                                span_end = token.head.idx + len(token.head.text)

                                pronouns_to_replace.append({
                                    'pronoun': token,
                                    'referent': 'the ' + full_phrase,
                                    'start': span_start,
                                    'end': span_end
                                })
                            break

    # Replace pronouns with referents (in reverse order to maintain indices)
    result = text
    for item in reversed(pronouns_to_replace):
        start = item['start']
        end = item['end']
        result = result[:start] + item['referent'] + result[end:]

    return result


def find_best_referent(
    pronoun,
    entities: List[Dict],
    doc
) -> Optional[str]:
    """
    Find the best entity referent for a pronoun.

    Uses:
    - Gender agreement (he/she -> PERSON)
    - Number agreement (singular/plural)
    - Entity type (he/she -> PERSON, it -> ORG/PRODUCT)
    - Proximity (closer entities preferred)
    """
    pron_text = pronoun.text.lower()

    # Determine pronoun properties
    is_singular = pron_text in ['he', 'she', 'it', 'him', 'her']
    is_plural = pron_text in ['they', 'them']
    is_person = pron_text in ['he', 'she', 'him', 'her']
    is_thing = pron_text in ['it']

    # Score each entity
    candidates = []

    for entity in entities:
        score = 0.0

        # Proximity score (entities closer to pronoun are better)
        # Since entities come from context, those appearing later (higher start position) are closer
        proximity_score = entity['start'] / 1000.0  # Normalize by position
        score += proximity_score

        # Type matching
        if is_person and entity['label'] == 'PERSON':
            score += 2.0  # Strong preference for person entities
        elif is_thing and entity['label'] in ['ORG', 'PRODUCT', 'GPE']:
            score += 2.0  # Organizations/products for "it"

        # Recency: prefer entities that appear just before the pronoun
        if entity['end'] < pronoun.idx:
            distance = pronoun.idx - entity['end']
            recency = 1.0 / (1.0 + distance / 100.0)
            score += recency

        candidates.append((entity['text'], score))

    # Return the highest scoring candidate
    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    return None


def resolve_sentences_fast(sentences: List[str]) -> List[str]:
    """
    Fast batch coreference resolution using FastCoref.

    This is significantly faster than the sequential spaCy-based approach:
    - Processes entire document in one pass (O(n) instead of O(n²))
    - Uses efficient batching and neural model
    - Can process 2.8K documents in 25 seconds on GPU

    Args:
        sentences: List of sentences to resolve

    Returns:
        List of resolved sentences (self-contained)
    """
    if not sentences:
        return []

    # Join sentences into a single document for batch processing
    # Add markers to track sentence boundaries
    full_text = " ".join(sentences)

    # Get the fastcoref model
    model = get_fastcoref_model()

    # Predict coreferences in batch
    preds = model.predict(texts=[full_text])

    if not preds or len(preds) == 0:
        # No coreferences found, return original sentences
        return sentences

    # Get the first (and only) result
    result = preds[0]

    # Get clusters as text strings
    clusters = result.get_clusters(as_strings=True)

    if not clusters:
        return sentences

    # Build a replacement map: pronoun -> main referent
    replacements = {}
    for cluster in clusters:
        if len(cluster) < 2:
            continue

        # The first mention is typically the most complete referent
        main_referent = cluster[0]

        # Map all other mentions (pronouns/short references) to the main referent
        for mention in cluster[1:]:
            mention_lower = mention.lower()
            # Only replace if it's likely a pronoun or short reference
            if len(mention.split()) <= 2 and any(
                pron in mention_lower
                for pron in ['he', 'she', 'it', 'they', 'him', 'her', 'them', 'his', 'her', 'their', 'the']
            ):
                replacements[mention] = main_referent

    # Apply replacements to each sentence
    resolved = []
    for sentence in sentences:
        resolved_sentence = sentence
        for mention, referent in replacements.items():
            # Case-insensitive replacement but preserve capitalization context
            if mention in resolved_sentence:
                resolved_sentence = resolved_sentence.replace(mention, referent)
        resolved.append(resolved_sentence)

    return resolved


def resolve_sentences(sentences: List[str]) -> List[str]:
    """
    Resolve pronouns across a list of sentences.

    Uses FastCoref for efficient batch processing.
    Falls back to legacy spaCy method if FastCoref fails.

    Args:
        sentences: List of sentences to resolve

    Returns:
        List of resolved sentences (self-contained)
    """
    try:
        return resolve_sentences_fast(sentences)
    except Exception as e:
        # Fallback to legacy method
        print(f"FastCoref failed ({e}), falling back to spaCy method")
        return resolve_sentences_legacy(sentences)


def resolve_sentences_legacy(sentences: List[str]) -> List[str]:
    """
    Legacy sequential pronoun resolution (slower, O(n²) complexity).

    Kept as fallback in case FastCoref is unavailable or fails.

    Args:
        sentences: List of sentences to resolve

    Returns:
        List of resolved sentences (self-contained)
    """
    resolved = []

    for i, sentence in enumerate(sentences):
        # Use all previous sentences as context
        context = resolved[:i] if i > 0 else []

        # Resolve pronouns in this sentence
        resolved_sentence = resolve_pronouns_in_text(sentence, context)

        resolved.append(resolved_sentence)

    return resolved
