"""
Performance test for coreference resolution.
"""
import time
from memory.coref_resolver import resolve_sentences, resolve_sentences_fast, resolve_sentences_legacy


def test_coref_performance():
    """Compare performance of fast vs legacy coreference resolution."""

    # Sample sentences with coreferences
    test_sentences = [
        "John is a software engineer.",
        "He works at a tech company.",
        "The company is based in San Francisco.",
        "He enjoys working on AI projects.",
        "The projects involve machine learning.",
        "John believes AI will transform the industry.",
        "He has been working on this for 5 years.",
        "The experience has been valuable.",
        "John plans to continue his research.",
        "He is passionate about the field.",
    ] * 10  # Repeat 10 times to make it 100 sentences

    print(f"\nTesting with {len(test_sentences)} sentences...")

    # Test fast method
    start = time.time()
    resolved_fast = resolve_sentences_fast(test_sentences)
    fast_time = time.time() - start
    print(f"FastCoref: {fast_time:.3f} seconds")

    # Test legacy method (with smaller dataset to avoid timeout)
    small_test = test_sentences[:20]
    start = time.time()
    resolved_legacy = resolve_sentences_legacy(small_test)
    legacy_time = time.time() - start
    print(f"Legacy (20 sentences): {legacy_time:.3f} seconds")

    # Extrapolate legacy time
    extrapolated_legacy = legacy_time * (len(test_sentences) / len(small_test)) ** 2
    print(f"Legacy (extrapolated for {len(test_sentences)}): {extrapolated_legacy:.3f} seconds")

    speedup = extrapolated_legacy / fast_time if fast_time > 0 else float('inf')
    print(f"Speedup: {speedup:.1f}x faster")

    # Verify resolution worked
    print("\nSample resolved sentences (FastCoref):")
    for i, sent in enumerate(resolved_fast[:3]):
        print(f"  {i+1}. {sent}")

    assert len(resolved_fast) == len(test_sentences)
    assert fast_time < extrapolated_legacy


if __name__ == "__main__":
    test_coref_performance()
