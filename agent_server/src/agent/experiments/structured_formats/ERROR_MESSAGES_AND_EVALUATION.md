# Error Messages & Evaluation System

## Summary

Investigation of:
1. **Error messages sent to LLM during retries** - Are they helpful?
2. **Evaluation scoring** - Why were F1 scores so low?

## 1. Error Messages to LLM

### Finding: Error messages are GOOD ✅

Each format converts Pydantic ValidationErrors into clear, actionable, format-specific guidance.

### Examples:

#### Array Instead of String Error

**JSON:**
```
Field 'facts -> 0 -> entities -> goods' must be a STRING (single value),
not an ARRAY ["silk", "spices"]. Choose one value or create multiple separate objects.
```

**XML:**
```
Field 'facts -> 0 -> entities -> goods' must contain a SINGLE value inside the tag,
not multiple values. Create separate <item> tags for multiple values.
```

**YAML:**
```
Field 'facts -> 0 -> entities -> goods' must be a SINGLE STRING value,
not a list. If you need multiple values, create separate list items with '- '.
```

**S-Expression:**
```
Field 'facts -> 0 -> entities -> goods' must be a SINGLE quoted string: (field "value"),
not multiple values. Use separate list items if needed.
```

#### Null for Required Field

**JSON:**
```
Field 'facts -> 0 -> confidence' cannot be null.
You must provide an actual value (the field is required).
```

**XML:**
```
Tag '<facts -> 0 -> confidence>' cannot be empty.
Provide a value between the opening and closing tags.
```

### Why They're Good:

✅ **Clear and specific** - Points to exact field path
✅ **Format-appropriate** - Uses terminology specific to each format (tags for XML, parentheses for S-Exp)
✅ **Actionable** - Tells LLM exactly what to do
✅ **Much better than raw Pydantic** - Compare to: "Input should be a valid string [type=string_type, input_value=['education', 'governance'], input_type=list]"

### Conclusion

**No changes needed** - Error messages are already well-designed.

---

## 2. Evaluation Scoring

### Problem: Low F1 Scores

Even JSON (which works perfectly) was getting F1 scores like:
- Byzantine trade: 0.29
- Quantum physics: 0.03
- Minimal test: 1.00 ✅

Only the trivial "minimal" test got perfect scores.

### Root Cause: Strict Exact Matching

The evaluation was too strict - it required **exact matches** for:
- Predicate names
- Entity role names
- Entity values

### Example of Problem:

**Text:** "The Renaissance began in Italy"

**Ground truth:**
```python
{
  "predicate": "began_in",
  "entities": {
    "event": "Renaissance",
    "location": "Italy"
  }
}
```

**LLM extracted** (semantically correct!):
```python
{
  "predicate": "originated_in",  # Synonym of "began_in"
  "entities": {
    "movement": "Renaissance",   # Different role name
    "place": "Italy"             # Different role name
  }
}
```

**Strict evaluation result:** F1 = 0.00 ❌

**Problem:** Both are semantically correct, but don't match exactly.

---

## 3. Solution: Flexible Evaluation

Created new `flexible_evaluation.py` module that:

### Features:

1. **Synonym Recognition** - Recognizes common predicate synonyms:
   - `began/started/commenced/originated/initiated`
   - `traded/exchanged/bartered`
   - `ruled/governed/controlled/led`

2. **Fuzzy String Matching** - Uses SequenceMatcher for 80%+ similarity

3. **Partial Credit** - Gives fractional scores instead of binary 0/1:
   - Close matches get 0.8-0.9
   - Extra fields penalized less than missing fields
   - Semantic similarity for predicate names

4. **Entity Value Matching** - Checks if entity values are present regardless of role names

### Results Comparison:

#### Example 1: Synonym Predicates

| Evaluation | Precision | Recall | F1 |
|------------|-----------|--------|-----|
| **Strict** | 0.00 | 0.00 | **0.00** |
| **Flexible** | 1.00 | 0.78 | **0.88** |

**Improvement:** +0.88 F1 points ✅

#### Example 2: Multiple Facts with Variations

| Evaluation | F1 Score |
|------------|----------|
| **Strict** | **0.00** |
| **Flexible** | **0.66** |

**Improvement:** +0.66 F1 points ✅

---

## 4. Integration

### Command-Line Usage:

```bash
# Use flexible evaluation (default, recommended)
uv run python -m agent.experiments.structured_formats.run_experiment

# Use strict exact-match evaluation
uv run python -m agent.experiments.structured_formats.run_experiment --strict-eval
```

### How It Works:

1. **ExperimentRunner** has `use_flexible_eval` parameter (default: True)
2. After successful extraction, calculates quality metrics using chosen evaluation method
3. Flexible evaluation:
   - Recognizes synonyms
   - Fuzzy string matching
   - Partial credit for close matches
4. Strict evaluation:
   - Original exact matching
   - Useful for very simple tests with objective answers

---

## 5. Expected Impact

### Before Flexible Evaluation:
- JSON: F1 = 0.26 (despite 100% parsing success)
- Many semantically correct extractions scored 0.00

### After Flexible Evaluation:
- JSON: F1 = 0.65-0.80 (expected)
- YAML: F1 = 0.60-0.75 (expected)
- S-Exp/XML: Will show meaningful scores instead of 0.00

### When to Use Which:

**Flexible (default):**
- Complex semantic extraction tasks
- Real-world usage
- When multiple valid interpretations exist

**Strict:**
- Simple objective tasks with one right answer
- When you want to measure exact schema compliance
- When testing parsing/formatting (not semantic understanding)

---

## 6. Summary

### Error Messages: ✅ No Changes Needed
- Already clear, specific, and actionable
- Format-appropriate terminology
- Much better than raw Pydantic errors

### Evaluation: ✅ Fixed with Flexible Scoring
- New flexible evaluation recognizes semantic correctness
- Gives partial credit and synonym matching
- Dramatically improves F1 scores for correct extractions
- Integrated as default (can use --strict-eval to disable)

### Next Steps:
1. Re-run experiment with flexible evaluation
2. Expect much more meaningful F1 scores
3. Formats that parse correctly should now show good scores
