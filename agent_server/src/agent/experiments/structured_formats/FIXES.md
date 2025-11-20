# Structured Format Experiment - Fixes Applied

## Issues Found & Fixed

### 1. S-Expression Parser - Complete Rewrite
**Problem:** 0% success rate due to multiple parsing bugs

**Fixes:**
- ✅ Strip markdown code blocks (`\`\`\`lisp ... \`\`\``)
- ✅ Rewrite tokenizer to handle quoted strings properly (was splitting "14th century" on spaces)
- ✅ Fix dict detection (was returning Python dict syntax instead of parsing it)
- ✅ Fix list detection (single-item lists were becoming dicts)
- ✅ Improve schema generation to show dict format as nested pairs
- ✅ Update prompts to clarify syntax and forbid Python dict literals

**Status:** Now parses successfully! ✅

### 2. XML Parser - Markdown Stripping
**Problem:** 41.9% success (likely due to markdown code blocks)

**Fixes:**
- ✅ Added markdown code block stripping (`\`\`\`xml ... \`\`\``)

**Status:** Should improve success rate

### 3. Experiment Runner - Better Error Logging
**Problem:** Parse errors were silent (only logged at DEBUG level)

**Fixes:**
- ✅ Added WARNING-level logging for parse errors
- ✅ Log first 500 chars of failed LLM response

**Status:** Easier to debug format issues

## Current Results

### After Fixes:
- **JSON:** 100% success (baseline working well)
- **YAML:** 99.5% success (already handled code blocks)
- **XML:** Should improve from 41.9% → likely 80-90%
- **S-Expressions:** Should improve from 0% → likely 70-80%

### F1 Score Issue (NOT A BUG)
**Finding:** Even JSON has low F1 scores (0.03-0.29) on complex tests

**Root Cause:** Ground truth is too specific. LLM extracts facts correctly but uses different:
- Predicate names ("began_in" vs "originated_in")
- Entity role names ("event" vs "movement", "location" vs "place")

**Example:**
- Text: "The Renaissance began in Italy"
- Ground truth: `{predicate: "began_in", entities: {event: "Renaissance", location: "Italy"}}`
- LLM extracts: `{predicate: "originated_in", entities: {movement: "Renaissance", place: "Italy"}}`
- Result: F1=0.00 despite both being semantically correct

**Not a bug - this is expected behavior.** Real experiment would need:
1. Multiple acceptable ground truth variations, OR
2. Semantic similarity matching, OR
3. Simpler test cases with unambiguous extraction

### What Works Well
✅ Pluggable format system
✅ Statistical analysis with multiple runs
✅ Comprehensive metrics tracking
✅ Clear error messages for LLMs
✅ Depth validation
✅ JSON and YAML formats

## Remaining Work

### High Priority
1. **Test XML improvements** - Run experiment to verify code block stripping helps
2. **Test S-Expression improvements** - Verify all fixes work end-to-end
3. **Type coercion** - Add automatic string→int/float conversion if needed

### Medium Priority
4. **Add more formats** - TOML, Python Dict, Markdown Tables, etc.
5. **Better ground truth** - Create test cases with unambiguous answers
6. **Few-shot examples** - Test impact of adding examples to prompts

### Low Priority
7. **Temperature testing** - Test different temperature settings
8. **Model comparison** - Test Q8 vs Q4 quantization
9. **Grammar-constrained generation** - Test Ollama's native format parameter

## How to Verify Fixes

### Quick Test (S-Expression on minimal case):
```bash
uv run python test_format_debug.py
```

Expected: ✅ Success (was ❌ Failed before)

### Full Experiment:
```bash
uv run python -m agent.experiments.structured_formats.run_experiment --num-runs 10
```

Expected improvements:
- S-Exp: 0% → 70-80%
- XML: 41.9% → 80-90%
- JSON/YAML: Remain at ~100%

## Summary

**Major Achievement:** Fixed critical S-Expression parser bugs that caused 100% failure rate.

**Key Learning:** Low F1 scores are not bugs - they reflect semantic mismatch between ground truth and LLM extraction style. This is a test design issue, not a code issue.

**Next Steps:** Re-run experiment to verify fixes, then decide whether to:
1. Improve ground truth flexibility, OR
2. Accept that formats work but LLM extraction varies, OR
3. Use simpler test cases with objective right/wrong answers
