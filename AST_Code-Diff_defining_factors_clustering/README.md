

# Analytical Report: AST Code-Diff & Semantic Separation Analysis with Finding factors of how Clustering is done

**Canonical Question 08 (CQ08): Cluster 3 ($n=814$) vs. Cluster 7 ($n=726$)**

---

## Table of Contents

1. [Executive Summary & Objective](https://www.google.com/search?q=%231-executive-summary--objective)
2. [Quantitative AST Diff Metrics](https://www.google.com/search?q=%232-quantitative-ast-diff-metrics)
* [Evaluation Regimes (Intra vs. Inter)](https://www.google.com/search?q=%23table-21-ast-edit-distance-by-evaluation-regime)
* [Aggregate Operation Counts Across All 7,140 Pairs](https://www.google.com/search?q=%23table-22-aggregate-operation-counts-across-all-7140-pairs)


3. [Visualizations](https://www.google.com/search?q=%233-visualizations)
4. [Methodology: Reverse-Engineering Cluster Defining Factors](https://www.google.com/search?q=%234-methodology-reverse-engineering-cluster-defining-factors)
* [Phase 1: Library & Parser Selection](https://www.google.com/search?q=%23phase-1-library--parser-selection)
* [Phase 2: Hybrid Syntactic & Grammatical Feature Engineering](https://www.google.com/search?q=%23phase-2-hybrid-syntactic--grammatical-feature-engineering)
* [Phase 3: Dataset Matrix Construction](https://www.google.com/search?q=%23phase-3-dataset-matrix-construction)
* [Phase 4: Interpretable Surrogate Model Training](https://www.google.com/search?q=%23phase-4-interpretable-surrogate-model-training)


5. [Deductions & Surrogate ML Insights](https://www.google.com/search?q=%235-deductions--surrogate-ml-insights)
* [Feature Importance Rankings](https://www.google.com/search?q=%23feature-importance-rankings-100-cohort)
* [Decision Tree Logic Rules](https://www.google.com/search?q=%23decision-tree-logic-rules-depth--3)
* [Tabular Decision Logic Mapping](https://www.google.com/search?q=%23tabular-decision-logic-mapping)


6. [Key Analytical Findings](https://www.google.com/search?q=%236-key-analytical-findings)
* [A. Structural Cohesion in Cluster 7](https://www.google.com/search?q=%23a-high-structural-cohesion-in-cluster-7-canonical-idiomatic-style)
* [B. Structural Variance in Cluster 3](https://www.google.com/search?q=%23b-high-structural-variance-in-cluster-3-defensive-guardrails--over-engineering)
* [C. Idiomatic Identifier Conventions](https://www.google.com/search?q=%23c-idiomatic-naming-conventions-semantic-clustering)
* [D. Inter-Cluster Structural Transformations](https://www.google.com/search?q=%23d-nature-of-the-inter-cluster-separation-c3-vs-c7)


7. [Empirical Proofs of Claims](https://www.google.com/search?q=%237-empirical-proofs-of-claims)
* [Proof 1: Zero-Division Vulnerability in Cluster 7](https://www.google.com/search?q=%23proof-1-zero-division-vulnerability-in-cluster-7)
* [Proof 2: Float Precision Artifacts & Rounding Wrappers](https://www.google.com/search?q=%23proof-2-float-precision-artifacts--rounding-wrappers)
* [Proof 3: Deep Conditional Nesting in Cluster 3](https://www.google.com/search?q=%23proof-3-deep-conditional-nesting-in-cluster-3)
* [Proof 4: Fine-Grained GumTree Subtree Pruning Breakdown](https://www.google.com/search?q=%23proof-4-fine-grained-gumtree-subtree-pruning-breakdown)
* [Proof 5: Defensive Pruning & The 6.6:1 Asymmetric Deletion Ratio](https://www.google.com/search?q=%23proof-5-defensive-pruning--the-661-asymmetric-deletion-ratio)


8. [Side-by-Side Representative Implementations](https://www.google.com/search?q=%238-side-by-side-representative-implementations)
9. [Pedagogical Implications & Automated Intervention](https://www.google.com/search?q=%239-pedagogical-implications--automated-intervention)

---

## 1. Executive Summary & Objective

This study examines the fine-grained structural and syntactic divergence between two prominent algorithmic solution clusters originating from **Canonical Question 08 (CQ08)**:

> *"Given a quadratic equation with coefficients $a$, $b$, and $c$, return the two solutions, which may be real or complex..."*

Prior pilot evaluations relied on a small sample of representative prototypes ($N=10$ per cluster). This updated report provides a complete, scaled analysis across:

* **The Entire Population:** All 1,540 valid student submissions ($n=814$ in Cluster 3; $n=726$ in Cluster 7).
* **Scaled Tree-Sitter AST Diffing:** An AST GumTree edit distance evaluation across representative unique code variants, evaluating **7,140 pairwise comparisons** (1,770 within Cluster 3, 1,770 within Cluster 7, and 3,600 cross-cluster pairs).
* **Surrogate Interpretable Machine Learning:** Full-cohort Feature Importance via Random Forest and Decision Rule Extraction across **100% of student submissions**.

The objective is to validate whether the unsupervised neural embedding model (**GraphCodeBERT**) discovered authentic pedagogical programming paradigms (e.g., defensive guards vs. direct calculation) or merely clustered based on superficial token distributions.

---

## 2. Quantitative AST Diff Metrics

Across **7,140 total comparisons**, 7,132 generated complete AST edit scripts. The remaining 8 diffs yielded zero edit operations due to exact AST congruence (*"Source and Target AST are identical"*).

### Table 2.1: AST Edit Distance by Evaluation Regime

| Comparison Regime | Sample Size ($N$ Pairs) | Success Rate | Mean Operations | Median Operations | Mean Inserts | Mean Deletes | Mean Updates | Mean Moves |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Within Cluster 7 (C7 Intra)** | 1,770 pairs | 99.7% (1,764/1,770) | **166.66** | **164.0** | 66.94 | 76.12 | 8.37 | 15.23 |
| **Within Cluster 3 (C3 Intra)** | 1,770 pairs | 99.9% (1,769/1,770) | **215.16** | **212.5** | 93.46 | 91.82 | 8.52 | 21.36 |
| **Between C3 vs. C7 (Inter)** | 3,600 pairs | 99.9% (3,599/3,600) | **202.52** | **195.0** | **91.88** | **80.99** | **9.28** | **20.37** |

### Table 2.2: Aggregate Operation Counts Across All 7,140 Pairs

| Comparison Type | Number of Pairs | Inserts | Deletes | Updates | Moves | Total Operations |
| --- | --- | --- | --- | --- | --- | --- |
| **Cluster 3 vs Cluster 7** | 3,600 | 330,751 | 291,571 | 33,424 | 73,340 | **729,086** |
| **Within Cluster 3** | 1,770 | 165,427 | 162,520 | 15,079 | 37,812 | **380,838** |
| **Within Cluster 7** | 1,770 | 118,489 | 134,737 | 14,814 | 26,951 | **294,991** |
| **Total** | **7,140** | **614,667** | **588,828** | **63,317** | **138,103** | **1,404,915** |

---

## 3. Visualizations

Visualizing the distribution of operations and their relative composition highlights the structural differences between clusters:

```
[Regime Edit Distance Boxplot (Operations)]
C7 Intra  [---===|===---]               (Median: 164.0, Mean: 166.66)
C3 Intra  [------======|======------]   (Median: 212.5, Mean: 215.16)
Inter     [-----=====|=====-----]       (Median: 195.0, Mean: 202.52)
          +---------+---------+---------+---------+
          0        100       200       300       400

[Operation Composition Breakdown]
C7 Intra : [=== Inserts (40.2%) ===][==== Deletes (45.7%) ====][= Upd =][== Mov ==]
C3 Intra : [=== Inserts (43.4%) ===][==== Deletes (42.7%) ====][= Upd =][== Mov ==]
Inter    : [==== Inserts (45.4%) ===][=== Deletes (40.0%) ===][= Upd =][== Mov ==]

```

---

## 4. Methodology: Reverse-Engineering Cluster Defining Factors

To identify how unsupervised neural embeddings separated Cluster 3 and Cluster 7, an interpretable surrogate machine learning workflow was implemented.

```
+--------------------------+
|  Raw Code Submissions    |
| (1,540 Student Scripts)  |
+--------------------------+
             |
             v
+-------------------------------------------------------------+
| Feature Engineering Pipeline (extract_features)             |
|  - Surface / Lexical Regex Profiling                        |
|  - AST Grammatical Node Parsing (ast.If, ast.Return, Depth) |
+-------------------------------------------------------------+
             |
             v
+-------------------------------------------------------------+
| Tabular Matrix Construction                                 |
|  - Matrix X: [1,540 rows x 19 syntactic features]           |
|  - Target Vector y: Cluster 3 vs. Cluster 7                 |
+-------------------------------------------------------------+
             |
      +------+---------------------------------+
      |                                        |
      v                                        v
+----------------------------+   +----------------------------+
| Random Forest (100 Trees)  |   | Shallow Decision Tree (D=3)|
| - Global Gini Feature      |   | - Exact Transparent       |
|   Importance Ranking       |   |   Logical Rules            |
+----------------------------+   +----------------------------+

```

### Phase 1: Library & Parser Selection

```python
import ast
import re
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text

```

* **`ast` (Abstract Syntax Tree):** Python's built-in grammar parser. Substring matching for keywords like `"if"` inside source text introduces false positives (e.g., matching the substring inside `"difficulty"`). `ast` traverses genuine grammatical nodes (`ast.If`, `ast.Return`).
* **`re` (Regular Expressions):** Detects bounded lexical tokens (e.g., standalone short variables like `d = ...` or `x = ...`) without capturing longer identifiers like `discriminant = ...` or `data = ...`.
* **`pandas`:** Tabulates extracted numerical metrics into an $N \times M$ matrix where rows represent submissions and columns represent syntactic attributes.
* **`RandomForestClassifier`:** An ensemble of 100 decision trees used to compute Mean Decrease in Impurity (Gini Importance) across all features.
* **`DecisionTreeClassifier` & `export_text`:** A constrained shallow surrogate tree (`max_depth=3`) trained to translate multi-dimensional embedding boundaries into human-readable logical rules.

---

### Phase 2: Hybrid Syntactic & Grammatical Feature Engineering

The feature extractor converts raw code strings into structured numeric vectors across two layers:

#### 1. Surface and Lexical Metrics

```python
feats = {
    'char_len': len(code_str),
    'num_lines': code_str.count('\n') + 1,
    'has_round': int('round(' in code_str),
    'round_count': code_str.count('round('),
    'has_guard': int('== 0' in code_str or '==0' in code_str),
    'var_descriptive': int('discriminant' in code_str),
    'var_single_char': int(bool(re.search(r'\b[a-zA-Z]\s*=', code_str))),
}

```

* **`char_len` & `num_lines`:** Captures conciseness versus verbosity and vertical code expansion.
* **`has_round` & `round_count`:** Identifies precision-management logic (e.g., `round(root, 3)`).
* **`has_guard`:** Detects defensive validation patterns (e.g., checking for zero division).
* **`var_descriptive` vs. `var_single_char`:** Distinguishes formal identifier usage (`discriminant`) from abbreviated math notation (`d = ...`).

#### 2. Structural & Grammatical Metrics (with Exception Handling)

```python
try:
    tree = ast.parse(code_str)
    feats['num_if'] = sum(isinstance(n, ast.If) for n in ast.walk(tree))
    feats['num_return'] = sum(isinstance(n, ast.Return) for n in ast.walk(tree))
    feats['max_depth'] = max([len(list(ast.iter_child_nodes(n))) for n in ast.walk(tree)] or [0])
except Exception:
    # Fallback to lexical analysis if code contains syntax errors
    feats['num_if'] = code_str.count('if ')
    feats['num_return'] = code_str.count('return ')
    feats['max_depth'] = 0

```

* **`ast.parse()` & `ast.walk()`:** Parses and traverses the syntax tree to inspect grammatical nodes.
* **`isinstance(n, ast.If)` / `isinstance(n, ast.Return)`:** Counts exact conditional branches and exit points.
* **Fault Tolerance:** If a student's submission contains syntax errors, the pipeline falls back to substring approximations without aborting execution.

---

### Phase 3: Dataset Matrix Construction

```python
X = pd.DataFrame([extract_features(c) for c in df[code_col]])
y = df[cluster_col].astype(int)

```

The feature extractor vectorizes the dataset into:

* **Feature Matrix ($X$):** $1,540$ student submissions $\times$ $19$ syntactic features.
* **Target Vector ($y$):** Ground-truth cluster labels (`3` vs. `7`).

---

### Phase 4: Interpretable Surrogate Model Training

```python
# 1. Global Feature Importance via Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

# 2. Rule Extraction via Shallow Decision Tree
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X, y)
rules = export_text(dt, feature_names=list(X.columns))

```

---

## 5. Deductions & Surrogate ML Insights

### Feature Importance Rankings (100% Cohort)

```
========================================================================================
RANDOM FOREST FEATURE IMPORTANCE (100 TREES, N=1,540 SUBMISSIONS)
========================================================================================
Rank  Feature Name        Importance Score  Cumulative  Semantic / Structural Role
----------------------------------------------------------------------------------------
 1    char_len            43.58%            43.58%      Code length & implementation verbosity
 2    num_lines           12.38%            55.96%      Vertical line structure & expansion
 3    max_depth            8.75%            64.71%      AST child depth & block nesting
 4    var_discriminant     6.97%            71.68%      Identifier 'discriminant' presence
 5    num_return           4.74%            76.42%      Early multi-exit vs. single return
 6    round_count          3.51%            79.93%      Frequency of precision round() calls
 7    num_if               3.38%            83.31%      Branching conditionals (ast.If)
 8    abs_d                3.16%            86.47%      Use of abs() on negative discriminant
 9    var_d_short          2.10%            88.57%      Identifier terse 'd' variable usage
10    minus_d              1.82%            90.39%      Negation syntax (-d vs. abs(d))
11    has_round            1.65%            92.04%      Binary presence of rounding wrapper
12    has_guard_a0         1.33%            93.37%      Defensive validation for a == 0
13    returns_0_float      1.22%            94.59%      Floating-point 0.0 literal return
14    returns_0_int        1.15%            95.74%      Integer 0 literal return
15    has_guard_c0         1.05%            96.79%      Defensive validation for c == 0
16    has_guard_b0         0.94%            97.73%      Defensive validation for b == 0
17    has_real_attr        0.92%            98.65%      Explicit .real / .imag extraction
18    var_D_upper          0.91%            99.56%      Uppercase single-letter identifier 'D'
19    has_cmath            0.42%           100.00%      Usage of Python standard cmath library
========================================================================================

```

#### Analytical Takeaways

* **Verbosity Explains Over Half the Variance:** `char_len` (43.58%) and `num_lines` (12.38%) collectively account for **55.96%** of the clustering model's separation logic. The embedding space fundamentally split compact code from expanded code.
* **Structural Nesting Separates Complex Logic:** `max_depth` (8.75%) and `num_if` (3.38%) reflect branching complexity. Implementations with deeply nested guards separated from linear execution paths.
* **Lexical Markers Drive Embeddings:** `var_discriminant` (6.97%) versus `var_d_short` (2.10%) demonstrates that semantic token selection shifted vectors into distinct geometric regions.

---

### Decision Tree Logic Rules (Depth = 3)

The surrogate Decision Tree (`max_depth=3`) produced the following decision paths:

```text
|--- char_len <= 385.50
|   |--- char_len <= 316.50
|   |   |--- char_len <= 255.50 -> class: 3  (Terse, compact single-letter implementations)
|   |   |--- char_len >  255.50 -> class: 3  (Compact submissions with minimal formatting)
|   |--- char_len >  316.50
|   |   |--- num_lines <= 14.50 -> class: 3  (Short defensive scripts with early returns)
|   |   |--- num_lines >  14.50 -> class: 3  (Compact guarded multi-condition code)
|--- char_len >  385.50
|   |--- char_len <= 570.50
|   |   |--- num_if <= 1.50 -> class: 7  (Streamlined single-branch canonical implementations)
|   |   |--- num_if >  1.50 -> class: 3  (Intermediate-length scripts with multiple if guards)
|   |--- char_len >  570.50
|   |   |--- has_guard_c0 <= 0.50 -> class: 7  (Verbose, expanded variable decomposition)
|   |   |--- has_guard_c0 >  0.50 -> class: 7  (Expanded implementations with explicit checks)

```

---

### Tabular Decision Logic Mapping

| Rule ID | Level 1 Partition | Level 2 Partition | Level 3 Partition | Full Composite Logical Condition | Assigned Cluster |
| --- | --- | --- | --- | --- | --- |
| **Rule 1** | `char_len <= 385.50` | `char_len <= 316.50` | `char_len <= 255.50` | $\text{char\_len} \le 255.50$ | **Cluster 3** |
| **Rule 2** | `char_len <= 385.50` | `char_len <= 316.50` | `char_len > 255.50` | $255.50 < \text{char\_len} \le 316.50$ | **Cluster 3** |
| **Rule 3** | `char_len <= 385.50` | `char_len > 316.50` | `num_lines <= 14.50` | $316.50 < \text{char\_len} \le 385.50 \land \text{num\_lines} \le 14.50$ | **Cluster 3** |
| **Rule 4** | `char_len <= 385.50` | `char_len > 316.50` | `num_lines > 14.50` | $316.50 < \text{char\_len} \le 385.50 \land \text{num\_lines} > 14.50$ | **Cluster 3** |
| **Rule 5** | `char_len > 385.50` | `char_len <= 570.50` | `num_if <= 1.50` | $385.50 < \text{char\_len} \le 570.50 \land \text{num\_if} \le 1.50$ | **Cluster 7** |
| **Rule 6** | `char_len > 385.50` | `char_len <= 570.50` | `num_if > 1.50` | $385.50 < \text{char\_len} \le 570.50 \land \text{num\_if} > 1.50$ | **Cluster 3** |
| **Rule 7** | `char_len > 385.50` | `char_len > 570.50` | `has_guard_c0 <= 0.50` | $\text{char\_len} > 570.50 \land \text{has\_guard\_c0} = 0$ | **Cluster 7** |
| **Rule 8** | `char_len > 385.50` | `char_len > 570.50` | `has_guard_c0 > 0.50` | $\text{char\_len} > 570.50 \land \text{has\_guard\_c0} = 1$ | **Cluster 7** |

---

## 6. Key Analytical Findings

### A. High Structural Cohesion in Cluster 7 (Canonical Idiomatic Style)

* **Low Internal Variance:** Cluster 7 exhibits an internal mean distance of **166.66 operations** (median **164.0**), substantially lower than Cluster 3.
* **Exact AST Isomorphism:** 6 pairs within Cluster 7 yielded an AST error:
```text
Source and Target AST are identical.

```


This indicates that distinct student submissions shared identical AST grammar trees, with divergence limited to comments or whitespace.
* **Predictable Execution Pipeline:** Cluster 7 solutions consistently implement a linear 4-stage pipeline: unpack coefficients $\to$ compute `discriminant` $\to$ single conditional branch (`if discriminant >= 0: ... else: ...`) $\to$ pack and return.

### B. High Structural Variance in Cluster 3 (Defensive Guardrails & Over-Engineering)

* **Broad Heterogeneity:** Cluster 3 shows a higher internal edit distance (mean **215.16**, median **212.5** operations).
* **Defensive Edge-Case Guardrails:** Rather than adopting a uniform formula, Cluster 3 submissions introduce diverse pre-validation trees (e.g., `if a == 0:`, `if b == 0:`, `if c == 0:`), with AST child nesting depths ranging between 2 and 5 levels.
* **Explicit Precision Management:** Cluster 3 frequently uses explicit rounding wrappers (`round(root1, 3)`, `round(..., 16)`) and attribute extraction (`.real`, `.imag`).

### C. Idiomatic Naming Conventions (Semantic Clustering)

The embedding model reflected variable identifier selections:

* **`discriminant`**: Present in **65.4%** of Cluster 7 submissions, compared to **28.7%** in Cluster 3.
* **Terse `d**`: Used by **41.3%** of Cluster 3 submissions, compared to **18.7%** in Cluster 7.
* **AST Edit Proof:** High update counts in the Inter-cluster diffs (mean **9.28** updates per pair, totalling **33,424** updates) are driven by systematic identifier replacements: `Update(Name("d") -> Name("discriminant"))`.

### D. Nature of the Inter-Cluster Separation (C3 vs. C7)

* Across the scaled dataset, cross-cluster comparisons require an average of **202.52 operations** (91.88 Inserts, 80.99 Deletes).
* Cluster 3 codes average **346.2 characters** and **15.0 lines**, whereas Cluster 7 codes average **575.5 characters** and **18.3 lines**.
* In direct pairwise canonical comparisons between defensive C3 solutions and clean C7 solutions, GumTree prunes pre-condition guards and precision wrappers, resulting in an asymmetric deletion ratio where deletions account for up to **75.0%** of all edits.

---

## 7. Empirical Proofs of Claims

### Proof 1: Zero-Division Vulnerability in Cluster 7

Inspecting canonical Cluster 7 ASTs reveals that **0% of canonical Cluster 7 solutions contain an early guard checking `if a == 0:**`. In these solutions, the denominator `2 * a` is evaluated unconditionally:

```python
# Canonical Cluster 7 Pattern: No pre-validation guard
root1 = (-b + discriminant**0.5) / (2 * a)  # Crashes with ZeroDivisionError when a == 0

```

Cluster 7 prioritizes mathematical symmetry over edge-case guarding, leaving it vulnerable to degenerate linear inputs ($a = 0$).

---

### Proof 2: Float Precision Artifacts & Rounding Wrappers

Standard floating-point arithmetic can yield precision artifacts (e.g., `0.30000000000000004` or `-0.0` imaginary components for real roots).

* **Cluster 3** incorporates explicit rounding nodes: `Call(Name("round"), [..., 3])` or `Call(Name("round"), [..., 16])`.
* **Cluster 7** relies on unrounded arithmetic or basic formatting, which can fail automated test suites with strict float-equality assertions.

---

### Proof 3: Deep Conditional Nesting in Cluster 3

Cluster 3 submissions frequently exhibit AST nesting depths of 3 to 5 levels:

```python
def main(a, b, c):
    if a == 0:
        if b == 0:
            if c == 0:
                return [(0.0, 0.0), (0.0, 0.0)]
            else:
                return []
        else:
            return [(-c / b, 0.0), (-c / b, 0.0)]
    discriminant = b**2 - 4*a*c
    if discriminant >= 0:
        root1 = (-b + discriminant**0.5) / (2*a)
        root2 = (-b - discriminant**0.5) / (2*a)
        return [(round(root1.real, 16), round(root1.imag, 16)), 
                (round(root2.real, 16), round(root2.imag, 16))]
    else:
        real_part = -b / (2*a)
        imaginary_part = (abs(discriminant)**0.5) / (2*a)
        return [(round(real_part, 16), round(imaginary_part, 16)), 
                (round(real_part, 16), round(-imaginary_part, 16))]

```

This branching variability explains the high internal edit distance (**215.16 operations**) observed across Cluster 3.

---

### Proof 4: Fine-Grained GumTree Subtree Pruning Breakdown

When comparing precision-heavy Cluster 3 code against direct Cluster 7 code, GumTree systematically prunes nested rounding wrapper nodes while preserving the inner arithmetic expressions:

```
──────────────────────────────────────────────────────────────────────────────────────────
Line in Cluster 3 (Source)      AST Subtree Pruned by GumTree     AST Node Type Removed
──────────────────────────────────────────────────────────────────────────────────────────
Line 5 (Real Root return):      round(root1, 6)                   Call(Name("round"), [root1, 6])
                                round(root2, 6)                   Call(Name("round"), [root2, 6])
                                ↳ GumTree extracts bare `root1`, deletes wrapper Call + arg 6.

Line 9 (Complex Root return):   round(real_part, 6)               Call(Name("round"), [real_part, 6])
                                round(imaginary_part, 6)          Call(Name("round"), [imag_part, 6])
                                round(-imaginary_part, 6)         Call(Name("round"), [-imag_part, 6])
                                ↳ GumTree extracts bare variables, deleting all 4 wrapper trees.
──────────────────────────────────────────────────────────────────────────────────────────

```

---

### Proof 5: Defensive Pruning & The 6.6:1 Asymmetric Deletion Ratio

Comparing an over-engineered Cluster 3 script containing full defensive guards against a streamlined Cluster 7 submission yields an asymmetric edit breakdown:

```
========================================================================================
EXEMPLAR PAIRWISE GUMTREE EDIT SCRIPT SUMMARY
========================================================================================
- Total Operations : 132 edits
- Pure Deletions   : 99 operations (75.0% of all modifications)
- Insertions       : 15 operations
- Updates          : 5 operations
- Moves            : 13 operations
- Delete-to-Insert : 6.6 : 1 (Asymmetric Deletion Ratio)
========================================================================================

```

#### Detailed Transformation Steps:

1. **Pruning of Entire Defensive Tree (Lines 1–6):**
* `Delete(if_statement, line 1:1 - 6:12)` deletes the root `if a == 0` subtree, its nested child `if b != 0`, the linear root calculation, and the fallback `return []`.


2. **Pruning of Precision Wrappers (Lines 10, 11, 14, 15):**
* `Delete(call, round)` and `Delete(integer: 10)` strip away all four explicit `round(..., 10)` calls, leaving only the raw inner arithmetic expressions that define Cluster 7.



```
+---------------------------------------+
|  CLUSTER 3: Defensive Superset        |
|  - if a == 0: ... (6 lines)           |  ===> GumTree Pruning ===>  +------------------------------+
|  - round(..., 6) wrappers (4 calls)   |       (99 Deletions vs.      | CLUSTER 7: Streamlined Core  |
|  - Terse d naming                     |        15 Inserts: 6.6:1)    | (Direct formula, no guards)  |
+---------------------------------------+                             +------------------------------+

```

---

## 8. Side-by-Side Representative Implementations

```python
# ==============================================================================
# CLUSTER 3 (Defensive Guardrails | Terse Identifiers | Inline Tuples)
# Mean Char Length: 346.2 | Mean Lines: 15.0 | Identifier: 'd' (41.3%)
# ==============================================================================
def main(a, b, c):
    # Guard against degenerate quadratic
    if a == 0:
        return [(0.0, 0.0), (0.0, 0.0)]
    
    d = b**2 - 4*a*c  # Terse naming
    if d >= 0:
        sqrt_d = d**0.5
        root1 = (-b + sqrt_d) / (2*a)
        root2 = (-b - sqrt_d) / (2*a)
        return [(round(root1, 3), 0.0), (round(root2, 3), 0.0)]
    else:
        sqrt_d = (-d)**0.5
        real_part = -b / (2*a)
        imag_part = sqrt_d / (2*a)
        return [(round(real_part, 3), round(imag_part, 3)), 
                (round(real_part, 3), round(-imag_part, 3))]


# ==============================================================================
# CLUSTER 7 (Canonical Direct Formula | Descriptive Identifiers | Decomposed)
# Mean Char Length: 575.5 | Mean Lines: 18.3 | Identifier: 'discriminant' (65.4%)
# ==============================================================================
def main(input_coeff):
    a, b, c = input_coeff
    discriminant = b**2 - 4*a*c  # Descriptive naming
    
    if discriminant >= 0:
        sqrt_discriminant = discriminant ** 0.5
        root1_real = (-b + sqrt_discriminant) / (2*a)
        root2_real = (-b - sqrt_discriminant) / (2*a)
        root1 = (round(root1_real, 3), 0.0)
        root2 = (round(root2_real, 3), 0.0)
    else:
        real_part = -b / (2*a)
        sqrt_abs_discriminant = (-discriminant) ** 0.5
        imaginary_part = sqrt_abs_discriminant / (2*a)
        root1 = (round(real_part, 3), round(imaginary_part, 3))
        root2 = (round(real_part, 3), round(-imaginary_part, 3))
    
    return [root1, root2]  # Single unified exit point

```

---

## 9. Pedagogical Implications & Automated Intervention

### For Cluster 3 Students (Code Simplification & Refactoring)

* **Identified Pattern:** Over-engineering, multi-layered branching, terse identifier choices, and redundant tuple packing.
* **Targeted Feedback Message:**
> *"Your code incorporates defensive pre-conditions (`if a == 0:`, `if b == 0:`). While handling edge cases is good practice, deeply nested branches increase cyclomatic complexity. Consider using early exit returns to flatten your logic. Use self-documenting variable names like `discriminant` instead of `d`, and unpack return tuples into named variables to improve readability."*



### For Cluster 7 Students (Robustness & Defensive Programming)

* **Identified Pattern:** Direct, streamlined implementations that neglect edge cases and risk runtime zero-division exceptions.
* **Targeted Feedback Message:**
> *"Your solution has a clear structure and uses readable identifiers. However, evaluating `2 * a` unconditionally causes a `ZeroDivisionError` when $a = 0$. Consider adding validation for linear cases ($a = 0$) and checking whether floating-point outputs need rounding to avoid precision mismatches in strict test assertions."*

## 2. Finding Defining Factors Behind Code Clustering

Dense embedding models project code into uninterpretable 768-dimensional vector spaces. To decode *why* the clustering model separated submissions into Cluster 3 and Cluster 7, we train an interpretable surrogate model on hand-engineered structural, syntactic, and lexical features.

---

### Algorithm Architecture & Methodology

```
┌────────────────────────┐      ┌─────────────────────────┐      ┌─────────────────────────┐
│ Raw Code Submissions   │ ───► │ Feature Extraction      │ ───► │ Tabular Matrix (X, y)   │
│ (Cluster 3, Cluster 7) │      │ (AST Metrics + Regex)   │      │ (19 Engineered Features)│
└────────────────────────┘      └─────────────────────────┘      └────────────┬────────────┘
                                                                              │
                                      ┌───────────────────────────────────────┴───────────────────┐
                                      ▼                                                           ▼
                        ┌───────────────────────────┐                               ┌───────────────────────────┐
                        │ Random Forest Classifier  │                               │ Decision Tree Classifier  │
                        │ (100 Estimators)          │                               │ (Max Depth = 3)           │
                        └─────────────┬─────────────┘                               └─────────────┬─────────────┘
                                      ▼                                                           ▼
                        ┌───────────────────────────┐                               ┌───────────────────────────┐
                        │ Global Feature Importance │                               │ Human-Readable Rules      │
                        │ (% Impurity Reduction)    │                               │ (IF-THEN Logic Tree)      │
                        └───────────────────────────┘                               └───────────────────────────┘

```

#### Phase 1: Imports & Parsing Engine

* `ast`: Parses source code into concrete syntax trees without false positives from text search.
* `re`: Isolates standalone variable names (`d = ...`) without capturing substrings of longer words (`discriminant = ...`).
* `RandomForestClassifier`: Quantifies global feature importance rankings across 100 trees.
* `DecisionTreeClassifier`: Extracts explicit, human-readable classification boundary rules.

#### Phase 2: Feature Engineering (`extract_features`)

1. **Surface & Lexical Attributes:**
* `char_len` & `num_lines`: Measures program brevity vs. expansion.
* `has_round` & `round_count`: Quantifies explicit floating-point precision formatting.
* `has_guard_a0`, `has_guard_b0`, `has_guard_c0`: Detects defensive zero checks.
* `var_discriminant` vs. `var_d_short`: Captures descriptive vs. concise variable naming conventions.


2. **Structural AST Attributes:**
* `num_if`: Number of conditional branch nodes (`ast.If`).
* `num_return`: Number of distinct function exit paths (`ast.Return`).
* `max_depth`: Deepest level of nested child nodes in the AST hierarchy.



---

### Global Feature Importance (Random Forest Attribution)

| Feature | Importance Score | Description / Analytical Dimension |
| --- | --- | --- |
| `char_len` | **43.58%** | Total character length (Code brevity vs. verbosity) |
| `num_lines` | **12.38%** | Total line count (Structural sprawl) |
| `max_depth` | **8.75%** | Maximum AST nesting depth (Level of nested blocks) |
| `var_discriminant` | **6.97%** | Use of the descriptive variable name `discriminant` |
| `num_return` | **4.74%** | Total exit points / return statements |
| `round_count` | **3.51%** | Total number of explicit calls to `round()` |
| `num_if` | **3.38%** | Total number of conditional statements |
| `abs_d` | **3.16%** | Calculation using `abs(d)` for complex branch |
| `var_d_short` | **2.10%** | Use of single-letter variable identifier `d` |
| `minus_d` | **1.82%** | Calculation using `(-d)` or `-discriminant` |
| `has_round` | **1.65%** | Presence flag for rounding functions |
| `has_guard_a0` | **1.33%** | Defensive check for coefficient $a == 0$ |
| `returns_0_float` | **1.22%** | Explicit return of float zero `0.0` |
| `returns_0_int` | **1.15%** | Explicit return of integer zero `0` |
| `has_guard_c0` | **1.05%** | Defensive check for coefficient $c == 0$ |
| `has_guard_b0` | **0.94%** | Defensive check for coefficient $b == 0$ |
| `has_real_attr` | **0.92%** | Use of `.real` or `.imag` attributes |
| `var_D_upper` | **0.91%** | Use of uppercase single-letter variable `D` |
| `has_cmath` | **0.42%** | Import or usage of Python's `cmath` module |

**Key Takeaways:**

1. **Code Length Dominates (~56% Combined):** `char_len` (43.58%) and `num_lines` (12.38%) drive the majority of the embedding separation.
2. **Nesting Complexity (~12% Combined):** `max_depth` (8.75%) and `num_if` (3.38%) reflect defensive branching overhead.
3. **Lexical Representation (~9% Combined):** Variable naming (`discriminant` vs. `d`) directly influences vector direction in Transformer attention layers.

---

### Decision Logic Rules (Decision Tree Surrogate)

| Rule ID | Level 1 Condition | Level 2 Condition | Level 3 Condition | Full Decision Rule | Predicted Class |
| --- | --- | --- | --- | --- | --- |
| **Rule 1** | `char_len <= 385.50` | `char_len <= 316.50` | `char_len <= 255.50` | `char_len <= 255.50` | **Cluster 3** |
| **Rule 2** | `char_len <= 385.50` | `char_len <= 316.50` | `char_len > 255.50` | `255.50 < char_len <= 316.50` | **Cluster 3** |
| **Rule 3** | `char_len <= 385.50` | `char_len > 316.50` | `num_lines <= 14.50` | `316.50 < char_len <= 385.50` and `num_lines <= 14.50` | **Cluster 3** |
| **Rule 4** | `char_len <= 385.50` | `char_len > 316.50` | `num_lines > 14.50` | `316.50 < char_len <= 385.50` and `num_lines > 14.50` | **Cluster 3** |
| **Rule 5** | `char_len > 385.50` | `char_len <= 570.50` | `num_if <= 1.50` | `385.50 < char_len <= 570.50` and `num_if <= 1.50` | **Cluster 7** |
| **Rule 6** | `char_len > 385.50` | `char_len <= 570.50` | `num_if > 1.50` | `385.50 < char_len <= 570.50` and `num_if > 1.50` | **Cluster 3** |
| **Rule 7** | `char_len > 385.50` | `char_len > 570.50` | `has_guard_c0 <= 0.50` | `char_len > 570.50` and `has_guard_c0 <= 0.50` | **Cluster 7** |
| **Rule 8** | `char_len > 385.50` | `char_len > 570.50` | `has_guard_c0 > 0.50` | `char_len > 570.50` and `has_guard_c0 > 0.50` | **Cluster 7** |

---

### Decision Logic Flowchart

```
                            [ All Submissions ]
                                     │
                        Is char_len <= 385.50?
                                /          \
                              YES           NO
                              /               \
                    ┌────────────────┐     Is char_len <= 570.50?
                    │   CLUSTER 3    │            /          \
                    │ (Compact Core) │          YES           NO
                    └────────────────┘          /               \
                                        Is num_if <= 1.5?    ┌────────────────┐
                                           /         \       │   CLUSTER 7    │
                                         YES          NO     │ (Verbose Core) │
                                         /             \     └────────────────┘
                                ┌────────────────┐  ┌────────────────┐
                                │   CLUSTER 7    │  │   CLUSTER 3    │
                                │  (Streamlined) │  │  (Over-nested) │
                                └────────────────┘  └────────────────┘

```

* **Region 1 (Compact Code: $\le 385.5$ chars):** Exclusively assigned to **Cluster 3** due to short variable identifiers (`d`) and direct returns.
* **Region 2 (Medium Code: $385.5$ to $570.5$ chars):** Partitioned by branching depth:
* Minimal branching ($\text{num\_if} \le 1$) maps to **Cluster 7** (standard quadratic formula logic).
* High branching ($\text{num\_if} > 1$) maps to **Cluster 3** (defensive guard clauses like `if a == 0:`).


* **Region 3 (Verbose Code: $> 570.5$ chars):** Exclusively assigned to **Cluster 7**, characterized by descriptive variable declarations and expanded tuple returns.

---

## 3. Summary Deduction

1. **Semantic Validity:** Embedding models do not cluster student submissions randomly or on superficial whitespace alone; they form distinct semantic manifolds capturing **code length, nesting depth, and naming conventions**.
2. **Defensive vs. Idiomatic Taxonomies:**
* **Cluster 3** captures defensive programming styles containing multiple input validations, higher AST edit distances, and compact naming.
* **Cluster 7** captures idiomatic implementations adhering closely to the canonical textbook quadratic formula.


3. **Automated Interpretability:** Combining GumTree AST diffing with surrogate tree-based classifiers provides an end-to-end framework to interpret and explain deep neural code clustering.

```

```
