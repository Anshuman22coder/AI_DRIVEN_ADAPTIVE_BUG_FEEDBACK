Here is the complete, professionally formatted `README.md` file ready for you to copy and paste directly:

```markdown
# AI-Driven Adaptive Bug Feedback: Cluster Differentiation & AST Diff Analysis

This repository investigates the structural, syntactic, and semantic divergence within student programming submissions. Using **Abstract Syntax Tree (AST) GumTree diffing** alongside **surrogate interpretability models (Random Forest & Decision Tree)**, this pipeline reveals the exact factors governing code clustering in dense embedding spaces (e.g., GraphCodeBERT).

---

## Table of Contents
- [1. Analytical Report: AST Code-Diff (CQ08 Case Study)](#1-analytical-report-ast-code-diff-cq08-case-study)
  - [Executive Summary & Objective](#executive-summary--objective)
  - [Quantitative AST Diff Metrics](#quantitative-ast-diff-metrics)
  - [Visualization: Edit Distance Distribution](#visualization-edit-distance-distribution)
  - [Key Analytical Findings](#key-analytical-findings)
  - [Pedagogical Implications](#pedagogical-implications)
- [2. Finding Defining Factors Behind Code Clustering](#2-finding-defining-factors-behind-code-clustering)
  - [Algorithm Architecture & Methodology](#algorithm-architecture--methodology)
  - [Global Feature Importance (Random Forest Attribution)](#global-feature-importance-random-forest-attribution)
  - [Decision Logic Rules (Decision Tree Surrogate)](#decision-logic-rules-decision-tree-surrogate)
  - [Decision Logic Flowchart](#decision-logic-flowchart)
- [3. Summary Deduction](#3-summary-deduction)

---

## 1. Analytical Report: AST Code-Diff (CQ08 Case Study)

### Executive Summary & Objective
This case study evaluates the structural and syntactic boundaries between two dominant solution clusters—**Cluster 3 ($n = 814$)** and **Cluster 7 ($n = 726$)**—originating from **Canonical Question 08 (CQ08)**:
> *"Given a quadratic equation with coefficients $a, b,$ and $c$, return the two solutions, which may be real or complex..."*

Using **GumTree AST Diffing** on representative submissions from each cluster, we compute:
1. **Intra-Cluster Homogeneity:** 45 pairwise comparisons within Cluster 3, and 45 pairwise comparisons within Cluster 7.
2. **Inter-Cluster Separation:** 100 cross-cluster pairwise comparisons between Cluster 3 and Cluster 7.

---

### Quantitative AST Diff Metrics

| Comparison Regime | Sample Size ($N$) | Success Rate | Mean Operations | Median Operations | Mean Inserts | Mean Deletes | Mean Updates | Mean Moves |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Within Cluster 7 (C7 Intra)** | 45 pairs | 84.4% (38/45) | **23.02** | **9.0** | 7.78 | 12.38 | 0.00 | 2.87 |
| **Within Cluster 3 (C3 Intra)** | 45 pairs | 97.8% (44/45) | **140.89** | **150.0** | 54.31 | 68.27 | 5.11 | 13.20 |
| **Between C3 vs. C7 (Inter)** | 100 pairs | 100.0% (100/100) | **125.32** | **132.0** | 14.31 | 94.41 | 3.32 | 13.28 |

---

### Visualization: Edit Distance Distribution

```text
       AST Edit Distance Distribution                 Mean Composition of Edit Operations
  300 ┌──────────────────────────────────────┐    140 ┌──────────────────────────────────────┐
      │                                      │        │                             [Move]   │
  200 │                    ┌───┐      ┌───┐  │    100 │                 [Move]      [Delete] │
      │                    │   │      │   │  │        │                 [Delete]             │
  100 │         ┌───┐      │   │      │   │  │     50 │                             [Insert] │
      │         │   │      └───┘      └───┘  │        │    [Delete]                              │
    0 └─────────┴───┴────────────────────────┘      0 └────[Insert]──────────────────────────┘
              Within C7    C3 vs C7  Within C3           Within C7    C3 vs C7   Within C3

```

---

### Key Analytical Findings

#### A. High Structural Cohesion in Cluster 7 (Canonical Idiomatic Style)

* **Tight Semantic Core:** Cluster 7 exhibits minimal internal variance (median of only **9.0 AST operations** and 0 updates).
* **Identical AST Frequency:** 7 comparisons in Cluster 7 failed to generate an edit script due to `Source and Target AST are identical`, proving that top representatives share 100% identical tree syntax despite minor whitespace/comment variations.
* **Code Pattern:** Submissions directly calculate `discriminant = b**2 - 4*a*c` without nested guards, handle the branch as `if discriminant >= 0: ... else: ...`, and return clean raw tuples `[(root1, 0), (root2, 0)]` and `[(real_part, imaginary_part), (real_part, -imaginary_part)]`.

#### B. High Structural Variance in Cluster 3 (Defensive Guardrails & Over-Engineering)

* **Heterogeneous Implementations:** Cluster 3 shows high internal variance (mean **140.89 operations**, std dev **86.43**).
* **Defensive Edge-Case Handling:** Submissions extensively check input corner cases before computing roots, introducing deeply nested conditional trees (`if a == 0: if b == 0: ...`).
* **Explicit Precision Management:** Cluster 3 frequently uses explicit rounding (`round(..., 10)`, `round(..., 15)`, `round(..., 16)`) and attribute extractions (`root1.real`, `root1.imag`).

#### C. Nature of Inter-Cluster Separation (C3 vs. C7)

* **Asymmetric Deletion Dominance:** The cross-cluster comparison (C3 vs. C7) averages **94.41 deletes** versus only **14.31 inserts** when transforming C3 into C7. This reveals that **Cluster 3 is a structurally bloated superset** containing redundant validation logic, whereas **Cluster 7 is a streamlined direct implementation**.
* **Variable Naming Separation:** Across the full dataset, **59.4%** of Cluster 7 submissions use the explicit identifier `discriminant = ...` compared to **28.1%** in Cluster 3; conversely, Cluster 3 predominantly uses single-letter short identifiers `d = ...` or `D = ...` (**56.9%**).

---

### Pedagogical Implications

* **For Cluster 7 Students:** Provide automated feedback addressing mathematical edge cases (e.g., zero-division exceptions when $a=0$, float precision handling).
* **For Cluster 3 Students:** Encourage algorithmic simplification, avoiding redundant nested branching, and preventing over-engineered input guards beyond assignment specifications.

---

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
