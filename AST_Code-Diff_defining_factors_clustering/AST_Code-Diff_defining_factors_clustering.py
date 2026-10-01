# ============================================================
# GENERALIZED FAST CODEDIFF & SURROGATE INTERPRETABILITY PIPELINE
# ============================================================
#
# PURPOSE:
#   Analyze ANY Canonical Question from the dataset.
#
# WORKFLOW:
#   1. Install required libraries
#   2. Setup compatible Tree-Sitter grammar
#   3. Test CodeDiff
#   4. Upload Excel dataset
#   5. Enter Question Number / Canonical ID
#   6. Automatically detect top-2 dominant clusters
#   7. Optionally select custom clusters
#   8. Select up to 60 unique code variants per cluster
#   9. Run AST CodeDiff:
#        - Within Cluster A
#        - Within Cluster B
#        - Cluster A vs Cluster B
#  10. Generate statistical summaries
#  11. Extract cluster-defining features
#  12. Random Forest feature importance
#  13. Decision Tree rules
#  14. Export everything to Excel
#
# IMPORTANT:
#   This is the FAST VARIANT-OPTIMIZED version.
#   It does NOT compare every original submission when a cluster
#   contains more than MAX_VARIANTS_PER_CLUSTER unique code variants.
#
# ============================================================


# ============================================================
# 1. INSTALL REQUIRED LIBRARIES
# ============================================================

#!pip install -q code-diff openpyxl scikit-learn joblib
#!pip install -q tree-sitter==0.20.4 code-tokenize apted


# ============================================================
# 2. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import re
import ast
import itertools
import traceback
import os
import subprocess

import code_diff as cd
import code_tokenize
import code_tokenize.parsers as parsers

from tree_sitter import Language

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text

from google.colab import files


# ============================================================
# 3. CONFIGURATION
# ============================================================

# Maximum number of unique code variants analyzed per cluster.
#
# Example:
#   Cluster A = 814 submissions
#   Cluster B = 726 submissions
#
# If there are more than 60 unique code variants:
#   only the 60 most frequent unique variants are used
#   for AST CodeDiff comparisons.
#
# Surrogate ML still uses 100% of the selected-question data.
#
MAX_VARIANTS_PER_CLUSTER = 60


# ============================================================
# 4. CODEDIFF TREE-SITTER COMPATIBILITY SETUP
# ============================================================

print("\n" + "=" * 70)
print("SETTING UP CODEDIFF TREE-SITTER")
print("=" * 70)

BUILD_DIR = code_tokenize.parsers._path_to_local()

SOURCE_PATH = os.path.join(
    BUILD_DIR,
    "tree-sitter-python"
)

COMPILED_PATH = os.path.join(
    BUILD_DIR,
    "python-lang.so"
)


# ------------------------------------------------------------
# Download compatible Python grammar if required
# ------------------------------------------------------------

if not os.path.exists(SOURCE_PATH):

    print("Downloading tree-sitter-python v0.20.4...")

    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            "v0.20.4",
            "https://github.com/tree-sitter/tree-sitter-python.git",
            SOURCE_PATH
        ],
        check=True
    )


# ------------------------------------------------------------
# Compile compatible grammar
# ------------------------------------------------------------

if not os.path.exists(COMPILED_PATH):

    print("Compiling compatible Python grammar...")

    Language.build_library(
        COMPILED_PATH,
        [SOURCE_PATH]
    )


# ------------------------------------------------------------
# Verify grammar
# ------------------------------------------------------------

test_language = Language(
    COMPILED_PATH,
    "python"
)

print(
    "Python grammar ABI:",
    test_language.version
)

if test_language.version not in [13, 14]:

    raise RuntimeError(
        f"Incompatible Python grammar ABI: "
        f"{test_language.version}"
    )


# ------------------------------------------------------------
# Patch code-tokenize language loader
# ------------------------------------------------------------

_original_load_language = parsers.load_language


def _compatible_load_language(lang):

    if lang == "python":

        return Language(
            COMPILED_PATH,
            "python"
        )

    return _original_load_language(lang)


parsers.load_language = _compatible_load_language


print(
    "CodeDiff Tree-sitter compatibility patch applied."
)


# ============================================================
# 5. TEST CODEDIFF
# ============================================================

print("\n" + "=" * 70)
print("TESTING CODEDIFF")
print("=" * 70)


test_code_a = """
x = int(input())
print(x + 1)
""".strip()


test_code_b = """
x = int(input())
y = x + 1
print(y)
""".strip()


try:

    test_result = cd.difference(
        test_code_a,
        test_code_b,
        lang="python"
    )

    print("CodeDiff test: SUCCESS")

except Exception as e:

    print(
        "CodeDiff test: FAILED ->",
        type(e).__name__,
        ":",
        e
    )

    raise


# ============================================================
# 6. UPLOAD & READ EXCEL DATASET
# ============================================================

print("\n" + "=" * 70)
print("UPLOAD YOUR DATASET")
print("=" * 70)

uploaded = files.upload()

if not uploaded:
    raise ValueError("No file was uploaded.")

input_file = list(uploaded.keys())[0]

print(
    f"\nLoaded file: {input_file}"
)

df = pd.read_excel(input_file)

print(
    f"Dataset shape: {df.shape}"
)


# ============================================================
# 7. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "canonical_question_id",
    "canonical_question",
    "concept_type",
    "concept_name",
    "question",
    "code",
    "exec_feedback",
    "ai_explanation",
    "cluster"
]


missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]


if missing_columns:

    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(missing_columns)
    )


print(
    "\nAll required columns are present."
)


# ============================================================
# 8. QUESTION SELECTION
# ============================================================

print("\n" + "=" * 70)
print("SELECT TARGET QUESTION")
print("=" * 70)

target_q_input = input(
    "\nEnter Question Number or Canonical ID "
    "(e.g., 8, CQ08, 15, CQ15): "
).strip()


if not target_q_input:

    raise ValueError(
        "Question ID cannot be empty."
    )


# ------------------------------------------------------------
# Normalize question input
# ------------------------------------------------------------

digits_found = re.findall(
    r"\d+",
    target_q_input
)

target_num = (
    int(digits_found[0])
    if digits_found
    else None
)

target_clean_str = (
    target_q_input
    .upper()
    .strip()
)


# ============================================================
# 9. FLEXIBLE QUESTION MATCHING
# ============================================================

def is_target_question(value):

    if pd.isna(value):
        return False

    text = (
        str(value)
        .strip()
        .upper()
    )

    # Exact match
    if text == target_clean_str:
        return True

    # If user entered CQ08 / Q08 etc.
    if target_clean_str in text:
        return True

    # Numeric match
    text_digits = re.findall(
        r"\d+",
        text
    )

    if (
        target_num is not None
        and text_digits
    ):

        try:

            if int(text_digits[0]) == target_num:
                return True

        except:
            pass

    return False


mask_target = (

    df["canonical_question_id"]
    .apply(is_target_question)

    |

    df["canonical_question"]
    .apply(is_target_question)
)


target_df = df[
    mask_target
].copy()


if len(target_df) == 0:

    raise ValueError(
        f"Could not find any submissions "
        f"matching question '{target_q_input}'."
    )


# ============================================================
# 10. IDENTIFY CANONICAL QUESTION LABEL
# ============================================================

available_ids = (
    target_df["canonical_question_id"]
    .dropna()
)


if not available_ids.empty:

    canonical_label = str(
        available_ids.iloc[0]
    ).strip()

else:

    if target_num is not None:

        canonical_label = (
            f"CQ{target_num:02d}"
        )

    else:

        canonical_label = (
            target_clean_str
        )


print(
    f"\nTarget Question Identified: "
    f"{canonical_label}"
)

print(
    f"Total submissions found: "
    f"{len(target_df)}"
)


# ============================================================
# 11. CLEAN AND VALIDATE CODE
# ============================================================

target_df["code"] = (
    target_df["code"]
    .fillna("")
    .astype(str)
)


# Remove empty code
target_df = target_df[
    target_df["code"]
    .str.strip()
    != ""
].copy()


print(
    f"Submissions with non-empty code: "
    f"{len(target_df)}"
)


def clean_code(code):

    if pd.isna(code):
        return ""

    code = str(code)

    code = (
        code
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )

    return code.strip()


target_df["code_clean"] = (
    target_df["code"]
    .apply(clean_code)
)


def check_python_code(code):

    try:

        ast.parse(code)

        return True

    except:

        return False


target_df["python_valid"] = (
    target_df["code_clean"]
    .apply(check_python_code)
)


# Convert cluster to numeric
target_df["cluster"] = pd.to_numeric(
    target_df["cluster"],
    errors="coerce"
)


# Remove rows without cluster
target_df = target_df.dropna(
    subset=["cluster"]
).copy()


target_df["cluster"] = (
    target_df["cluster"]
    .astype(int)
)


print(
    f"Valid rows after cleaning: "
    f"{len(target_df)}"
)


# ============================================================
# 12. CLUSTER DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print(
    f"CLUSTER DISTRIBUTION FOR {canonical_label}"
)
print("=" * 70)


cluster_counts = (
    target_df["cluster"]
    .value_counts()
    .sort_values(
        ascending=False
    )
)


print(
    cluster_counts.to_string()
)


if len(cluster_counts) < 2:

    raise ValueError(
        f"Target question has fewer than "
        f"2 clusters. Found: "
        f"{len(cluster_counts)}"
    )


# ============================================================
# 13. AUTOMATIC TOP-2 CLUSTER SELECTION
# ============================================================

default_c_a = int(
    cluster_counts.index[0]
)

default_c_b = int(
    cluster_counts.index[1]
)


print(
    "\nDefault Top-2 Dominant Clusters:"
)

print(
    f"Cluster {default_c_a} "
    f"(n={cluster_counts.iloc[0]})"
)

print(
    f"Cluster {default_c_b} "
    f"(n={cluster_counts.iloc[1]})"
)


# ============================================================
# 14. OPTIONAL CUSTOM CLUSTER SELECTION
# ============================================================

override_choice = input(
    "\nPress [Enter] to use the default "
    f"Cluster {default_c_a} vs "
    f"Cluster {default_c_b}\n"
    "OR enter a custom pair "
    "(example: 3,7): "
).strip()


if override_choice:

    try:

        parts = [
            int(p.strip())
            for p in override_choice.split(",")
            if p.strip()
        ]

        if len(parts) != 2:

            raise ValueError

        c_a = parts[0]
        c_b = parts[1]

    except:

        print(
            "\nInvalid cluster input."
        )

        print(
            "Using default top-2 clusters."
        )

        c_a = default_c_a
        c_b = default_c_b

else:

    c_a = default_c_a
    c_b = default_c_b


# ------------------------------------------------------------
# Verify clusters exist
# ------------------------------------------------------------

if c_a not in set(target_df["cluster"]):

    raise ValueError(
        f"Cluster {c_a} does not exist "
        f"for {canonical_label}."
    )


if c_b not in set(target_df["cluster"]):

    raise ValueError(
        f"Cluster {c_b} does not exist "
        f"for {canonical_label}."
    )


if c_a == c_b:

    raise ValueError(
        "Cluster A and Cluster B "
        "must be different."
    )


# ============================================================
# 15. EXTRACT SELECTED CLUSTERS
# ============================================================

cluster_a_df = (
    target_df[
        target_df["cluster"] == c_a
    ]
    .copy()
    .reset_index(drop=True)
)


cluster_b_df = (
    target_df[
        target_df["cluster"] == c_b
    ]
    .copy()
    .reset_index(drop=True)
)


print("\n" + "=" * 70)
print("SELECTED CLUSTERS")
print("=" * 70)


print(
    f"Cluster {c_a}: "
    f"{len(cluster_a_df)} submissions"
)


print(
    f"Cluster {c_b}: "
    f"{len(cluster_b_df)} submissions"
)


# ============================================================
# 16. VARIANT OPTIMIZATION
# ============================================================
#
# Important:
#
# Suppose:
#
# Cluster A = 814 submissions
# Cluster B = 726 submissions
#
# Full comparison would be:
#
# C_A internal = 814 × 813 / 2
# C_B internal = 726 × 725 / 2
# Cross       = 814 × 726
#
# This becomes extremely expensive.
#
# Therefore:
#
#   1. Group identical code
#   2. Keep one representative for each unique code
#   3. Count how frequently that code occurs
#   4. Sort by frequency
#   5. Keep maximum 60 variants
#
# This preserves the most frequent code variants while
# dramatically reducing AST comparisons.
#
# ============================================================


print("\n" + "=" * 70)
print("SELECTING REPRESENTATIVE CODE VARIANTS")
print("=" * 70)


def get_representative_variants(
    cluster_df,
    max_n=MAX_VARIANTS_PER_CLUSTER
):

    data = cluster_df.copy()


    # --------------------------------------------------------
    # Count frequency of each exact code variant
    # --------------------------------------------------------

    counts = (
        data["code_clean"]
        .value_counts()
        .reset_index()
    )


    counts.columns = [
        "code_clean",
        "variant_frequency"
    ]


    # --------------------------------------------------------
    # Keep one row for each unique code
    # --------------------------------------------------------

    unique_df = (
        data
        .drop_duplicates(
            subset=["code_clean"]
        )
        .copy()
    )


    # --------------------------------------------------------
    # Add frequency
    # --------------------------------------------------------

    unique_df = unique_df.merge(
        counts,
        on="code_clean",
        how="left"
    )


    # --------------------------------------------------------
    # Most frequent variants first
    # --------------------------------------------------------

    unique_df = (
        unique_df
        .sort_values(
            by="variant_frequency",
            ascending=False
        )
        .reset_index(drop=True)
    )


    # --------------------------------------------------------
    # Keep maximum number of variants
    # --------------------------------------------------------

    if len(unique_df) > max_n:

        return unique_df.head(
            max_n
        ).copy()


    return unique_df.copy()


c_a_variants = get_representative_variants(
    cluster_a_df,
    MAX_VARIANTS_PER_CLUSTER
)


c_b_variants = get_representative_variants(
    cluster_b_df,
    MAX_VARIANTS_PER_CLUSTER
)


print(
    f"\nCluster {c_a}: "
    f"{len(cluster_a_df)} total submissions "
    f"-> {len(c_a_variants)} unique variants analyzed"
)


print(
    f"Cluster {c_b}: "
    f"{len(cluster_b_df)} total submissions "
    f"-> {len(c_b_variants)} unique variants analyzed"
)


# ============================================================
# 17. EXPECTED NUMBER OF COMPARISONS
# ============================================================

n_a = len(c_a_variants)
n_b = len(c_b_variants)


expected_a_internal = (
    n_a * (n_a - 1) // 2
)


expected_b_internal = (
    n_b * (n_b - 1) // 2
)


expected_cross = (
    n_a * n_b
)


expected_total = (
    expected_a_internal
    + expected_b_internal
    + expected_cross
)


print("\n" + "=" * 70)
print("EXPECTED AST COMPARISONS")
print("=" * 70)


print(
    f"Within Cluster {c_a}: "
    f"{expected_a_internal:,}"
)


print(
    f"Within Cluster {c_b}: "
    f"{expected_b_internal:,}"
)


print(
    f"Cluster {c_a} vs Cluster {c_b}: "
    f"{expected_cross:,}"
)


print(
    f"TOTAL: "
    f"{expected_total:,}"
)


# ============================================================
# 18. CODEDIFF ENGINE
# ============================================================

def run_codediff(
    code_a,
    code_b
):

    # --------------------------------------------------------
    # Identical code shortcut
    # --------------------------------------------------------

    if code_a == code_b:

        return {
            "success": True,
            "edit_script": "[]",
            "error": ""
        }


    # --------------------------------------------------------
    # AST size safeguard
    # --------------------------------------------------------

    if (
        len(code_a) > 2500
        or len(code_b) > 2500
    ):

        return {
            "success": False,
            "edit_script": "",
            "error":
                "Code snippet too long "
                "for APTED tree diff"
        }


    # --------------------------------------------------------
    # Run CodeDiff
    # --------------------------------------------------------

    try:

        result = cd.difference(
            code_a,
            code_b,
            lang="python"
        )


        return {
            "success": True,
            "edit_script":
                str(result.edit_script()),
            "error": ""
        }


    except Exception as e:

        return {
            "success": False,
            "edit_script": "",
            "error": str(e)
        }


# ============================================================
# 19. EXTRACT AST OPERATION COUNTS
# ============================================================

OP_PATTERN = re.compile(
    r"(Insert|Delete|Update|Move)\("
)


def extract_operation_counts(
    edit_script
):

    if (
        not edit_script
        or edit_script == "[]"
    ):

        return (
            0,
            0,
            0,
            0,
            0,
            ""
        )


    matches = OP_PATTERN.findall(
        edit_script
    )


    total = len(matches)

    inserts = matches.count(
        "Insert"
    )

    deletes = matches.count(
        "Delete"
    )

    updates = matches.count(
        "Update"
    )

    moves = matches.count(
        "Move"
    )


    unique_types = "|".join(
        sorted(
            set(matches)
        )
    )


    return (
        total,
        inserts,
        deletes,
        updates,
        moves,
        unique_types
    )


# ============================================================
# 20. COMPARE TWO SUBMISSIONS
# ============================================================

def compare_submissions(
    row_a,
    row_b,
    comparison_type
):

    code_a = row_a[
        "code_clean"
    ]

    code_b = row_b[
        "code_clean"
    ]


    result = run_codediff(
        code_a,
        code_b
    )


    (
        total_ops,
        ins,
        dels,
        upds,
        movs,
        op_types
    ) = extract_operation_counts(
        result["edit_script"]
    )


    return {

        "comparison_type":
            comparison_type,

        "cluster_a":
            row_a["cluster"],

        "cluster_b":
            row_b["cluster"],

        "canonical_question_id_a":
            row_a["canonical_question_id"],

        "canonical_question_id_b":
            row_b["canonical_question_id"],

        "row_index_a":
            row_a.name,

        "row_index_b":
            row_b.name,

        "code_chars_a":
            len(code_a),

        "code_chars_b":
            len(code_b),

        "code_lines_a":
            len(code_a.splitlines()),

        "code_lines_b":
            len(code_b.splitlines()),

        "diff_success":
            result["success"],

        "number_of_operations":
            total_ops,

        "insert_count":
            ins,

        "delete_count":
            dels,

        "update_count":
            upds,

        "move_count":
            movs,

        "operation_types":
            op_types,

        "edit_script":
            result["edit_script"],

        "error":
            result["error"],

        "code_a":
            code_a,

        "code_b":
            code_b
    }


# ============================================================
# 21. WITHIN-CLUSTER COMPARISON
# ============================================================

def compare_within_cluster_safe(
    data,
    cluster_id
):

    records = list(
        data.iterrows()
    )


    pairs = list(
        itertools.combinations(
            records,
            2
        )
    )


    total = len(pairs)


    print(
        f"\nProcessing Cluster "
        f"{cluster_id} "
        f"({total:,} pairs)..."
    )


    results = []


    for idx, (
        (_, row_a),
        (_, row_b)
    ) in enumerate(
        pairs,
        1
    ):

        results.append(
            compare_submissions(
                row_a,
                row_b,
                f"Within Cluster {cluster_id}"
            )
        )


        if (
            idx % 300 == 0
            or idx == total
        ):

            print(
                f" -> Cluster {cluster_id}: "
                f"{idx:,}/{total:,} "
                f"completed "
                f"({idx / total * 100:.1f}%)"
            )


    return pd.DataFrame(
        results
    )


# ============================================================
# 22. BETWEEN-CLUSTER COMPARISON
# ============================================================

def compare_between_clusters_safe(
    cluster_a,
    cluster_b,
    c_a_id,
    c_b_id
):

    records_a = list(
        cluster_a.iterrows()
    )

    records_b = list(
        cluster_b.iterrows()
    )


    pairs = [
        (row_a, row_b)
        for _, row_a in records_a
        for _, row_b in records_b
    ]


    total = len(pairs)


    print(
        f"\nProcessing Cross-Cluster "
        f"C{c_a_id} vs C{c_b_id} "
        f"({total:,} pairs)..."
    )


    results = []


    for idx, (
        row_a,
        row_b
    ) in enumerate(
        pairs,
        1
    ):

        results.append(
            compare_submissions(
                row_a,
                row_b,
                f"Cluster {c_a_id} vs Cluster {c_b_id}"
            )
        )


        if (
            idx % 500 == 0
            or idx == total
        ):

            print(
                f" -> C{c_a_id} vs C{c_b_id}: "
                f"{idx:,}/{total:,} "
                f"completed "
                f"({idx / total * 100:.1f}%)"
            )


    return pd.DataFrame(
        results
    )


# ============================================================
# 23. RUN AST CODEDIFF
# ============================================================

print("\n" + "=" * 70)
print(
    f"RUNNING AST CODEDIFF FOR "
    f"{canonical_label}"
)
print("=" * 70)


c_a_internal_df = (
    compare_within_cluster_safe(
        c_a_variants,
        c_a
    )
)


c_b_internal_df = (
    compare_within_cluster_safe(
        c_b_variants,
        c_b
    )
)


inter_cluster_df = (
    compare_between_clusters_safe(
        c_a_variants,
        c_b_variants,
        c_a,
        c_b
    )
)


all_diff_df = pd.concat(
    [
        c_a_internal_df,
        c_b_internal_df,
        inter_cluster_df
    ],
    ignore_index=True
)


print(
    "\nAll AST diffs completed successfully!"
)


# ============================================================
# 24. STATISTICAL SUMMARIES
# ============================================================

print("\n" + "=" * 70)
print("GENERATING STATISTICAL SUMMARIES")
print("=" * 70)


def operation_summary(
    diff_df
):

    if diff_df.empty:

        return pd.DataFrame()


    rows = []


    for (
        comp_type,
        group
    ) in diff_df.groupby(
        "comparison_type"
    ):

        rows.append({

            "comparison_type":
                comp_type,

            "comparisons":
                len(group),

            "successful_diffs":
                group[
                    "diff_success"
                ].sum(),

            "failed_diffs":
                (
                    ~group[
                        "diff_success"
                    ]
                ).sum(),

            "average_operations":
                group[
                    "number_of_operations"
                ].mean(),

            "median_operations":
                group[
                    "number_of_operations"
                ].median(),

            "average_inserts":
                group[
                    "insert_count"
                ].mean(),

            "average_deletes":
                group[
                    "delete_count"
                ].mean(),

            "average_updates":
                group[
                    "update_count"
                ].mean(),

            "average_moves":
                group[
                    "move_count"
                ].mean()
        })


    return pd.DataFrame(
        rows
    )


diff_summary_df = (
    operation_summary(
        all_diff_df
    )
)


# ============================================================
# 25. EDIT OPERATION SUMMARY
# ============================================================

operation_rows = []


for (
    comp_type,
    group
) in all_diff_df.groupby(
    "comparison_type"
):

    for (
        op_name,
        col_name
    ) in [

        ("Insert", "insert_count"),

        ("Delete", "delete_count"),

        ("Update", "update_count"),

        ("Move", "move_count")
    ]:

        operation_rows.append({

            "comparison_type":
                comp_type,

            "operation":
                op_name,

            "total_count":
                group[col_name].sum(),

            "average_per_comparison":
                group[col_name].mean()
        })


operation_summary_df = pd.DataFrame(
    operation_rows
)


# ============================================================
# 26. FAILED / NO-CHANGE / MOST-DIFFERENT
# ============================================================

failed_diffs_df = (
    all_diff_df[
        all_diff_df[
            "diff_success"
        ] == False
    ].copy()
)


no_change_df = (
    all_diff_df[
        (
            all_diff_df[
                "diff_success"
            ] == True
        )
        &
        (
            all_diff_df[
                "number_of_operations"
            ] == 0
        )
    ].copy()
)


most_different_df = (
    all_diff_df[
        all_diff_df[
            "diff_success"
        ] == True
    ]
    .sort_values(
        "number_of_operations",
        ascending=False
    )
    .copy()
)


# ============================================================
# 27. SUBMISSION TABLES
# ============================================================

def submission_table(
    data
):

    columns = [

        "canonical_question_id",

        "canonical_question",

        "concept_type",

        "concept_name",

        "question",

        "cluster",

        "python_valid",

        "code_clean",

        "variant_frequency"
    ]


    return data[
        [
            c
            for c in columns
            if c in data.columns
        ]
    ].copy()


c_a_output = submission_table(
    cluster_a_df
)


c_b_output = submission_table(
    cluster_b_df
)


c_a_variants_output = submission_table(
    c_a_variants
)


c_b_variants_output = submission_table(
    c_b_variants
)


# ============================================================
# 28. DYNAMIC CLUSTER-DEFINING FEATURES
# ============================================================

print("\n" + "=" * 70)
print(
    f"EXTRACTING DEFINING FACTORS "
    f"(Cluster {c_a} vs Cluster {c_b})"
)
print("=" * 70)


def extract_code_features(
    code_str
):

    code_str = str(
        code_str
    )


    feats = {

        "char_len":
            len(code_str),

        "num_lines":
            code_str.count("\n") + 1,

        "has_round":
            int(
                "round(" in code_str
            ),

        "round_count":
            code_str.count(
                "round("
            ),

        "has_guard_a0":
            int(
                "a == 0" in code_str
                or
                "a==0" in code_str
            ),

        "has_guard_b0":
            int(
                "b == 0" in code_str
                or
                "b==0" in code_str
            ),

        "has_guard_c0":
            int(
                "c == 0" in code_str
                or
                "c==0" in code_str
            ),

        "has_real_attr":
            int(
                ".real" in code_str
                or
                ".imag" in code_str
            ),

        "has_cmath":
            int(
                "cmath" in code_str
            ),

        "var_discriminant":
            int(
                "discriminant"
                in code_str.lower()
            ),

        "var_d_short":
            int(
                bool(
                    re.search(
                        r"\bd\b",
                        code_str
                    )
                )
            ),

        "var_D_upper":
            int(
                bool(
                    re.search(
                        r"\bD\b",
                        code_str
                    )
                )
            ),

        "returns_0_float":
            int(
                "0.0" in code_str
            ),

        "returns_0_int":
            int(
                "(0)" in code_str
                or
                ", 0)" in code_str
                or
                ",0)" in code_str
            ),

        "abs_d":
            int(
                "abs(" in code_str
            ),

        "minus_d":
            int(
                "(-d)" in code_str
                or
                "-d" in code_str
                or
                "(-discriminant)"
                in code_str
            )
    }


    # --------------------------------------------------------
    # AST features
    # --------------------------------------------------------

    try:

        tree = ast.parse(
            code_str
        )


        feats["num_if"] = sum(
            isinstance(
                node,
                ast.If
            )
            for node in ast.walk(tree)
        )


        feats["num_return"] = sum(
            isinstance(
                node,
                ast.Return
            )
            for node in ast.walk(tree)
        )


        feats["num_for_while"] = sum(
            isinstance(
                node,
                (ast.For, ast.While)
            )
            for node in ast.walk(tree)
        )


        feats["max_depth"] = max(
            [
                len(
                    list(
                        ast.iter_child_nodes(
                            node
                        )
                    )
                )
                for node in ast.walk(tree)
            ]
            or [0]
        )


    except Exception:

        feats["num_if"] = (
            code_str.count("if ")
        )

        feats["num_return"] = (
            code_str.count("return ")
        )

        feats["num_for_while"] = (
            code_str.count("for ")
            +
            code_str.count("while ")
        )

        feats["max_depth"] = 0


    return feats


# ------------------------------------------------------------
# IMPORTANT:
# Feature model uses ALL submissions belonging to the
# selected two clusters, not just the CodeDiff variants.
# ------------------------------------------------------------

selected_cluster_data = (
    target_df[
        target_df["cluster"].isin(
            [c_a, c_b]
        )
    ].copy()
)


selected_cluster_data[
    "code_clean"
] = (
    selected_cluster_data["code"]
    .apply(clean_code)
)


feature_records = [

    extract_code_features(
        code
    )

    for code
    in selected_cluster_data[
        "code_clean"
    ]
]


X_features = pd.DataFrame(
    feature_records
)


y_labels = (
    selected_cluster_data[
        "cluster"
    ].astype(int)
)


# ============================================================
# 29. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


rf_model.fit(
    X_features,
    y_labels
)


feature_importance_df = pd.DataFrame({

    "Feature":
        X_features.columns,

    "Importance_Score":
        rf_model.feature_importances_

}).sort_values(
    "Importance_Score",
    ascending=False
).reset_index(
    drop=True
)


# ============================================================
# 30. DECISION TREE RULES
# ============================================================

dt_model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)


dt_model.fit(
    X_features,
    y_labels
)


tree_rules_text = export_text(
    dt_model,
    feature_names=list(
        X_features.columns
    )
)


rules_df = pd.DataFrame({

    "Decision_Tree_Rules":
        tree_rules_text.splitlines()

})


# ============================================================
# 31. DISPLAY FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL ANALYSIS SUMMARY")
print("=" * 70)


print(
    f"\nQuestion: {canonical_label}"
)


print(
    f"Cluster {c_a}: "
    f"{len(cluster_a_df)} submissions"
)


print(
    f"Cluster {c_b}: "
    f"{len(cluster_b_df)} submissions"
)


print(
    f"Cluster {c_a} variants analyzed: "
    f"{len(c_a_variants)}"
)


print(
    f"Cluster {c_b} variants analyzed: "
    f"{len(c_b_variants)}"
)


print(
    f"\nTotal CodeDiff comparisons: "
    f"{len(all_diff_df):,}"
)


print(
    "\n--- Difference Summary ---"
)


display(
    diff_summary_df
)


print(
    "\n--- Top 10 Defining Features ---"
)


display(
    feature_importance_df.head(10)
)


print(
    "\n--- Decision Tree Rules ---"
)


print(
    tree_rules_text
)


# ============================================================
# 32. EXPORT TO EXCEL
# ============================================================

print("\n" + "=" * 70)
print("EXPORTING RESULTS TO EXCEL")
print("=" * 70)


# Remove problematic characters from sheet/file label
clean_q_tag = re.sub(
    r"[^A-Za-z0-9]",
    "",
    str(canonical_label)
)


output_file = (
    f"/content/"
    f"{clean_q_tag}_"
    f"CodeDiff_"
    f"Cluster{c_a}_"
    f"Cluster{c_b}_"
    f"Analysis.xlsx"
)


print(
    f"\nOutput file:\n"
    f"{output_file}"
)


with pd.ExcelWriter(
    output_file,
    engine="openpyxl"
) as writer:


    # --------------------------------------------------------
    # Complete selected question
    # --------------------------------------------------------

    target_df.to_excel(
        writer,
        sheet_name=f"{clean_q_tag}_Data",
        index=False
    )


    # --------------------------------------------------------
    # All submissions from selected clusters
    # --------------------------------------------------------

    c_a_output.to_excel(
        writer,
        sheet_name=f"C{c_a}_All_Submissions",
        index=False
    )


    c_b_output.to_excel(
        writer,
        sheet_name=f"C{c_b}_All_Submissions",
        index=False
    )


    # --------------------------------------------------------
    # Selected CodeDiff variants
    # --------------------------------------------------------

    c_a_variants_output.to_excel(
        writer,
        sheet_name=f"C{c_a}_Variants",
        index=False
    )


    c_b_variants_output.to_excel(
        writer,
        sheet_name=f"C{c_b}_Variants",
        index=False
    )


    # --------------------------------------------------------
    # AST Diff results
    # --------------------------------------------------------

    c_a_internal_df.to_excel(
        writer,
        sheet_name=f"C{c_a}_Internal_Diffs",
        index=False
    )


    c_b_internal_df.to_excel(
        writer,
        sheet_name=f"C{c_b}_Internal_Diffs",
        index=False
    )


    inter_cluster_df.to_excel(
        writer,
        sheet_name=f"C{c_a}_vs_C{c_b}_Diffs",
        index=False
    )


    # --------------------------------------------------------
    # Statistical summaries
    # --------------------------------------------------------

    diff_summary_df.to_excel(
        writer,
        sheet_name="Diff_Summary",
        index=False
    )


    operation_summary_df.to_excel(
        writer,
        sheet_name="Edit_Operation_Summary",
        index=False
    )


    # --------------------------------------------------------
    # Special pair reports
    # --------------------------------------------------------

    most_different_df.to_excel(
        writer,
        sheet_name="Most_Different_Pairs",
        index=False
    )


    no_change_df.to_excel(
        writer,
        sheet_name="No_Change_Pairs",
        index=False
    )


    failed_diffs_df.to_excel(
        writer,
        sheet_name="Failed_Diffs",
        index=False
    )


    # --------------------------------------------------------
    # Interpretability
    # --------------------------------------------------------

    feature_importance_df.to_excel(
        writer,
        sheet_name="Feature_Importance",
        index=False
    )


    rules_df.to_excel(
        writer,
        sheet_name="Decision_Rules",
        index=False
    )


print(
    "\nExcel export complete!"
)


# ============================================================
# 33. DOWNLOAD
# ============================================================

print("\n" + "=" * 70)
print("PROCESSING COMPLETE")
print("=" * 70)


print(
    f"\nQuestion analyzed: "
    f"{canonical_label}"
)


print(
    f"Compared Cluster {c_a} "
    f"vs Cluster {c_b}"
)


print(
    f"Total AST comparisons: "
    f"{len(all_diff_df):,}"
)


print(
    f"\nDownloading:\n"
    f"{output_file}"
)


files.download(
    output_file
)