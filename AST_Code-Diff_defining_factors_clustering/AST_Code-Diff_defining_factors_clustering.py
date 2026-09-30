# ============================================================
# GENERALIZED CODEDIFF & SURROGATE INTERPRETABILITY PIPELINE
# ============================================================
#
# Automatically processes ANY Canonical Question:
#   1. Prompts for Question Number / ID (e.g., "8", "CQ08", "15")
#   2. Detects the Top-2 dominant clusters (or allows custom selection)
#   3. Executes intra-cluster and inter-cluster AST CodeDiffs
#   4. Trains Random Forest & Decision Tree to explain separation
#   5. Exports a beautifully formatted multi-sheet Excel file
#
# ============================================================


# ============================================================
# 1. INSTALL REQUIRED LIBRARIES
# ============================================================

#!pip install -q code-diff openpyxl scikit-learn
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

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text

from google.colab import files


# ============================================================
# 3. CODEDIFF TREE-SITTER COMPATIBILITY SETUP
# ============================================================

BUILD_DIR = code_tokenize.parsers._path_to_local()
SOURCE_PATH = os.path.join(BUILD_DIR, "tree-sitter-python")
COMPILED_PATH = os.path.join(BUILD_DIR, "python-lang.so")

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

if not os.path.exists(COMPILED_PATH):
    print("Compiling compatible Python grammar...")
    Language.build_library(
        COMPILED_PATH,
        [SOURCE_PATH]
    )

test_language = Language(COMPILED_PATH, "python")
print(f"Python grammar ABI: {test_language.version}")

if test_language.version not in [13, 14]:
    raise RuntimeError(f"Incompatible Python grammar ABI: {test_language.version}")

_original_load_language = parsers.load_language

def _compatible_load_language(lang):
    if lang == "python":
        return Language(COMPILED_PATH, "python")
    return _original_load_language(lang)

parsers.load_language = _compatible_load_language
print("CodeDiff Tree-sitter compatibility patch applied.")


# ============================================================
# 4. UPLOAD EXCEL FILE
# ============================================================

print("\n" + "=" * 70)
print("UPLOAD YOUR DATASET (Excel File)")
print("=" * 70)

uploaded = files.upload()
input_file = list(uploaded.keys())[0]
print(f"\nLoaded file: {input_file}")

df = pd.read_excel(input_file)
print(f"Dataset shape: {df.shape}")

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

missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns in dataset: {missing_columns}")

print("All required columns are present.")


# ============================================================
# 5. USER QUESTION SELECTION & NORMALIZATION
# ============================================================

print("\n" + "=" * 70)
print("SELECT TARGET QUESTION")
print("=" * 70)

target_q_input = input("Enter Question Number or Canonical ID (e.g., 8, CQ08, 15, CQ15): ").strip()

# Build flexible question matcher
digits_found = re.findall(r'\d+', target_q_input)
target_num = int(digits_found[0]) if digits_found else None
target_clean_str = target_q_input.upper()

def is_target_question(val):
    if pd.isna(val):
        return False
    text = str(val).strip().upper()
    if target_clean_str in text:
        return True
    num_match = re.findall(r'\d+', text)
    if target_num is not None and num_match and int(num_match[0]) == target_num:
        return True
    return False

mask_target = (
    df["canonical_question_id"].apply(is_target_question)
    | df["canonical_question"].apply(is_target_question)
)

target_df = df[mask_target].copy()
if len(target_df) == 0:
    raise ValueError(f"Could not find any submissions matching question '{target_q_input}'.")

canonical_label = target_df["canonical_question_id"].dropna().iloc[0] if not target_df["canonical_question_id"].dropna().empty else f"CQ{target_num:02d}"
print(f"\nTarget Question Identified: {canonical_label}")
print(f"Total Submissions Found: {len(target_df)}")

# Remove empty code submissions
target_df["code"] = target_df["code"].fillna("").astype(str)
target_df = target_df[target_df["code"].str.strip() != ""].copy()
print(f"Submissions with non-empty code: {len(target_df)}")

# Clean code
def clean_code(code):
    if pd.isna(code): return ""
    code = str(code)
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    return code.strip()

target_df["code_clean"] = target_df["code"].apply(clean_code)

def check_python_code(code):
    try:
        ast.parse(code)
        return True
    except:
        return False

target_df["python_valid"] = target_df["code_clean"].apply(check_python_code)
target_df["cluster"] = pd.to_numeric(target_df["cluster"], errors="coerce")
target_df = target_df.dropna(subset=["cluster"]).copy()
target_df["cluster"] = target_df["cluster"].astype(int)


# ============================================================
# 6. DYNAMIC CLUSTER DETECTION & SELECTION
# ============================================================

print("\n" + "=" * 70)
print(f"CLUSTER DISTRIBUTION FOR {canonical_label}")
print("=" * 70)

cluster_counts = target_df["cluster"].value_counts()
print(cluster_counts.to_string())

if len(cluster_counts) < 2:
    raise ValueError(f"Target question has fewer than 2 clusters ({len(cluster_counts)} found). Cannot perform cluster comparison.")

default_c_a = int(cluster_counts.index[0])
default_c_b = int(cluster_counts.index[1])

print(f"\nDefault Top-2 Dominant Clusters: Cluster {default_c_a} (n={cluster_counts.iloc[0]}) and Cluster {default_c_b} (n={cluster_counts.iloc[1]})")
override_choice = input(f"Press [Enter] to use Cluster {default_c_a} vs Cluster {default_c_b}, or enter custom pair (e.g. '3, 7'): ").strip()

if override_choice:
    parts = [int(p.strip()) for p in override_choice.split(',') if p.strip().isdigit()]
    if len(parts) == 2:
        c_a, c_b = parts[0], parts[1]
    else:
        print("Invalid input format. Falling back to default top-2 clusters.")
        c_a, c_b = default_c_a, default_c_b
else:
    c_a, c_b = default_c_a, default_c_b

cluster_a_df = target_df[target_df["cluster"] == c_a].copy().reset_index(drop=True)
cluster_b_df = target_df[target_df["cluster"] == c_b].copy().reset_index(drop=True)

print(f"\nSelected Comparison: Cluster {c_a} (n={len(cluster_a_df)}) vs Cluster {c_b} (n={len(cluster_b_df)})")


# ============================================================
# 7. TF-IDF REPRESENTATIVE SELECTION
# ============================================================

def select_representatives(data, n_representatives=10):
    data = data.copy()
    if len(data) <= n_representatives:
        return data.copy()
    codes = data["code_clean"].tolist()
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b[A-Za-z_][A-Za-z0-9_]*\b")
    try:
        matrix = vectorizer.fit_transform(codes)
    except Exception:
        return data.head(n_representatives).copy()
    similarity = cosine_similarity(matrix)
    average_similarity = (similarity.sum(axis=1) - 1) / (len(data) - 1)
    data["representativeness_score"] = average_similarity
    return data.sort_values("representativeness_score", ascending=False).head(n_representatives).copy()

N_REPS = 10
c_a_reps = select_representatives(cluster_a_df, N_REPS)
c_b_reps = select_representatives(cluster_b_df, N_REPS)

print(f"Selected {len(c_a_reps)} reps for Cluster {c_a} and {len(c_b_reps)} reps for Cluster {c_b}.")


# ============================================================
# 8. AST CODEDIFF EXECUTION
# ============================================================

def run_codediff(code_a, code_b):
    try:
        result = cd.difference(code_a, code_b, lang="python")
        return {"success": True, "edit_script": str(result.edit_script()), "error": ""}
    except Exception as e:
        return {"success": False, "edit_script": "", "error": str(e)}

def extract_operation_types(edit_script):
    if not edit_script: return []
    operations = []
    text = str(edit_script)
    for op in ["Insert", "Delete", "Update", "Move"]:
        operations.extend([op] * text.count(op + "("))
    return operations

def compare_submissions(row_a, row_b, comparison_type):
    code_a = row_a["code_clean"]
    code_b = row_b["code_clean"]
    result = run_codediff(code_a, code_b)
    ops = extract_operation_types(result["edit_script"])
    return {
        "comparison_type": comparison_type,
        "cluster_a": row_a["cluster"],
        "cluster_b": row_b["cluster"],
        "canonical_question_id_a": row_a["canonical_question_id"],
        "canonical_question_id_b": row_b["canonical_question_id"],
        "row_index_a": row_a.name,
        "row_index_b": row_b.name,
        "code_chars_a": len(code_a),
        "code_chars_b": len(code_b),
        "code_lines_a": len(code_a.splitlines()),
        "code_lines_b": len(code_b.splitlines()),
        "diff_success": result["success"],
        "number_of_operations": len(ops),
        "insert_count": ops.count("Insert"),
        "delete_count": ops.count("Delete"),
        "update_count": ops.count("Update"),
        "move_count": ops.count("Move"),
        "operation_types": "|".join(sorted(set(ops))),
        "edit_script": result["edit_script"],
        "error": result["error"],
        "code_a": code_a,
        "code_b": code_b
    }

def compare_within_cluster(data, cluster_id):
    rows = []
    records = list(data.iterrows())
    for (_, row_a), (_, row_b) in itertools.combinations(records, 2):
        rows.append(compare_submissions(row_a, row_b, f"Within Cluster {cluster_id}"))
    return pd.DataFrame(rows)

def compare_between_clusters(data_a, data_b, c_a_id, c_b_id):
    rows = []
    for _, row_a in data_a.iterrows():
        for _, row_b in data_b.iterrows():
            rows.append(compare_submissions(row_a, row_b, f"Cluster {c_a_id} vs Cluster {c_b_id}"))
    return pd.DataFrame(rows)

print("\n" + "=" * 70)
print(f"RUNNING CODEDIFF COMPARISONS FOR {canonical_label}")
print("=" * 70)

c_a_internal_df = compare_within_cluster(c_a_reps, c_a)
c_b_internal_df = compare_within_cluster(c_b_reps, c_b)
inter_cluster_df = compare_between_clusters(c_a_reps, c_b_reps, c_a, c_b)
all_diff_df = pd.concat([c_a_internal_df, c_b_internal_df, inter_cluster_df], ignore_index=True)


# ============================================================
# 9. OPERATION & SUMMARY TABLES
# ============================================================

def operation_summary(diff_df):
    if diff_df.empty: return pd.DataFrame()
    rows = []
    for comp_type, group in diff_df.groupby("comparison_type"):
        rows.append({
            "comparison_type": comp_type,
            "comparisons": len(group),
            "successful_diffs": group["diff_success"].sum(),
            "failed_diffs": (~group["diff_success"]).sum(),
            "average_operations": group["number_of_operations"].mean(),
            "median_operations": group["number_of_operations"].median(),
            "average_inserts": group["insert_count"].mean(),
            "average_deletes": group["delete_count"].mean(),
            "average_updates": group["update_count"].mean(),
            "average_moves": group["move_count"].mean()
        })
    return pd.DataFrame(rows)

diff_summary_df = operation_summary(all_diff_df)

op_rows = []
for comp_type, group in all_diff_df.groupby("comparison_type"):
    for op in ["Insert", "Delete", "Update", "Move"]:
        col = f"{op.lower()}_count"
        op_rows.append({
            "comparison_type": comp_type,
            "operation": op,
            "total_count": group[col].sum(),
            "average_per_comparison": group[col].mean()
        })
operation_summary_df = pd.DataFrame(op_rows)

failed_diffs_df = all_diff_df[all_diff_df["diff_success"] == False].copy()
no_change_df = all_diff_df[(all_diff_df["diff_success"] == True) & (all_diff_df["number_of_operations"] == 0)].copy()
most_different_df = all_diff_df[all_diff_df["diff_success"] == True].sort_values("number_of_operations", ascending=False).copy()

rep_cols = ["canonical_question_id", "canonical_question", "question", "cluster", "representativeness_score", "python_valid", "code_clean"]
c_a_rep_out = c_a_reps[[c for c in rep_cols if c in c_a_reps.columns]].copy()
c_b_rep_out = c_b_reps[[c for c in rep_cols if c in c_b_reps.columns]].copy()


# ============================================================
# 10. DYNAMIC FEATURE EXTRACTION & SURROGATE ATTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print(f"EXTRACTING DEFINING FACTORS (Cluster {c_a} vs Cluster {c_b})")
print("=" * 70)

# Extract domain keywords from question description
q_text = str(target_df["question"].dropna().iloc[0]).lower()
keywords = set(re.findall(r'\b[a-zA-Z_]{3,}\b', q_text)) - {'given', 'return', 'input', 'output', 'format', 'code', 'function', 'test', 'case', 'value', 'type'}
top_keywords = list(keywords)[:4]

def extract_generalized_features(code_str):
    code_str = str(code_str)
    feats = {
        'char_len': len(code_str),
        'num_lines': code_str.count('\n') + 1,
        'has_round': int('round(' in code_str),
        'round_count': code_str.count('round('),
        'has_zero_check': int('== 0' in code_str or '==0' in code_str),
        'has_real_imag_attr': int('.real' in code_str or '.imag' in code_str),
        'has_type_cast_float': int('float(' in code_str),
        'has_type_cast_int': int('int(' in code_str),
        'has_type_cast_str': int('str(' in code_str),
        'has_single_char_var': int(bool(re.search(r'\b[a-zA-Z]\s*=', code_str))),
        'has_list_comp': int(bool(re.search(r'\[.*for\s+.*in\s+.*\]', code_str))),
        'has_abs': int('abs(' in code_str),
    }
    
    # Question specific keyword identifiers
    for kw in top_keywords:
        feats[f'var_contains_{kw}'] = int(kw in code_str.lower())
        
    try:
        tree = ast.parse(code_str)
        feats['num_if'] = sum(isinstance(node, ast.If) for node in ast.walk(tree))
        feats['num_return'] = sum(isinstance(node, ast.Return) for node in ast.walk(tree))
        feats['num_for_while'] = sum(isinstance(node, (ast.For, ast.While)) for node in ast.walk(tree))
        feats['max_depth'] = max([len(list(ast.iter_child_nodes(node))) for node in ast.walk(tree)] or [0])
    except Exception:
        feats['num_if'] = code_str.count('if ')
        feats['num_return'] = code_str.count('return ')
        feats['num_for_while'] = code_str.count('for ') + code_str.count('while ')
        feats['max_depth'] = 0
    return feats

surrogate_data = target_df[target_df['cluster'].isin([c_a, c_b])].copy()
X_features = pd.DataFrame([extract_generalized_features(c) for c in surrogate_data["code_clean"]])
y_labels = surrogate_data["cluster"].astype(int)

# 1. Random Forest Feature Importance
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_features, y_labels)

feature_importance_df = pd.DataFrame({
    'Feature': X_features.columns,
    'Importance_Score': [f"{score * 100:.2f}%" for score in rf_model.feature_importances_],
    'Raw_Score': rf_model.feature_importances_
}).sort_values('Raw_Score', ascending=False).drop(columns=['Raw_Score']).reset_index(drop=True)

# 2. Decision Tree Rules
dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_model.fit(X_features, y_labels)
tree_rules_text = export_text(dt_model, feature_names=list(X_features.columns))
rules_df = pd.DataFrame({'Decision_Tree_Rules': tree_rules_text.splitlines()})

print("\n--- Top 10 Defining Features by Random Forest Importance ---")
display(feature_importance_df.head(10))

print("\n--- Extracted Decision Logic Rules ---")
print(tree_rules_text)


# ============================================================
# 11. EXPORT TO EXCEL
# ============================================================

clean_q_tag = re.sub(r'[^A-Za-z0-9]', '', str(canonical_label))
output_file = f"/content/{clean_q_tag}_CodeDiff_Cluster{c_a}_Cluster{c_b}_Analysis.xlsx"

print("\n" + "=" * 70)
print(f"CREATING EXCEL REPORT: {output_file}")
print("=" * 70)

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    target_df.to_excel(writer, sheet_name=f"{clean_q_tag}_Data", index=False)
    c_a_rep_out.to_excel(writer, sheet_name=f"C{c_a}_Representatives", index=False)
    c_b_rep_out.to_excel(writer, sheet_name=f"C{c_b}_Representatives", index=False)
    c_a_internal_df.to_excel(writer, sheet_name=f"C{c_a}_Internal_Diffs", index=False)
    c_b_internal_df.to_excel(writer, sheet_name=f"C{c_b}_Internal_Diffs", index=False)
    inter_cluster_df.to_excel(writer, sheet_name=f"C{c_a}_vs_C{c_b}_Diffs", index=False)
    diff_summary_df.to_excel(writer, sheet_name="Diff_Summary", index=False)
    operation_summary_df.to_excel(writer, sheet_name="Edit_Operation_Summary", index=False)
    most_different_df.to_excel(writer, sheet_name="Most_Different_Pairs", index=False)
    no_change_df.to_excel(writer, sheet_name="No_Change_Pairs", index=False)
    failed_diffs_df.to_excel(writer, sheet_name="Failed_Diffs", index=False)
    feature_importance_df.to_excel(writer, sheet_name="Feature_Importance", index=False)
    rules_df.to_excel(writer, sheet_name="Decision_Rules", index=False)

print("\nProcessing complete! Initiating download...")
files.download(output_file)