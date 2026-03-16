"""
Training data for all RAVEN ML models.

  OPERATOR_TRAIN_DATA  — (phrase, canonical_operator) pairs
  BOUNDARY_SENTENCES   — tokenised condition sequences for CRF training
  BOUNDARY_LABELS      — BIO labels aligned with BOUNDARY_SENTENCES
  BM25_CORPUS          — field names + operator strings for autocomplete
"""

# ------------------------------------------------------------------ #
# OperatorClassifier training data                                     #
# ------------------------------------------------------------------ #

OPERATOR_TRAIN_DATA = [
    # lessThan
    ("under", "lessThan"),
    ("below", "lessThan"),
    ("lower than", "lessThan"),
    ("less than", "lessThan"),
    ("smaller than", "lessThan"),
    ("beneath", "lessThan"),
    # lessThanOrEqual
    ("at most", "lessThanOrEqual"),
    ("no more than", "lessThanOrEqual"),
    ("up to", "lessThanOrEqual"),
    ("maximum", "lessThanOrEqual"),
    ("max", "lessThanOrEqual"),
    ("not exceeding", "lessThanOrEqual"),
    # greaterThan
    ("greater than", "greaterThan"),
    ("above", "greaterThan"),
    ("over", "greaterThan"),
    ("more than", "greaterThan"),
    ("exceeds", "greaterThan"),
    ("higher than", "greaterThan"),
    # greaterThanOrEqual
    ("at least", "greaterThanOrEqual"),
    ("minimum", "greaterThanOrEqual"),
    ("min", "greaterThanOrEqual"),
    ("no less than", "greaterThanOrEqual"),
    ("not less than", "greaterThanOrEqual"),
    ("or more", "greaterThanOrEqual"),
    # equals
    ("equals", "equals"),
    ("is", "equals"),
    ("equal to", "equals"),
    ("same as", "equals"),
    ("=", "equals"),
    # notEquals
    ("is not", "notEquals"),
    ("not equal", "notEquals"),
    ("not equal to", "notEquals"),
    ("differs from", "notEquals"),
    ("!=", "notEquals"),
    # contains
    ("contains", "contains"),
    ("includes", "contains"),
    ("has", "contains"),
    ("with", "contains"),
    # notContains
    ("does not contain", "notContains"),
    ("excludes", "notContains"),
    ("without", "notContains"),
    # in
    ("in", "in"),
    ("one of", "in"),
    ("any of", "in"),
    # notIn
    ("not in", "notIn"),
    ("none of", "notIn"),
    ("not one of", "notIn"),
]

# Alias for backward compatibility
TRAIN_DATA = OPERATOR_TRAIN_DATA


# ------------------------------------------------------------------ #
# BoundaryDetector training data  (BIO scheme)                        #
# ------------------------------------------------------------------ #
#
# Labels:
#   O        — not part of a logical group boundary
#   B-GROUP  — beginning of a parenthesised / nested group
#   I-GROUP  — inside a parenthesised / nested group
#

BOUNDARY_SENTENCES: list[list[str]] = [
    # "age > 30 AND status = active"
    ["age", ">", "30", "AND", "status", "=", "active"],
    # "(age > 30) AND status = active"
    ["(", "age", ">", "30", ")", "AND", "status", "=", "active"],
    # "age > 30 AND (status = active OR role = admin)"
    ["age", ">", "30", "AND", "(", "status", "=", "active", "OR", "role", "=", "admin", ")"],
    # "NOT (age < 18)"
    ["NOT", "(", "age", "<", "18", ")"],
    # "score >= 80 AND level = gold"
    ["score", ">=", "80", "AND", "level", "=", "gold"],
    # "region = US OR (tier = premium AND balance > 500)"
    ["region", "=", "US", "OR", "(", "tier", "=", "premium", "AND", "balance", ">", "500", ")"],
    # "status = active"
    ["status", "=", "active"],
    # "(a = 1 AND b = 2) OR c = 3"
    ["(", "a", "=", "1", "AND", "b", "=", "2", ")", "OR", "c", "=", "3"],
]

BOUNDARY_LABELS: list[list[str]] = [
    ["O", "O", "O", "O", "O", "O", "O"],
    ["B-GROUP", "I-GROUP", "I-GROUP", "I-GROUP", "I-GROUP", "O", "O", "O", "O"],
    ["O", "O", "O", "O", "B-GROUP", "I-GROUP", "I-GROUP", "I-GROUP", "I-GROUP", "I-GROUP", "I-GROUP", "I-GROUP", "I-GROUP"],
    ["O", "B-GROUP", "I-GROUP", "I-GROUP", "I-GROUP", "I-GROUP"],
    ["O", "O", "O", "O", "O", "O", "O"],
    ["O", "O", "O", "O", "B-GROUP", "I-GROUP", "I-GROUP", "I-GROUP", "I-GROUP", "I-GROUP", "I-GROUP", "I-GROUP", "I-GROUP"],
    ["O", "O", "O"],
    ["B-GROUP", "I-GROUP", "I-GROUP", "I-GROUP", "I-GROUP", "I-GROUP", "I-GROUP", "I-GROUP", "O", "O", "O", "O", "O"],
]


# ------------------------------------------------------------------ #
# BM25Autocomplete corpus                                              #
# ------------------------------------------------------------------ #

BM25_CORPUS: list[str] = [
    # Common field names
    "accountBalance",
    "age",
    "status",
    "region",
    "country",
    "tier",
    "role",
    "score",
    "level",
    "balance",
    "gender",
    "email",
    "username",
    "createdAt",
    "updatedAt",
    "isActive",
    "isPremium",
    # Canonical operator names
    "lessThan",
    "lessThanOrEqual",
    "greaterThan",
    "greaterThanOrEqual",
    "equals",
    "notEquals",
    "contains",
    "notContains",
    "in",
    "notIn",
    # Logical keywords
    "AND",
    "OR",
    "NOT",
]