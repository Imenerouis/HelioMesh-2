from pathlib import Path

path = Path(
    r"validation\real_spacecraft\opssat_ad\phase3_ablation.py"
)

text = path.read_text(encoding="utf-8")

# Fix broken multiline print statements introduced by the previous patch.
text = text.replace(
    'print("\n--- Lead-Time Incremental Value: C vs D ---")',
    'print("\\n--- Lead-Time Incremental Value: C vs D ---")'
)

# Also handle the exact broken form where the newline was physically
# inserted inside the string literal.
text = text.replace(
    'print("\n--- Lead-Time Incremental Value: C vs D ---")',
    'print("\\n--- Lead-Time Incremental Value: C vs D ---")'
)

# Defensive repair for a physically split literal:
text = text.replace(
    'print("\n'
    '--- Lead-Time Incremental Value: C vs D ---")',
    'print("\\n--- Lead-Time Incremental Value: C vs D ---")'
)

path.write_text(
    text,
    encoding="utf-8"
)

print("SYNTAX_FIX_APPLIED")