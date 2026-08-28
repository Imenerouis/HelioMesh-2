from pathlib import Path

path = Path(
    r"validation\real_spacecraft\opssat_ad\phase3_ablation.py"
)

text = path.read_text(encoding="utf-8")

start_marker = """    # ---------------------------------------------------------------
    # C -> D incremental analysis
    # Use the EXACT C policy already computed above.
"""

end_marker = """    # ═══════════════════════════════════════════════════════════════════════
    # D — FULL HELIOMESH
    # ═══════════════════════════════════════════════════════════════════════
"""

if start_marker not in text:
    raise RuntimeError(
        "C->D incremental analysis block not found."
    )

if end_marker not in text:
    raise RuntimeError(
        "D marker not found."
    )

start = text.index(start_marker)
end = text.index(end_marker)

block = text[start:end]

# Remove C->D block from its current invalid position.
text = text[:start] + text[end:]

# Insert it immediately before the TABLE section, after D is fully built.
table_marker = """    # ═══════════════════════════════════════════════════════════════════════
    # TABLE
    # ═══════════════════════════════════════════════════════════════════════
"""

if table_marker not in text:
    raise RuntimeError(
        "TABLE marker not found."
    )

text = text.replace(
    table_marker,
    block + "\n" + table_marker,
    1
)

path.write_text(
    text,
    encoding="utf-8"
)

print("ORDER_FIX_OK")
print(path)