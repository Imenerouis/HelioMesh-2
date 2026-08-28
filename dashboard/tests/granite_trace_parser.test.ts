/**
 * Unit tests for the deterministic GraniteTrace header-to-header parser logic.
 *
 * These tests replicate the exact algorithm used in GraniteTrace (page.tsx) so
 * they can run outside the React environment with plain Node/jest.
 *
 * Run: npx jest dashboard/tests/granite_trace_parser.test.ts
 */

// ---------------------------------------------------------------------------
// Parser implementation (mirrors page.tsx GraniteTrace exactly)
// ---------------------------------------------------------------------------
interface Section { key: string }
interface ParsedPart { label: string; content: string }

const SECTIONS: Section[] = [
  { key: "OBSERVATION" },
  { key: "PREDICTION" },
  { key: "MODEL STATUS" },
  { key: "EVIDENCE" },
  { key: "RECOMMENDED ACTION" },
  { key: "CONFIDENCE" },
  { key: "REASON" },
];

function parseGraniteTrace(text: string): ParsedPart[] | null {
  if (!text) return null;

  type SectionHit = { secIdx: number; headerEnd: number; textStart: number };
  const hits: SectionHit[] = [];
  const upper = text.toUpperCase();

  for (let si = 0; si < SECTIONS.length; si++) {
    const key = SECTIONS[si].key;
    const pat = new RegExp(
      `(?:^|\\n)[ \\t]*(?:\\d+\\.\\s*)?${key.replace(/ /g, "[ \\t]+")}[ \\t]*:?`,
      "m"
    );
    const m = pat.exec(upper);
    if (m !== null) {
      const headerEnd = m.index + m[0].length;
      hits.push({ secIdx: si, headerEnd, textStart: m.index });
    }
  }

  if (hits.length === 0) return null;

  hits.sort((a, b) => a.textStart - b.textStart);

  const parts: ParsedPart[] = [];
  for (let i = 0; i < hits.length; i++) {
    const hit = hits[i];
    const contentStart = hit.headerEnd;
    const contentEnd = i + 1 < hits.length ? hits[i + 1].textStart : text.length;
    const content = text.slice(contentStart, contentEnd).trim();
    parts.push({ label: SECTIONS[hit.secIdx].key, content });
  }
  return parts;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function getSection(parts: ParsedPart[], key: string): string | undefined {
  return parts.find(p => p.label === key)?.content;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("GraniteTrace deterministic parser", () => {

  // T1 — happy path: all 7 sections in order
  test("T1: parses all 7 sections in canonical order", () => {
    const text = `
OBSERVATION: KP index is 2.1. System nominal.
PREDICTION: Mild geomagnetic activity expected.
MODEL STATUS: RF=NOMINAL GB=NOMINAL Agreement=HIGH
EVIDENCE: KP stable, orbit deviation minimal.
RECOMMENDED ACTION: Continue nominal operations.
CONFIDENCE: 92%
REASON: All indicators within safe bounds.
`.trim();
    const parts = parseGraniteTrace(text)!;
    expect(parts).toHaveLength(7);
    expect(parts.map(p => p.label)).toEqual([
      "OBSERVATION", "PREDICTION", "MODEL STATUS", "EVIDENCE",
      "RECOMMENDED ACTION", "CONFIDENCE", "REASON",
    ]);
  });

  // T2 — numbered headers (1. OBSERVATION:)
  test("T2: handles numbered section headers", () => {
    const text = `
1. OBSERVATION: KP=4.2, elevated.
2. PREDICTION: Storm likely.
3. MODEL STATUS: RF=CRITICAL GB=CRITICAL_AHEAD
4. EVIDENCE: KP spike confirmed.
5. RECOMMENDED ACTION: Safe mode.
6. CONFIDENCE: 87%
7. REASON: Both models concur.
`.trim();
    const parts = parseGraniteTrace(text)!;
    expect(parts).toHaveLength(7);
    expect(getSection(parts, "OBSERVATION")).toBe("KP=4.2, elevated.");
    expect(getSection(parts, "REASON")).toBe("Both models concur.");
  });

  // T3 — preamble echoes a section keyword mid-sentence → must NOT mis-assign
  test("T3: preamble echo of header keyword does not mis-assign content", () => {
    // "observation" appears in the preamble but is NOT a header line
    const text = `Here is my observation of the current state.
OBSERVATION: KP index is 1.5, fully nominal.
PREDICTION: No change expected.
MODEL STATUS: RF=NOMINAL GB=NOMINAL Agreement=HIGH
EVIDENCE: KP stable below 2.
RECOMMENDED ACTION: No action required.
CONFIDENCE: 95%
REASON: Consistent nominal readings.`;
    const parts = parseGraniteTrace(text)!;
    expect(parts).toHaveLength(7);
    // Content must begin with the actual value, not the preamble tail
    expect(getSection(parts, "OBSERVATION")).toBe("KP index is 1.5, fully nominal.");
  });

  // T4 — missing sections: only 5 present
  test("T4: partial response — only present sections are returned", () => {
    const text = `
OBSERVATION: KP=3.0
PREDICTION: Mild storm possible.
EVIDENCE: Elevated KP trend.
CONFIDENCE: 70%
REASON: Partial data available.
`.trim();
    const parts = parseGraniteTrace(text)!;
    expect(parts).toHaveLength(5);
    const labels = parts.map(p => p.label);
    expect(labels).toContain("OBSERVATION");
    expect(labels).not.toContain("MODEL STATUS");
    expect(labels).not.toContain("RECOMMENDED ACTION");
  });

  // T5 — empty/null text → returns null
  test("T5: empty text returns null (plain-text fallback)", () => {
    expect(parseGraniteTrace("")).toBeNull();
  });

  // T6 — no section headers at all → returns null
  test("T6: unstructured text returns null (plain-text fallback)", () => {
    const text = "The satellite is operating normally. No anomalies detected.";
    expect(parseGraniteTrace(text)).toBeNull();
  });

  // T7 — multi-line content per section
  test("T7: multi-line section content is preserved correctly", () => {
    const text = `OBSERVATION: KP index is currently 3.8.
Trend is upward over the last 30 minutes.
Delta-KP = +1.8 from window start.
PREDICTION: CRITICAL_AHEAD forecast by gradient boost.
MODEL STATUS: RF=NOMINAL GB=CRITICAL_AHEAD Agreement=DISAGREE
EVIDENCE: KP rose from 2.0 to 3.8 in six steps.
RECOMMENDED ACTION: Request operator approval before executing maneuver.
CONFIDENCE: 91%
REASON: Model disagreement requires human review.`;
    const parts = parseGraniteTrace(text)!;
    expect(parts).toHaveLength(7);
    const obs = getSection(parts, "OBSERVATION")!;
    expect(obs).toContain("KP index is currently 3.8.");
    expect(obs).toContain("Delta-KP = +1.8");
    expect(obs).not.toContain("PREDICTION"); // must not bleed into next section
  });

  // T8 — section keyword appears inside body text (not as a header)
  test("T8: keyword inside body text is not treated as a new section", () => {
    const text = `OBSERVATION: KP is nominal. The prediction models agree.
PREDICTION: No storm expected. This is a reason for confidence.
MODEL STATUS: Both models nominal.
EVIDENCE: KP < 3 for 12 hours.
RECOMMENDED ACTION: Continue.
CONFIDENCE: 96%
REASON: All clear.`;
    const parts = parseGraniteTrace(text)!;
    // "prediction" and "reason" appear in OBSERVATION body — must not create extra hits
    expect(parts).toHaveLength(7);
    expect(getSection(parts, "OBSERVATION")).toBe("KP is nominal. The prediction models agree.");
  });

  // T9 — case-insensitive header matching
  test("T9: lowercase or mixed-case headers are matched", () => {
    const text = `observation: KP=2.5
prediction: Stable.
model status: RF=NOMINAL GB=NOMINAL Agreement=HIGH
evidence: KP low.
recommended action: None.
confidence: 90%
reason: Normal.`;
    const parts = parseGraniteTrace(text)!;
    expect(parts).toHaveLength(7);
    // Labels are always uppercased from SECTIONS[].key
    expect(parts.map(p => p.label)).toEqual([
      "OBSERVATION", "PREDICTION", "MODEL STATUS", "EVIDENCE",
      "RECOMMENDED ACTION", "CONFIDENCE", "REASON",
    ]);
  });

  // T10 — sections out of template order
  test("T10: sections returned in text-occurrence order when out of template order", () => {
    const text = `REASON: No anomaly detected.
OBSERVATION: KP=1.2
CONFIDENCE: 99%
PREDICTION: Calm.
MODEL STATUS: RF=NOMINAL GB=NOMINAL Agreement=HIGH
EVIDENCE: Historical baseline.
RECOMMENDED ACTION: Maintain course.`;
    const parts = parseGraniteTrace(text)!;
    expect(parts).toHaveLength(7);
    // First in text = REASON
    expect(parts[0].label).toBe("REASON");
    expect(parts[1].label).toBe("OBSERVATION");
    expect(parts[2].label).toBe("CONFIDENCE");
  });

  // T11 — leading whitespace / tab-indented headers
  test("T11: tab-indented headers are matched", () => {
    const text = `\tOBSERVATION: KP nominal.
\tPREDICTION: Stable.
\tMODEL STATUS: RF=NOMINAL GB=NOMINAL Agreement=HIGH
\tEVIDENCE: KP low.
\tRECOMMENDED ACTION: None.
\tCONFIDENCE: 88%
\tREASON: Clean.`;
    const parts = parseGraniteTrace(text)!;
    expect(parts).toHaveLength(7);
  });

  // T12 — RECOMMENDED ACTION (two-word key with space) must parse correctly
  test("T12: multi-word key RECOMMENDED ACTION parsed correctly", () => {
    const text = `OBSERVATION: KP=2.0
PREDICTION: Nominal.
MODEL STATUS: RF=NOMINAL GB=NOMINAL Agreement=HIGH
EVIDENCE: Stable.
RECOMMENDED ACTION: No action required.
CONFIDENCE: 94%
REASON: All clear.`;
    const parts = parseGraniteTrace(text)!;
    expect(getSection(parts, "RECOMMENDED ACTION")).toBe("No action required.");
  });

});
