"use client";

import { useState, useEffect, useRef } from "react";

const API = "http://localhost:8000";

// â”€â”€ Unified color system â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
const STATUS: Record<
  string,
  { label: string; color: string; dot: string }
> = {
  auto_executed: {
    label: 'AUTO ROUTE SELECTED',
    color: "#00ff88",
    dot: "#00ff88",
  },
  pending_approval: {
    label: "PENDING APPROVAL",
    color: "#ffcc00",
    dot: "#ffcc00",
  },
  escalated: {
    label: "ESCALATED",
    color: "#ff3344",
    dot: "#ff3344",
  },
  rejected: {
    label: "REJECTED",
    color: "#4a7fa0",
    dot: "#4a7fa0",
  },
};

const RISK_COLOR: Record<string, string> = {
  CRITICAL: "#ff3344",
  HIGH: "#ff8800",
  MODERATE: "#ffcc00",
  LOW: "#00ff88",
};

// â”€â”€ Loading pipeline steps â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
const PIPELINE_STEPS = [
  "Loading simulated space weather data...",
  "Running orbital simulation...",
  "Snapshot ML: Random Forest classifying current state...",
  "Temporal ML: Gradient Boosting forecasting t+30 min...",
  "Decision Engine evaluating confidence + risk...",
  "Connecting to IBM watsonx...",
  "IBM Granite explaining the determined decision...",
  "Mission Decision Ready âœ“",
];

// â”€â”€ Sub-components â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
function Dot({
  color,
  pulse = false,
}: {
  color: string;
  pulse?: boolean;
}) {
  return (
    <span
      className={
        pulse
          ? color === "#00ff88"
            ? "dot-pulse"
            : "dot-pulse-yellow"
          : ""
      }
      style={{
        display: "inline-block",
        width: 8,
        height: 8,
        borderRadius: "50%",
        background: color,
        boxShadow: `0 0 5px ${color}`,
        marginRight: 7,
        flexShrink: 0,
      }}
    />
  );
}

function Panel({
  children,
  style = {},
}: {
  children: React.ReactNode;
  style?: React.CSSProperties;
}) {
  return (
    <div
      style={{
        background: "var(--bg-card)",
        border: "1px solid var(--border)",
        borderRadius: 3,
        position: "relative",
        overflow: "hidden",
        ...style,
      }}
    >
      <span
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          width: 10,
          height: 10,
          borderTop: "1px solid var(--accent)",
          borderLeft: "1px solid var(--accent)",
        }}
      />
      <span
        style={{
          position: "absolute",
          top: 0,
          right: 0,
          width: 10,
          height: 10,
          borderTop: "1px solid var(--accent)",
          borderRight: "1px solid var(--accent)",
        }}
      />
      <span
        style={{
          position: "absolute",
          bottom: 0,
          left: 0,
          width: 10,
          height: 10,
          borderBottom: "1px solid var(--accent)",
          borderLeft: "1px solid var(--accent)",
        }}
      />
      <span
        style={{
          position: "absolute",
          bottom: 0,
          right: 0,
          width: 10,
          height: 10,
          borderBottom: "1px solid var(--accent)",
          borderRight: "1px solid var(--accent)",
        }}
      />
      {children}
    </div>
  );
}

function Label({ children }: { children: React.ReactNode }) {
  return (
    <div
      style={{
        fontSize: 10,
        letterSpacing: "0.2em",
        color: "var(--accent)",
        borderBottom: "1px solid var(--border)",
        padding: "7px 14px",
        textTransform: "uppercase",
        fontWeight: 600,
        background: "rgba(0,170,255,0.04)",
        fontFamily: "var(--font-mono)",
      }}
    >
      {children}
    </div>
  );
}

// â”€â”€ Progress Bar â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
function ProgressBar({
  value,
  max = 100,
  color,
  label,
  showValue = true,
}: {
  value: number;
  max?: number;
  color: string;
  label?: string;
  showValue?: boolean;
}) {
  const pct = Math.min(100, (value / max) * 100);

  return (
    <div style={{ width: "100%" }}>
      {label && (
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            marginBottom: 4,
          }}
        >
          <span
            style={{
              fontSize: 10,
              color: "var(--text-muted)",
              letterSpacing: "0.12em",
            }}
          >
            {label}
          </span>
          {showValue && (
            <span style={{ fontSize: 10, color, fontWeight: 600 }}>
              {value}
            </span>
          )}
        </div>
      )}

      <div
        style={{
          height: 4,
          background: "var(--border)",
          borderRadius: 2,
          overflow: "hidden",
        }}
      >
        <div
          style={{
            height: "100%",
            width: `${pct}%`,
            background: color,
            boxShadow: `0 0 6px ${color}88`,
            borderRadius: 2,
            transition: "width 0.6s ease",
          }}
        />
      </div>
    </div>
  );
}

// â”€â”€ Risk Breakdown Panel â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
function RiskBreakdown({
  breakdown,
}: {
  breakdown: Record<string, number>;
}) {
  const rows = [
    {
      key: "solar_weather",
      label: "SOLAR WEATHER",
      max: 40,
      color: "#ff8800",
    },
    {
      key: "orbital_instability",
      label: "ORBITAL INSTABILITY",
      max: 30,
      color: "#ff3344",
    },
    {
      key: "power_degradation",
      label: "POWER DEGRADATION",
      max: 20,
      color: "#ffcc00",
    },
    {
      key: "solar_wind",
      label: "SOLAR WIND",
      max: 10,
      color: "#00aaff",
    },
  ];

  const total = breakdown?.total ?? 0;

  const totalColor =
    total >= 75
      ? "#ff3344"
      : total >= 50
        ? "#ff8800"
        : total >= 25
          ? "#ffcc00"
          : "#00ff88";

  return (
    <div style={{ padding: "14px 18px" }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "baseline",
          marginBottom: 14,
        }}
      >
        <span
          style={{
            fontSize: 10,
            color: "var(--text-muted)",
            letterSpacing: "0.15em",
          }}
        >
          OPERATIONAL RISK SCORE
        </span>

        <span
          style={{
            fontSize: 28,
            fontWeight: 700,
            color: totalColor,
            textShadow: `0 0 16px ${totalColor}55`,
          }}
        >
          {total}
          <span
            style={{
              fontSize: 13,
              color: "var(--text-dim)",
            }}
          >
            /100
          </span>
        </span>
      </div>

      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: 10,
        }}
      >
        {rows.map((r) => (
          <div key={r.key}>
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                marginBottom: 4,
              }}
            >
              <span
                style={{
                  fontSize: 10,
                  color: "var(--text-muted)",
                  letterSpacing: "0.12em",
                }}
              >
                {r.label}
              </span>

              <span
                style={{
                  fontSize: 10,
                  color: r.color,
                  fontWeight: 600,
                }}
              >
                +{breakdown?.[r.key] ?? 0}
              </span>
            </div>

            <div
              style={{
                height: 3,
                background: "var(--border)",
                borderRadius: 2,
                overflow: "hidden",
              }}
            >
              <div
                style={{
                  height: "100%",
                  width: `${Math.min(
                    100,
                    ((breakdown?.[r.key] ?? 0) / r.max) * 100
                  )}%`,
                  background: r.color,
                  boxShadow: `0 0 5px ${r.color}88`,
                  borderRadius: 2,
                  transition: "width 0.6s ease",
                }}
              />
            </div>
          </div>
        ))}

        <div
          style={{
            borderTop: "1px solid var(--border)",
            paddingTop: 10,
            display: "flex",
            justifyContent: "flex-end",
          }}
        >
          <span
            style={{
              fontSize: 11,
              color: "var(--text-muted)",
              marginRight: 8,
            }}
          >
            OPERATIONAL RISK
          </span>

          <span
            style={{
              fontSize: 11,
              fontWeight: 700,
              color: totalColor,
            }}
          >
            {total}/100
          </span>
        </div>
      </div>
    </div>
  );
}

// â”€â”€ Granite Trace Formatter â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
function GraniteTrace({ text }: { text: string }) {
  if (!text) {
    return (
      <span style={{ color: "var(--text-dim)" }}>
        No trace available.
      </span>
    );
  }
  // Graceful fallback when IBM Granite Runtime is unavailable
  const graniteUnavailable =
    /invalid_instance_status_error|status_code['"]?\s*:\s*403|current status:\s*Inactive/i.test(
      text
    );

  if (graniteUnavailable) {
    return (
      <div
        style={{
          padding: "10px 12px",
          borderLeft: "2px solid #ffcc00",
          background: "rgba(255,204,0,0.06)",
          borderRadius: "0 3px 3px 0",
        }}
      >
        <div
          style={{
            fontSize: 9,
            color: "#ffcc00",
            fontWeight: 700,
            letterSpacing: "0.18em",
            marginBottom: 6,
          }}
        >
          IBM GRANITE — UNAVAILABLE
        </div>

        <div
          style={{
            fontSize: 12,
            color: "var(--text-primary)",
            lineHeight: 1.6,
          }}
        >
          Runtime service is currently inactive.
        </div>

        <div
          style={{
            fontSize: 11,
            color: "var(--text-dim)",
            marginTop: 4,
            lineHeight: 1.6,
          }}
        >
          Decision generated safely by the deterministic policy engine.
        </div>
      </div>
    );
  }
  const SECTIONS: {
    key: string;
    color: string;
    bg: string;
  }[] = [
    {
      key: "OBSERVATION",
      color: "#00aaff",
      bg: "rgba(0,170,255,0.08)",
    },
    {
      key: "PREDICTION",
      color: "#ffcc00",
      bg: "rgba(255,204,0,0.08)",
    },
    {
      key: "MODEL STATUS",
      color: "#a78bfa",
      bg: "rgba(167,139,250,0.08)",
    },
    {
      key: "EVIDENCE",
      color: "#00ccff",
      bg: "rgba(0,204,255,0.08)",
    },
    {
      key: "RECOMMENDED ACTION",
      color: "#00ff88",
      bg: "rgba(0,255,136,0.08)",
    },
    {
      key: "CONFIDENCE",
      color: "#a78bfa",
      bg: "rgba(167,139,250,0.08)",
    },
    {
      key: "REASON",
      color: "#ff8800",
      bg: "rgba(255,136,0,0.08)",
    },
  ];

  type SectionHit = {
    secIdx: number;
    headerEnd: number;
    textStart: number;
  };

  const hits: SectionHit[] = [];
  const upper = text.toUpperCase();

  for (let si = 0; si < SECTIONS.length; si++) {
    const key = SECTIONS[si].key;

    const pat = new RegExp(
      `(?:^|\\n)[ \\t]*(?:\\d+\\.\\s*)?${key.replace(
        / /g,
        "[ \\t]+"
      )}[ \\t]*:?`,
      "m"
    );

    const m = pat.exec(upper);

    if (m !== null) {
      const headerEnd = m.index + m[0].length;

      hits.push({
        secIdx: si,
        headerEnd,
        textStart: m.index,
      });
    }
  }

  if (hits.length === 0) {
    return (
      <div
        style={{
          fontSize: 12,
          color: "var(--text-primary)",
          whiteSpace: "pre-wrap",
          lineHeight: 1.75,
        }}
      >
        {text}
      </div>
    );
  }

  hits.sort((a, b) => a.textStart - b.textStart);

  const parts: {
    label: string;
    content: string;
    color: string;
    bg: string;
  }[] = [];

  for (let i = 0; i < hits.length; i++) {
    const hit = hits[i];

    const contentStart = hit.headerEnd;

    const contentEnd =
      i + 1 < hits.length
        ? hits[i + 1].textStart
        : text.length;

    const sec = SECTIONS[hit.secIdx];

    const content = text
      .slice(contentStart, contentEnd)
      .trim();

    parts.push({
      label: sec.key,
      content,
      color: sec.color,
      bg: sec.bg,
    });
  }

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 8,
      }}
    >
      {parts.map((p, i) => (
        <div
          key={i}
          style={{
            background: p.bg,
            borderLeft: `2px solid ${p.color}`,
            borderRadius: "0 3px 3px 0",
            padding: "8px 12px",
          }}
        >
          <div
            style={{
              fontSize: 9,
              color: p.color,
              fontWeight: 700,
              letterSpacing: "0.2em",
              marginBottom: 4,
            }}
          >
            {p.label}
          </div>

          <div
            style={{
              fontSize: 12,
              color: "var(--text-primary)",
              lineHeight: 1.7,
              whiteSpace: "pre-wrap",
            }}
          >
            {p.content}
          </div>
        </div>
      ))}
    </div>
  );
}

function Clock() {
  const [time, setTime] = useState("");

  useEffect(() => {
    const fmt = () => {
      const now = new Date();

      const pad = (n: number) =>
        String(n).padStart(2, "0");

      return `${now.getUTCFullYear()}-${pad(
        now.getUTCMonth() + 1
      )}-${pad(now.getUTCDate())} ${pad(
        now.getUTCHours()
      )}:${pad(now.getUTCMinutes())}:${pad(
        now.getUTCSeconds()
      )} UTC`;
    };

    setTime(fmt());

    const id = setInterval(
      () => setTime(fmt()),
      1000
    );

    return () => clearInterval(id);
  }, []);

  return (
    <span
      style={{
        color: "var(--accent)",
        fontWeight: 600,
        fontSize: 12,
        fontFamily: "var(--font-mono)",
      }}
    >
      {time}
    </span>
  );
}

// â”€â”€ Loading overlay â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
function LoadingOverlay({
  steps,
}: {
  steps: string[];
}) {
  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 200,
        background: "rgba(2,11,24,0.92)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      <Panel style={{ padding: 0, minWidth: 420 }}>
        <Label>
          MISSION PIPELINE — PROCESSING
        </Label>

        <div style={{ padding: "24px 28px" }}>
          {PIPELINE_STEPS.map((step, i) => {
            const done = i < steps.length;
            const active =
              i === steps.length - 1;

            if (i > steps.length) return null;

            return (
              <div
                key={i}
                className="step-in"
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 12,
                  marginBottom: 14,
                  animationDelay: `${i * 0.15}s`,
                  opacity: done ? 1 : 0.3,
                }}
              >
                <span
                  style={{
                    width: 18,
                    height: 18,
                    borderRadius: "50%",
                    flexShrink: 0,
                    border: `1px solid ${
                      active
                        ? "#ffcc00"
                        : done
                          ? "#00ff88"
                          : "var(--border)"
                    }`,
                    background:
                      done && !active
                        ? "rgba(0,255,136,0.15)"
                        : "transparent",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: 10,
                  }}
                >
                  {done && !active
                    ? "✓"
                    : active
                      ? "•"
                      : ""}
                </span>

                <span
                  style={{
                    fontSize: 12,
                    letterSpacing: "0.05em",
                    color: active
                      ? "#ffcc00"
                      : done
                        ? "#00ff88"
                        : "var(--text-dim)",
                    fontFamily:
                      "var(--font-mono)",
                  }}
                >
                  {step}
                </span>
              </div>
            );
          })}
        </div>
      </Panel>
    </div>
  );
}

// â”€â”€ Architecture Diagram â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
function ArchDiagram() {
  const nodes = [
    {
      label: "Space Weather Data",
      sub: "Simulated OMNI-2 inputs",
      color: "#00aaff",
    },
    {
      label: "Orbital Simulator",
      sub: "Simplified physics-inspired model",
      color: "#00aaff",
    },
    {
      label: "Random Forest",
      sub: "Current state — 'What is now?'",
      color: "#a78bfa",
    },
    {
      label: "Temporal Predictor",
      sub: "Gradient Boosting — 'What in 30 min?'",
      color: "#a78bfa",
    },
    {
      label: "Decision Engine",
      sub: "Deterministic confidence + risk policy",
      color: "#ffcc00",
    },
    {
      label: "IBM Granite 4",
      sub: "Explanation over the determined decision",
      color: "#7c5cd8",
    },
    {
      label: "Human Oversight",
      sub: "Operator approval for pending decisions",
      color: "#ff8800",
    },
    {
      label: "FastAPI",
      sub: "REST API layer",
      color: "#00aaff",
    },
    {
      label: "Operator Dashboard",
      sub: "React / Next.js",
      color: "#00ff88",
    },
  ];

  return (
    <div
      style={{
        padding: "20px 24px",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 0,
      }}
    >
      {nodes.map((n, i) => (
        <div
          key={i}
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
          }}
        >
          <div
            style={{
              border: `1px solid ${n.color}44`,
              background: `${n.color}0d`,
              borderRadius: 3,
              padding: "9px 28px",
              textAlign: "center",
              minWidth: 260,
            }}
          >
            <div
              style={{
                fontSize: 12,
                fontWeight: 600,
                color: n.color,
                letterSpacing: "0.08em",
              }}
            >
              {n.label}
            </div>

            <div
              style={{
                fontSize: 9,
                color: "var(--text-muted)",
                marginTop: 2,
                letterSpacing: "0.08em",
              }}
            >
              {n.sub}
            </div>
          </div>

          {i < nodes.length - 1 && (
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                margin: "3px 0",
              }}
            >
              <div
                style={{
                  width: 1,
                  height: 10,
                  background: "var(--border-bright)",
                }}
              />

              <div
                style={{
                  color: "var(--text-muted)",
                  fontSize: 9,
                }}
              >
                â–¼
              </div>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

// â”€â”€ Main Component â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
export default function Home() {
  const [telemetry, setTelemetry] =
    useState<any>(null);

  const [prevTelemetry, setPrevTelemetry] =
    useState<any>(null);

  const [decisions, setDecisions] =
    useState<any[]>([]);

  const [selected, setSelected] =
    useState<any>(null);

  const [chat, setChat] =
    useState("");

  const [chatHistory, setChatHistory] =
    useState<{ q: string; a: string }[]>(
      []
    );

  const [loading, setLoading] =
    useState(false);

  const [pipelineSteps, setPipelineSteps] =
    useState<string[]>([]);

  const [view, setView] =
    useState<
      | "dashboard"
      | "detail"
      | "chat"
      | "report"
      | "validation"
      | "opssat"
    >("dashboard");

  const [report, setReport] =
    useState<any>(null);

  const [validation, setValidation] =
    useState<any>(null);

  const [opssatEvidence, setOpssatEvidence] =
    useState<any>(null);

  const [sysStatus, setSysStatus] =
    useState<"ONLINE" | "CONNECTING">(
      "CONNECTING"
    );

  const chatEndRef =
    useRef<HTMLDivElement>(null);

  const fetchTelemetry = async () => {
    try {
      const res = await fetch(
        `${API}/telemetry`
      );

      const data = await res.json();

      setPrevTelemetry(
        (p: any) => p ?? data
      );

      setTelemetry((prev: any) => {
        setPrevTelemetry(prev);
        return data;
      });

      setSysStatus("ONLINE");
    } catch {
      setSysStatus("CONNECTING");
    }
  };

  const fetchDecisions = async () => {
    try {
      const res = await fetch(
        `${API}/decision`
      );

      const data = await res.json();

      setDecisions(
        data.decisions || []
      );
    } catch {}
  };

  // â”€â”€ Animated pipeline â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  const createDecision = async (
    scenario: string
  ) => {
    setLoading(true);
    setPipelineSteps([]);

    const advance = (
      i: number
    ) =>
      new Promise<void>((resolve) => {
        setTimeout(() => {
          setPipelineSteps(
            PIPELINE_STEPS.slice(
              0,
              i + 1
            )
          );
          resolve();
        }, 600 * i);
      });

    for (
      let i = 0;
      i < PIPELINE_STEPS.length - 1;
      i++
    ) {
      await advance(i);
    }

    try {
      await fetch(
        `${API}/decision/new?sail_angle=45&scenario=${scenario}`,
        { method: "POST" }
      );

      await fetchDecisions();

      setPipelineSteps(
        PIPELINE_STEPS
      );

      await new Promise((r) =>
        setTimeout(r, 800)
      );
    } finally {
      setLoading(false);
      setPipelineSteps([]);
    }
  };

  const approveDecision = async (
    id: string
  ) => {
    await fetch(
      `${API}/approve/${id}`,
      { method: "POST" }
    );

    await fetchDecisions();

    if (
      selected?.decision_id === id
    ) {
      const res = await fetch(
        `${API}/decision/${id}`
      );

      setSelected(
        await res.json()
      );
    }
  };

  const rejectDecision = async (
    id: string
  ) => {
    await fetch(
      `${API}/reject/${id}`,
      { method: "POST" }
    );

    await fetchDecisions();

    if (
      selected?.decision_id === id
    ) {
      const res = await fetch(
        `${API}/decision/${id}`
      );

      setSelected(
        await res.json()
      );
    }
  };

  const sendChat = async () => {
    if (!chat.trim()) return;

    setLoading(true);

    const q = chat;
    setChat("");

    try {
      const res = await fetch(
        `${API}/chat?question=${encodeURIComponent(q)}`,
        { method: "POST" }
      );

      const data = await res.json();

      setChatHistory((h) => [
        ...h,
        {
          q,
          a: data.answer,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const fetchReport = async () => {
    try {
      const res = await fetch(
        `${API}/report`
      );

      setReport(
        await res.json()
      );
    } catch {}
  };

  const fetchValidation = async () => {
    try {
      const res = await fetch(
        `${API}/validation`
      );

      if (res.ok) {
        setValidation(
          await res.json()
        );
      }
    } catch {}
  };

  const fetchOpssatEvidence = async () => {
    try {
      const res = await fetch(
        `${API}/opssat/evidence`
      );

      if (res.ok) {
        setOpssatEvidence(
          await res.json()
        );
      }
    } catch {}
  };

  useEffect(() => {
    fetchTelemetry();
    fetchDecisions();

    const id = setInterval(() => {
      fetchTelemetry();
      fetchDecisions();
    }, 15000);

    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({
      behavior: "smooth",
    });
  }, [chatHistory]);

  // â”€â”€ Trend helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  const trend = (key: string) => {
    if (!prevTelemetry || !telemetry)
      return "";

    const diff =
      telemetry[key] -
      prevTelemetry[key];

    if (Math.abs(diff) < 0.001)
      return "";

    return diff > 0 ? " â†‘" : " â†“";
  };

  const trendColor = (
    key: string,
    higherIsBad = false
  ) => {
    if (!prevTelemetry || !telemetry)
      return "var(--text-muted)";

    const diff =
      telemetry[key] -
      prevTelemetry[key];

    if (Math.abs(diff) < 0.001)
      return "var(--text-muted)";

    const up = diff > 0;

    return (up && higherIsBad) ||
      (!up && !higherIsBad)
      ? "#ff3344"
      : "#00ff88";
  };

  const navItems = [
    {
      id: "dashboard",
      label: "MISSION CONTROL",
    },
    {
      id: "chat",
      label: "AI CONSOLE",
    },
    {
      id: "report",
      label: "MISSION REPORT",
    },
    {
      id: "validation",
      label: "VALIDATION",
    },
    {
      id: "opssat",
      label: "OPS-SAT REAL",
    },
  ];

  const mono: React.CSSProperties = {
    fontFamily: "var(--font-mono)",
  };

  const sans: React.CSSProperties = {
    fontFamily: "var(--font-sans)",
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "var(--bg-primary)",
        ...mono,
      }}
    >
      {/* â”€â”€ Loading overlay â”€â”€ */}
      {loading &&
        pipelineSteps.length > 0 && (
          <LoadingOverlay
            steps={pipelineSteps}
          />
        )}

      {/* â”€â”€ TOP BAR â”€â”€ */}
      <header
        style={{
          background: "var(--bg-panel)",
          borderBottom:
            "1px solid var(--border-bright)",
          padding: "0 24px",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          height: 54,
          position: "sticky",
          top: 0,
          zIndex: 100,
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 14,
          }}
        >
          <div
            style={{
              width: 34,
              height: 34,
              borderRadius: "50%",
              border:
                "1.5px solid var(--accent)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              boxShadow:
                "0 0 10px var(--accent-dim)",
              fontSize: 16,
            }}
          >
            🛰️
          </div>

          <div>
            <div
              style={{
                fontSize: 15,
                fontWeight: 700,
                color: "var(--accent)",
                letterSpacing: "0.18em",
                ...mono,
              }}
            >
              HELIOMESH
            </div>

            <div
              style={{
                fontSize: 9,
                color: "var(--text-muted)",
                letterSpacing: "0.22em",
                ...mono,
              }}
            >
              AI-ASSISTED SATELLITE MISSION
              CONTROL SYSTEM
            </div>
          </div>
        </div>

        <nav
          style={{
            display: "flex",
            gap: 3,
          }}
        >
          {navItems.map((n) => (
            <button
              key={n.id}
              onClick={() => {
                setView(n.id as any);

                if (n.id === "report")
                  fetchReport();

                if (n.id === "validation")
                  fetchValidation();

                if (n.id === "opssat")
                  fetchOpssatEvidence();
              }}
              style={{
                background:
                  view === n.id
                    ? "rgba(0,170,255,0.12)"
                    : "transparent",
                border:
                  view === n.id
                    ? "1px solid var(--accent)"
                    : "1px solid transparent",
                color:
                  view === n.id
                    ? "var(--accent)"
                    : "var(--text-muted)",
                padding: "5px 14px",
                fontSize: 10,
                letterSpacing: "0.15em",
                cursor: "pointer",
                borderRadius: 2,
                ...mono,
              }}
            >
              {n.label}
            </button>
          ))}
        </nav>

        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 18,
            fontSize: 11,
          }}
        >
          <div
            style={{
              display: "flex",
              alignItems: "center",
            }}
          >
            <Dot
              color={
                sysStatus === "ONLINE"
                  ? "#00ff88"
                  : "#ffcc00"
              }
              pulse
            />

            <span
              style={{
                color: "var(--text-muted)",
                letterSpacing: "0.1em",
                ...mono,
              }}
            >
              SYS:{" "}
              <span
                style={{
                  color:
                    sysStatus === "ONLINE"
                      ? "#00ff88"
                      : "#ffcc00",
                }}
              >
                {sysStatus}
              </span>
            </span>
          </div>

          <Clock />
        </div>
      </header>

      <div
        style={{
          padding: "18px 24px",
          maxWidth: 1400,
          margin: "0 auto",
        }}
      >
        {/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            DASHBOARD
        â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */}
        {view === "dashboard" && (
          <div>
            {/* Telemetry cards */}
            <div
              style={{
                display: "grid",
                gridTemplateColumns:
                  "repeat(4,1fr)",
                gap: 10,
                marginBottom: 14,
              }}
            >
              {[
                {
                  key: "kp_index",
                  label: "KP INDEX",
                  unit: "",
                  higherIsBad: true,
                },
                {
                  key: "sail_angle",
                  label: "SAIL ANGLE",
                  unit: "°",
                  higherIsBad: false,
                },
                {
                  key: "power_output",
                  label: "POWER OUTPUT",
                  unit: " W",
                  higherIsBad: false,
                },
                {
                  key: "orbit_deviation",
                  label: "ORBIT DEVIATION",
                  unit: " km",
                  higherIsBad: true,
                },
              ].map(
                ({
                  key,
                  label,
                  unit,
                  higherIsBad,
                }) => {
                  const val =
                    telemetry?.[key];

                  const critical =
                    key === "kp_index"
                      ? val > 6
                      : key ===
                          "power_output"
                        ? val < 10
                        : key ===
                            "orbit_deviation"
                          ? val > 1.5
                          : false;

                  const warning =
                    key === "kp_index"
                      ? val > 4
                      : key ===
                          "power_output"
                        ? val < 50
                        : key ===
                            "orbit_deviation"
                          ? val > 0.5
                          : false;

                  const c = critical
                    ? "#ff3344"
                    : warning
                      ? "#ffcc00"
                      : val != null
                        ? "#00ff88"
                        : "var(--accent)";

                  const tr = trend(key);
                  const tc = trendColor(
                    key,
                    higherIsBad
                  );

                  return (
                    <Panel key={key}>
                      <Label>{label}</Label>

                      <div
                        style={{
                          padding:
                            "14px 18px",
                        }}
                      >
                        <div
                          className="value-transition"
                          style={{
                            fontSize: 30,
                            fontWeight: 700,
                            color: c,
                            lineHeight: 1,
                            textShadow: `0 0 18px ${c}44`,
                            ...mono,
                          }}
                        >
                          {val ?? "—"}

                          <span
                            style={{
                              fontSize: 12,
                              color:
                                "var(--text-muted)",
                              marginLeft: 3,
                            }}
                          >
                            {unit}
                          </span>

                          {tr && (
                            <span
                              style={{
                                fontSize: 13,
                                color: tc,
                                marginLeft: 4,
                              }}
                            >
                              {tr}
                            </span>
                          )}
                        </div>
                      </div>
                    </Panel>
                  );
                }
              )}
            </div>

            {/* Mission status */}
            <div
              style={{
                display: "grid",
                gridTemplateColumns:
                  "1fr auto",
                gap: 10,
                marginBottom: 14,
              }}
            >
              <Panel>
                <Label>MISSION STATUS</Label>

                <div
                  style={{
                    padding: "10px 18px",
                    display: "flex",
                    gap: 28,
                    alignItems: "center",
                    flexWrap: "wrap",
                  }}
                >
                  {[
                    {
                      label: "SOLAR WIND",
                      value: `${telemetry?.solar_wind_speed ?? "—"} km/s`,
                    },
                    {
                      label: "DRAG FACTOR",
                      value:
                        telemetry?.drag_factor ??
                        "—",
                    },
                    {
                      label: "THRUST",
                      value:
                        telemetry?.thrust_output ??
                        "—",
                    },
                    {
                      label: "SAT STATUS",
                      value: (
                        telemetry?.status ??
                        "—"
                      ).toUpperCase(),
                      color:
                        telemetry?.status ===
                        "critical"
                          ? "#ff3344"
                          : telemetry?.status ===
                              "warning"
                            ? "#ffcc00"
                            : "#00ff88",
                    },
                    {
                      label: "TOTAL DECISIONS",
                      value:
                        decisions.length,
                      color:
                        "var(--accent)",
                    },
                  ].map((s) => (
                    <div key={s.label}>
                      <div
                        style={{
                          fontSize: 9,
                          color:
                            "var(--text-muted)",
                          letterSpacing:
                            "0.18em",
                          ...mono,
                        }}
                      >
                        {s.label}
                      </div>

                      <div
                        className="value-transition"
                        style={{
                          fontSize: 13,
                          marginTop: 2,
                          color:
                            (s as any)
                              .color ??
                            "var(--text-primary)",
                          ...mono,
                        }}
                      >
                        {s.value}
                      </div>
                    </div>
                  ))}
                </div>
              </Panel>

              <button
                onClick={fetchTelemetry}
                style={{
                  background:
                    "rgba(0,170,255,0.08)",
                  border:
                    "1px solid var(--accent-dim)",
                  color: "var(--accent)",
                  padding: "0 18px",
                  fontSize: 10,
                  letterSpacing: "0.15em",
                  cursor: "pointer",
                  borderRadius: 2,
                  ...mono,
                }}
              >
                âŸ³ REFRESH
              </button>
            </div>

            {/* Scenario launcher */}
            <Panel
              style={{
                marginBottom: 14,
              }}
            >
              <Label>
                SCENARIO LAUNCHER
              </Label>

              <div
                style={{
                  padding: "10px 14px",
                  display: "flex",
                  gap: 8,
                }}
              >
                {[
                  {
                    label: "▶ NORMAL",
                    scenario: "normal",
                    color: "#00ff88",
                  },
                  {
                    label: "⚠ WARNING",
                    scenario: "warning",
                    color: "#ffcc00",
                  },
                  {
                    label: "⛈  STORM",
                    scenario: "storm",
                    color: "#ff3344",
                  },
                ].map((btn) => (
                  <button
                    key={btn.scenario}
                    disabled={loading}
                    onClick={() =>
                      createDecision(
                        btn.scenario
                      )
                    }
                    style={{
                      background: `${btn.color}0d`,
                      border: `1px solid ${btn.color}44`,
                      color: btn.color,
                      padding: "9px 22px",
                      fontSize: 11,
                      letterSpacing:
                        "0.15em",
                      cursor: loading
                        ? "not-allowed"
                        : "pointer",
                      opacity: loading
                        ? 0.4
                        : 1,
                      borderRadius: 2,
                      ...mono,
                    }}
                  >
                    {btn.label} SCENARIO
                  </button>
                ))}
              </div>
            </Panel>

            {/* Decision log */}
            <Panel>
              <Label>
                DECISION LOG —{" "}
                {decisions.length} RECORDS
              </Label>

              {decisions.length ===
              0 ? (
                <div
                  style={{
                    padding: "52px 24px",
                    textAlign: "center",
                  }}
                >
                  <div
                    style={{
                      fontSize: 32,
                      marginBottom: 12,
                    }}
                  >
                    🛰️
                  </div>

                  <div
                    style={{
                      fontSize: 13,
                      color:
                        "var(--text-primary)",
                      marginBottom: 6,
                      ...sans,
                    }}
                  >
                    No mission decisions yet
                  </div>

                  <div
                    style={{
                      fontSize: 11,
                      color:
                        "var(--text-muted)",
                      letterSpacing:
                        "0.1em",
                      ...mono,
                    }}
                  >
                    LAUNCH A SCENARIO TO BEGIN
                  </div>
                </div>
              ) : (
                <div>
                  <div
                    style={{
                      display: "grid",
                      gridTemplateColumns:
                        "160px 1fr 110px 110px 160px 150px",
                      padding:
                        "7px 14px",
                      borderBottom:
                        "1px solid var(--border)",
                      fontSize: 9,
                      color:
                        "var(--text-dim)",
                      letterSpacing:
                        "0.18em",
                    }}
                  >
                    {[
                      "DECISION ID",
                      "TIMESTAMP",
                      "CONFIDENCE",
                      "RISK",
                      "MODE",
                      "STATUS",
                    ].map((h) => (
                      <span key={h}>
                        {h}
                      </span>
                    ))}
                  </div>

                  {[...decisions]
                    .reverse()
                    .map((d, i) => {
                      const sm =
                        STATUS[d.status] ??
                        STATUS.rejected;

                      return (
                        <div
                          key={
                            d.decision_id
                          }
                          onClick={() => {
                            setSelected(d);
                            setView(
                              "detail"
                            );
                          }}
                          style={{
                            display: "grid",
                            gridTemplateColumns:
                              "160px 1fr 110px 110px 160px 150px",
                            padding:
                              "11px 14px",
                            borderBottom:
                              i <
                              decisions.length -
                                1
                                ? "1px solid var(--border)"
                                : "none",
                            cursor:
                              "pointer",
                            fontSize: 12,
                            transition:
                              "background 0.15s",
                          }}
                          onMouseEnter={(e) =>
                            (e.currentTarget.style.background =
                              "var(--bg-card-hover)")
                          }
                          onMouseLeave={(e) =>
                            (e.currentTarget.style.background =
                              "transparent")
                          }
                        >
                          <span
                            style={{
                              color:
                                "var(--accent)",
                              fontWeight: 600,
                            }}
                          >
                            {d.decision_id}
                          </span>

                          <span
                            style={{
                              color:
                                "var(--text-muted)",
                              fontSize: 11,
                            }}
                          >
                            {d.timestamp}
                          </span>

                          <span
                            style={{
                              color:
                                "var(--text-primary)",
                            }}
                          >
                            {d.confidence_score}
                            <span
                              style={{
                                color:
                                  "var(--text-dim)",
                              }}
                            >
                              /100
                            </span>
                          </span>

                          <span
                            style={{
                              color:
                                RISK_COLOR[
                                  d.risk_level
                                ] ??
                                "var(--text-primary)",
                            }}
                          >
                            {d.risk_level}
                          </span>

                          <span
                            style={{
                              color:
                                "var(--text-muted)",
                              fontSize: 11,
                            }}
                          >
                            {d.mission_mode}
                          </span>

                          <span
                            style={{
                              display: "flex",
                              alignItems:
                                "center",
                            }}
                          >
                            <Dot
                              color={sm.dot}
                            />

                            <span
                              style={{
                                color:
                                  sm.color,
                                fontSize: 10,
                                letterSpacing:
                                  "0.08em",
                              }}
                            >
                              {sm.label}
                            </span>
                          </span>
                        </div>
                      );
                    })}
                </div>
              )}
            </Panel>
          </div>
        )}

        {/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            DETAIL
        â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */}
        {view === "detail" &&
          selected &&
          (() => {
            const sm =
              STATUS[selected.status] ??
              STATUS.rejected;

            return (
              <div>
                <button
                  onClick={() =>
                    setView("dashboard")
                  }
                  style={{
                    background: "transparent",
                    border: "none",
                    color:
                      "var(--accent)",
                    fontSize: 11,
                    cursor: "pointer",
                    letterSpacing:
                      "0.15em",
                    marginBottom: 14,
                    padding: 0,
                    ...mono,
                  }}
                >
                  ← BACK TO MISSION CONTROL
                </button>

                <Panel
                  style={{
                    marginBottom: 12,
                  }}
                >
                  <Label>
                    DECISION RECORD
                    {selected.decision_id}
                  </Label>

                  <div
                    style={{
                      padding:
                        "14px 18px",
                      display: "flex",
                      justifyContent:
                        "space-between",
                      alignItems:
                        "center",
                    }}
                  >
                    <div
                      style={{
                        display: "flex",
                        alignItems:
                          "center",
                      }}
                    >
                      <Dot
                        color={sm.dot}
                        pulse
                      />

                      <span
                        style={{
                          color:
                            sm.color,
                          fontSize: 13,
                          fontWeight: 600,
                          letterSpacing:
                            "0.1em",
                        }}
                      >
                        {sm.label}
                      </span>
                    </div>

                    <span
                      style={{
                        color:
                          "var(--text-muted)",
                        fontSize: 11,
                      }}
                    >
                      {selected.timestamp}
                    </span>
                  </div>
                </Panel>

                {/* Dual ML row */}
                {(selected.ml_prediction ||
                  selected.ml_forecast) && (
                  <div
                    style={{
                      display: "grid",
                      gridTemplateColumns:
                        "1fr 1fr",
                      gap: 10,
                      marginBottom: 12,
                    }}
                  >
                    {/* Random Forest */}
                    {selected.ml_prediction &&
                      (() => {
                        const ml =
                          selected.ml_prediction;

                        const stateColor =
                          ml.predicted_state ===
                          "SAFE_MODE"
                            ? "#ff3344"
                            : ml.predicted_state ===
                                "STANDBY"
                              ? "#ffcc00"
                              : "#00ff88";

                        const riskPct =
                          Math.round(
                            ml.risk_probability *
                              100
                          );

                        const confPct =
                          Math.round(
                            ml.model_confidence *
                              100
                          );

                        return (
                          <Panel>
                            <Label>
                              CURRENT STATE —
                              RANDOM FOREST
                            </Label>

                            <div
                              style={{
                                padding:
                                  "10px 14px 4px",
                                display:
                                  "flex",
                                alignItems:
                                  "center",
                                gap: 8,
                                borderBottom:
                                  "1px solid var(--border)",
                              }}
                            >
                              <span
                                style={{
                                  fontSize: 9,
                                  color:
                                    "var(--text-dim)",
                                  letterSpacing:
                                    "0.14em",
                                }}
                              >
                                ANSWERS: WHAT IS THE STATE RIGHT NOW?
                              </span>
                            </div>

                            <div
                              style={{
                                padding:
                                  "14px 18px",
                                display:
                                  "grid",
                                gridTemplateColumns:
                                  "1fr 1fr",
                                gap: 16,
                              }}
                            >
                              <div>
                                <div
                                  style={{
                                    fontSize: 9,
                                    color:
                                      "var(--text-muted)",
                                    letterSpacing:
                                      "0.15em",
                                    marginBottom: 6,
                                  }}
                                >
                                  PREDICTED STATE
                                </div>

                                <div
                                  style={{
                                    fontSize: 20,
                                    fontWeight: 700,
                                    color:
                                      stateColor,
                                    textShadow: `0 0 12px ${stateColor}55`,
                                    marginBottom: 10,
                                  }}
                                >
                                  {ml.predicted_state.replace(
                                    /_/g,
                                    " "
                                  )}
                                </div>

                                <div
                                  style={{
                                    display:
                                      "flex",
                                    flexDirection:
                                      "column",
                                    gap: 5,
                                  }}
                                >
                                  {Object.entries(
                                    ml.probabilities
                                  ).map(
                                    ([
                                      cls,
                                      prob,
                                    ]: [
                                      string,
                                      any
                                    ]) => {
                                      const c =
                                        cls ===
                                        "SAFE_MODE"
                                          ? "#ff3344"
                                          : cls ===
                                              "STANDBY"
                                            ? "#ffcc00"
                                            : "#00ff88";

                                      return (
                                        <div
                                          key={cls}
                                        >
                                          <div
                                            style={{
                                              display:
                                                "flex",
                                              justifyContent:
                                                "space-between",
                                              marginBottom:
                                                2,
                                            }}
                                          >
                                            <span
                                              style={{
                                                fontSize: 9,
                                                color:
                                                  "var(--text-muted)",
                                                letterSpacing:
                                                  "0.08em",
                                              }}
                                            >
                                              {cls.replace(
                                                /_/g,
                                                " "
                                              )}
                                            </span>

                                            <span
                                              style={{
                                                fontSize: 9,
                                                color:
                                                  c,
                                                fontWeight:
                                                  600,
                                              }}
                                            >
                                              {Math.round(
                                                prob *
                                                  100
                                              )}
                                              %
                                            </span>
                                          </div>

                                          <div
                                            style={{
                                              height: 2,
                                              background:
                                                "var(--border)",
                                              borderRadius:
                                                1,
                                            }}
                                          >
                                            <div
                                              style={{
                                                height:
                                                  "100%",
                                                width: `${Math.round(
                                                  prob *
                                                    100
                                                )}%`,
                                                background:
                                                  c,
                                                transition:
                                                  "width 0.6s ease",
                                              }}
                                            />
                                          </div>
                                        </div>
                                      );
                                    }
                                  )}
                                </div>
                              </div>

                              <div
                                style={{
                                  display:
                                    "flex",
                                  flexDirection:
                                    "column",
                                  gap: 12,
                                }}
                              >
                                <div>
                                  <div
                                    style={{
                                      fontSize: 9,
                                      color:
                                        "var(--text-muted)",
                                      letterSpacing:
                                        "0.12em",
                                      marginBottom: 4,
                                    }}
                                  >
                                    NON-NOMINAL STATE PROBABILITY
                                  </div>

                                  <div
                                    style={{
                                      fontSize: 20,
                                      fontWeight: 700,
                                      color:
                                        riskPct >=
                                        75
                                          ? "#ff3344"
                                          : riskPct >=
                                              40
                                            ? "#ffcc00"
                                            : "#00ff88",
                                      marginBottom: 4,
                                    }}
                                  >
                                    {riskPct}%
                                  </div>

                                  <ProgressBar
                                    value={
                                      riskPct
                                    }
                                    color={
                                      riskPct >=
                                      75
                                        ? "#ff3344"
                                        : riskPct >=
                                            40
                                          ? "#ffcc00"
                                          : "#00ff88"
                                    }
                                    showValue={
                                      false
                                    }
                                  />
                                </div>

                                <div>
                                  <div
                                    style={{
                                      fontSize: 9,
                                      color:
                                        "var(--text-muted)",
                                      letterSpacing:
                                        "0.12em",
                                      marginBottom: 4,
                                    }}
                                  >
                                    MODEL CONFIDENCE
                                  </div>

                                  <div
                                    style={{
                                      fontSize: 20,
                                      fontWeight: 700,
                                      color:
                                        "var(--accent)",
                                      marginBottom: 4,
                                    }}
                                  >
                                    {confPct}%
                                  </div>

                                  <ProgressBar
                                    value={
                                      confPct
                                    }
                                    color={
                                      "var(--accent)"
                                    }
                                    showValue={
                                      false
                                    }
                                  />
                                </div>

                                <div>
                                  <div
                                    style={{
                                      fontSize: 9,
                                      color:
                                        "var(--text-muted)",
                                      letterSpacing:
                                        "0.12em",
                                      marginBottom: 6,
                                    }}
                                  >
                                    TOP DRIVERS
                                  </div>

                                  {Object.entries(
                                    ml.feature_highlights
                                  ).map(
                                    ([
                                      feat,
                                      info,
                                    ]: [
                                      string,
                                      any
                                    ]) => {
                                      const maxImp =
                                        Math.max(
                                          ...Object.values(
                                            ml.feature_highlights
                                          ).map(
                                            (
                                              v: any
                                            ) =>
                                              v.importance
                                          )
                                        );

                                      return (
                                        <div
                                          key={feat}
                                          style={{
                                            marginBottom:
                                              7,
                                          }}
                                        >
                                          <div
                                            style={{
                                              display:
                                                "flex",
                                              justifyContent:
                                                "space-between",
                                              marginBottom:
                                                2,
                                            }}
                                          >
                                            <span
                                              style={{
                                                fontSize: 9,
                                                color:
                                                  "var(--text-muted)",
                                              }}
                                            >
                                              {feat
                                                .replace(
                                                  /_/g,
                                                  " "
                                                )
                                                .toUpperCase()}
                                            </span>

                                            <span
                                              style={{
                                                fontSize: 9,
                                                color:
                                                  "var(--accent)",
                                              }}
                                            >
                                              {
                                                info.value
                                              }
                                            </span>
                                          </div>

                                          <div
                                            style={{
                                              height: 2,
                                              background:
                                                "var(--border)",
                                              borderRadius:
                                                1,
                                            }}
                                          >
                                            <div
                                              style={{
                                                height:
                                                  "100%",
                                                width: `${Math.round(
                                                  (info.importance /
                                                    maxImp) *
                                                    100
                                                )}%`,
                                                background:
                                                  "var(--accent-dim)",
                                                transition:
                                                  "width 0.6s ease",
                                              }}
                                            />
                                          </div>
                                        </div>
                                      );
                                    }
                                  )}
                                </div>
                              </div>
                            </div>
                          </Panel>
                        );
                      })()}

                    {/* Temporal predictor */}
                    {selected.ml_forecast &&
                      (() => {
                        const fc =
                          selected.ml_forecast;

                        const isCritical =
                          fc.forecast_label ===
                          "CRITICAL_AHEAD";

                        const fcColor =
                          isCritical
                            ? "#ff3344"
                            : "#00ff88";

                        const critPct =
                          Math.round(
                            fc.critical_probability *
                              100
                          );

                        const confPct =
                          Math.round(
                            fc.forecast_confidence *
                              100
                          );

                        const kpTrend =
                          fc.delta_kp ?? 0;

                        const pwrTrend =
                          fc.delta_power ??
                          0;

                        const kpArrow =
                          kpTrend > 0.2
                            ? "▲"
                            : kpTrend < -0.2
                              ? "▼"
                              : "─";

                        const pwrArrow =
                          pwrTrend > 2
                            ? "▲"
                            : pwrTrend < -2
                              ? "▼"
                              : "─";

                        const kpArrowColor =
                          kpTrend > 0.2
                            ? "#ff3344"
                            : kpTrend < -0.2
                              ? "#00ff88"
                              : "var(--text-muted)";

                        const pwrArrowColor =
                          pwrTrend > 2
                            ? "#00ff88"
                            : pwrTrend < -2
                              ? "#ff3344"
                              : "var(--text-muted)";

                        return (
                          <Panel>
                            <Label>
                              30-MIN FORECAST —
                              TEMPORAL PREDICTOR
                            </Label>

                            <div
                              style={{
                                padding:
                                  "10px 14px 4px",
                                display:
                                  "flex",
                                alignItems:
                                  "center",
                                gap: 8,
                                borderBottom:
                                  "1px solid var(--border)",
                              }}
                            >
                              <span
                                style={{
                                  fontSize: 9,
                                  color:
                                    "var(--text-dim)",
                                  letterSpacing:
                                    "0.14em",
                                }}
                              >
                                ANSWERS: WHAT HAPPENS IN 30 MINUTES?

                                {fc.window_padded && (
                                  <span
                                    style={{
                                      color:
                                        "#ffcc00",
                                      marginLeft: 8,
                                    }}
                                  >
                                    ⚠  LIMITED HISTORY
                                  </span>
                                )}
                              </span>
                            </div>

                            <div
                              style={{
                                padding:
                                  "14px 18px",
                                display:
                                  "grid",
                                gridTemplateColumns:
                                  "1fr 1fr",
                                gap: 16,
                              }}
                            >
                              <div>
                                <div
                                  style={{
                                    fontSize: 9,
                                    color:
                                      "var(--text-muted)",
                                    letterSpacing:
                                      "0.15em",
                                    marginBottom: 6,
                                  }}
                                >
                                  FORECAST AT T+30 MIN
                                </div>

                                <div
                                  style={{
                                    fontSize: 20,
                                    fontWeight: 700,
                                    color:
                                      fcColor,
                                    textShadow: `0 0 12px ${fcColor}55`,
                                    marginBottom: 10,
                                  }}
                                >
                                  {fc.forecast_label.replace(
                                    /_/g,
                                    " "
                                  )}
                                </div>

                                <div
                                  style={{
                                    display:
                                      "flex",
                                    flexDirection:
                                      "column",
                                    gap: 5,
                                  }}
                                >
                                  {[
                                    {
                                      label:
                                        "CRITICAL AHEAD",
                                      val: critPct,
                                      color:
                                        "#ff3344",
                                    },
                                    {
                                      label:
                                        "NOMINAL  AHEAD",
                                      val:
                                        100 -
                                        critPct,
                                      color:
                                        "#00ff88",
                                    },
                                  ].map(
                                    (row) => (
                                      <div
                                        key={
                                          row.label
                                        }
                                      >
                                        <div
                                          style={{
                                            display:
                                              "flex",
                                            justifyContent:
                                              "space-between",
                                            marginBottom:
                                              2,
                                          }}
                                        >
                                          <span
                                            style={{
                                              fontSize: 9,
                                              color:
                                                "var(--text-muted)",
                                              letterSpacing:
                                                "0.08em",
                                            }}
                                          >
                                            {
                                              row.label
                                            }
                                          </span>

                                          <span
                                            style={{
                                              fontSize: 9,
                                              color:
                                                row.color,
                                              fontWeight:
                                                600,
                                            }}
                                          >
                                            {row.val}
                                            %
                                          </span>
                                        </div>

                                        <div
                                          style={{
                                            height: 2,
                                            background:
                                              "var(--border)",
                                            borderRadius:
                                              1,
                                          }}
                                        >
                                          <div
                                            style={{
                                              height:
                                                "100%",
                                              width: `${row.val}%`,
                                              background:
                                                row.color,
                                              transition:
                                                "width 0.6s ease",
                                            }}
                                          />
                                        </div>
                                      </div>
                                    )
                                  )}
                                </div>
                              </div>

                              <div
                                style={{
                                  display:
                                    "flex",
                                  flexDirection:
                                    "column",
                                  gap: 12,
                                }}
                              >
                                <div>
                                  <div
                                    style={{
                                      fontSize: 9,
                                      color:
                                        "var(--text-muted)",
                                      letterSpacing:
                                        "0.12em",
                                      marginBottom: 4,
                                    }}
                                  >
                                    FORECAST CONFIDENCE
                                  </div>

                                  <div
                                    style={{
                                      fontSize: 20,
                                      fontWeight: 700,
                                      color:
                                        "var(--accent)",
                                      marginBottom: 4,
                                    }}
                                  >
                                    {confPct}%
                                  </div>

                                  <ProgressBar
                                    value={
                                      confPct
                                    }
                                    color={
                                      "var(--accent)"
                                    }
                                    showValue={
                                      false
                                    }
                                  />
                                </div>

                                <div
                                  style={{
                                    borderTop:
                                      "1px solid var(--border)",
                                    paddingTop: 10,
                                  }}
                                >
                                  <div
                                    style={{
                                      fontSize: 9,
                                      color:
                                        "var(--text-muted)",
                                      letterSpacing:
                                        "0.12em",
                                      marginBottom: 8,
                                    }}
                                  >
                                    30-MIN TREND (WINDOW)
                                  </div>

                                  <div
                                    style={{
                                      display:
                                        "flex",
                                      flexDirection:
                                        "column",
                                      gap: 6,
                                    }}
                                  >
                                    <div
                                      style={{
                                        display:
                                          "flex",
                                        justifyContent:
                                          "space-between",
                                      }}
                                    >
                                      <span
                                        style={{
                                          fontSize: 10,
                                          color:
                                            "var(--text-muted)",
                                        }}
                                      >
                                        KP INDEX
                                      </span>

                                      <span
                                        style={{
                                          fontSize: 11,
                                          color:
                                            kpArrowColor,
                                          fontWeight:
                                            700,
                                        }}
                                      >
                                        {kpTrend >=
                                        0
                                          ? "+"
                                          : ""}
                                        {kpTrend.toFixed(
                                          2
                                        )}{" "}
                                        {kpArrow}
                                      </span>
                                    </div>

                                    <div
                                      style={{
                                        display:
                                          "flex",
                                        justifyContent:
                                          "space-between",
                                      }}
                                    >
                                      <span
                                        style={{
                                          fontSize: 10,
                                          color:
                                            "var(--text-muted)",
                                        }}
                                      >
                                        POWER OUTPUT
                                      </span>

                                      <span
                                        style={{
                                          fontSize: 11,
                                          color:
                                            pwrArrowColor,
                                          fontWeight:
                                            700,
                                        }}
                                      >
                                        {pwrTrend >=
                                        0
                                          ? "+"
                                          : ""}
                                        {pwrTrend.toFixed(
                                          1
                                        )}{" "}
                                        W{" "}
                                        {pwrArrow}
                                      </span>
                                    </div>
                                  </div>
                                </div>

                                <div
                                  style={{
                                    fontSize: 9,
                                    color:
                                      "var(--text-dim)",
                                    lineHeight: 1.5,
                                    marginTop: 4,
                                    ...sans,
                                  }}
                                >
                                  Supervised
                                  classification
                                  over last
                                  30-min
                                  window.
                                  Predicts
                                  HelioMesh
                                  simulation
                                  states — not
                                  real failures.
                                  Test-set
                                  recall on
                                  CRITICAL:
                                  98.2% vs
                                  84.5% baseline
                                  (+13.7 pts).
                                </div>
                              </div>
                            </div>
                          </Panel>
                        );
                      })()}
                  </div>
                )}

                {/* Model agreement + drift */}
                {(selected.model_agreement ||
                  selected.drift_status) && (
                  <div
                    style={{
                      display: "grid",
                      gridTemplateColumns:
                        "1fr 1fr",
                      gap: 10,
                      marginBottom: 12,
                    }}
                  >
                    {selected.model_agreement &&
                      (() => {
                        const agree =
                          selected.model_agreement ===
                          "AGREE";

                        const agreeColor =
                          agree
                            ? "#00ff88"
                            : "#ffcc00";

                        return (
                          <Panel>
                            <Label>
                              MODEL STATUS
                            </Label>

                            <div
                              style={{
                                padding:
                                  "12px 18px",
                                display:
                                  "flex",
                                alignItems:
                                  "center",
                                gap: 12,
                              }}
                            >
                              <div
                                style={{
                                  background:
                                    agree
                                      ? "rgba(0,255,136,0.12)"
                                      : "rgba(255,204,0,0.12)",
                                  border: `1px solid ${agreeColor}`,
                                  borderRadius:
                                    3,
                                  padding:
                                    "6px 14px",
                                }}
                              >
                                <span
                                  style={{
                                    color:
                                      agreeColor,
                                    fontSize: 13,
                                    fontWeight: 700,
                                    letterSpacing:
                                      "0.12em",
                                  }}
                                >
                                  {
                                    selected.model_agreement
                                  }
                                </span>
                              </div>

                              <span
                                style={{
                                  fontSize: 11,
                                  color:
                                    "var(--text-muted)",
                                  lineHeight: 1.5,
                                  ...sans,
                                }}
                              >
                                {agree
                                  ? "RF current-state and Temporal Predictor forecast agree on risk level."
                                  : "RF current-state and GB 30-min forecast diverge. Review both models."}
                              </span>
                            </div>
                          </Panel>
                        );
                      })()}

                    {selected.drift_status &&
                      (() => {
                        const stable =
                          selected.drift_status ===
                          "STABLE";

                        const moderate =
                          selected.drift_status ===
                          "MODERATE_DRIFT";

                        const driftColor =
                          stable
                            ? "#00ff88"
                            : moderate
                              ? "#ffcc00"
                              : "#ff3344";

                        return (
                          <Panel>
                            <Label>
                              DRIFT STATUS
                            </Label>

                            <div
                              style={{
                                padding:
                                  "12px 18px",
                                display:
                                  "flex",
                                alignItems:
                                  "center",
                                gap: 12,
                              }}
                            >
                              <div
                                style={{
                                  background: `${driftColor}18`,
                                  border: `1px solid ${driftColor}`,
                                  borderRadius:
                                    3,
                                  padding:
                                    "6px 14px",
                                }}
                              >
                                <span
                                  style={{
                                    color:
                                      driftColor,
                                    fontSize: 13,
                                    fontWeight: 700,
                                    letterSpacing:
                                      "0.12em",
                                  }}
                                >
                                  {selected.drift_status.replace(
                                    /_/g,
                                    " "
                                  )}
                                </span>
                              </div>

                              <span
                                style={{
                                  fontSize: 11,
                                  color:
                                    "var(--text-muted)",
                                  lineHeight: 1.5,
                                  ...sans,
                                }}
                              >
                                {stable
                                  ? "Telemetry within training distribution."
                                  : moderate
                                    ? "Moderate parameter shift from training range."
                                    : "High deviation from training distribution — treat predictions with caution."}
                              </span>
                            </div>
                          </Panel>
                        );
                      })()}
                  </div>
                )}

                {/* Confidence + risk */}
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns:
                      "1fr 1fr",
                    gap: 10,
                    marginBottom: 12,
                  }}
                >
                  <Panel>
                    <Label>
                      CONFIDENCE SCORE
                    </Label>

                    <div
                      style={{
                        padding:
                          "14px 18px",
                      }}
                    >
                      <div
                        style={{
                          display:
                            "flex",
                          justifyContent:
                            "space-between",
                          alignItems:
                            "baseline",
                          marginBottom: 10,
                        }}
                      >
                        <span
                          style={{
                            fontSize: 28,
                            fontWeight: 700,
                            color:
                              selected.confidence_score >=
                              70
                                ? "#00ff88"
                                : selected.confidence_score >=
                                    40
                                  ? "#ffcc00"
                                  : "#ff3344",
                            textShadow: `0 0 16px ${
                              selected.confidence_score >=
                              70
                                ? "#00ff88"
                                : selected.confidence_score >=
                                    40
                                  ? "#ffcc00"
                                  : "#ff3344"
                            }55`,
                          }}
                        >
                          {
                            selected.confidence_score
                          }
                        </span>

                        <span
                          style={{
                            fontSize: 11,
                            color:
                              "var(--text-dim)",
                          }}
                        >
                          /100
                        </span>
                      </div>

                      <ProgressBar
                        value={
                          selected.confidence_score
                        }
                        color={
                          selected.confidence_score >=
                          70
                            ? "#00ff88"
                            : selected.confidence_score >=
                                40
                              ? "#ffcc00"
                              : "#ff3344"
                        }
                        showValue={false}
                      />

                      <div
                        style={{
                          marginTop: 8,
                          display:
                            "flex",
                          gap: 16,
                        }}
                      >
                        <span
                          style={{
                            fontSize: 10,
                            color:
                              "var(--text-muted)",
                          }}
                        >
                          TIER:{" "}
                          <span
                            style={{
                              color:
                                "var(--text-primary)",
                            }}
                          >
                            {
                              selected.confidence_tier
                            }
                          </span>
                        </span>

                        <span
                          style={{
                            fontSize: 10,
                            color:
                              "var(--text-muted)",
                          }}
                        >
                          MODE:{" "}
                          <span
                            style={{
                              color:
                                "var(--accent)",
                            }}
                          >
                            {
                              selected.mission_mode
                            }
                          </span>
                        </span>
                      </div>
                    </div>
                  </Panel>

                  <Panel>
                    <Label>
                      MISSION RISK BREAKDOWN
                    </Label>

                    <RiskBreakdown
                      breakdown={
                        selected.risk_breakdown
                      }
                    />
                  </Panel>
                </div>

                {/* Granite + reasons/commands */}
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns:
                      "1fr 1fr",
                    gap: 10,
                    marginBottom: 12,
                  }}
                >
                  <Panel>
                    <Label>
                      MISSION DECISION TRACE — IBM GRANITE
                    </Label>

                    <div
                      style={{
                        padding:
                          "12px 14px",
                        maxHeight: 300,
                        overflowY:
                          "auto",
                      }}
                    >
                      <GraniteTrace
                        text={
                          selected.ai_trace
                        }
                      />
                    </div>
                  </Panel>

                  <div
                    style={{
                      display: "flex",
                      flexDirection:
                        "column",
                      gap: 10,
                    }}
                  >
                    <Panel
                      style={{
                        flex: 1,
                      }}
                    >
                      <Label>
                        DECISION REASONS
                      </Label>

                      <div
                        style={{
                          padding:
                            "10px 14px",
                        }}
                      >
                        {selected.reasons
                          ?.map(
                            (
                              r: string,
                              i: number
                            ) => (
                              <div
                                key={i}
                                style={{
                                  display:
                                    "flex",
                                  gap: 8,
                                  marginBottom:
                                    7,
                                  fontSize: 12,
                                }}
                              >
                                <span
                                  style={{
                                    color:
                                      "var(--accent)",
                                    flexShrink:
                                      0,
                                  }}
                                >
                                ►
                                </span>

                                <span
                                  style={{
                                    color:
                                      "var(--text-primary)",
                                    ...sans,
                                  }}
                                >
                                  {r}
                                </span>
                              </div>
                            )
                          )}
                      </div>
                    </Panel>

                    <Panel
                      style={{
                        flex: 1,
                      }}
                    >
                      <Label>
                        DECISION ENGINE COMMANDS
                      </Label>

                      <div
                        style={{
                          padding:
                            "10px 14px",
                        }}
                      >
                        {Array.isArray(
                          selected.subsystem_commands
                        ) &&
                        selected
                          .subsystem_commands
                          .length > 0 ? (
                          <>
                            <div
                              style={{
                                fontSize: 9,
                                color:
                                  "var(--text-dim)",
                                letterSpacing:
                                  "0.12em",
                                marginBottom:
                                  9,
                                ...mono,
                              }}
                            >
                              GENERATED BY DETERMINISTIC POLICY ENGINE
                            </div>

                            {Array.isArray(selected.subsystem_commands) &&
  [...new Set(selected.subsystem_commands)].map((cmd, i) => (
    <div
      key={`${String(cmd)}-${i}`}
      style={{
        display: "flex",
        gap: 8,
        marginBottom: 7,
        fontSize: 12,
      }}
    >
      <span
        style={{
          color: "#00ff88",
          flexShrink: 0,
        }}
      >
        →
      </span>

      <span
        style={{
          color: "#00ff88",
          ...sans,
        }}
      >
        {String(cmd)}
      </span>
    </div>
  ))}
                          </>
                        ) : (
                          <div
                            style={{
                              color:
                                "var(--text-dim)",
                              fontSize: 11,
                              ...sans,
                            }}
                          >
                            No subsystem
                            commands generated.
                          </div>
                        )}
                      </div>
                    </Panel>
                  </div>
                </div>

                {/* Human approval */}
                {selected.status ===
                  "pending_approval" && (
                  <div
                    style={{
                      display: "grid",
                      gridTemplateColumns:
                        "1fr 1fr",
                      gap: 10,
                    }}
                  >
                    <button
                      onClick={() =>
                        approveDecision(
                          selected.decision_id
                        )
                      }
                      style={{
                        background:
                          "rgba(0,255,136,0.08)",
                        border:
                          "1px solid #00ff88",
                        color: "#00ff88",
                        padding: "13px",
                        fontSize: 11,
                        letterSpacing:
                          "0.2em",
                        cursor: "pointer",
                        borderRadius: 2,
                        ...mono,
                      }}
                    >
                      PENDING OPERATOR REVIEW
                    </button>

                    <button
                      onClick={() =>
                        rejectDecision(
                          selected.decision_id
                        )
                      }
                      style={{
                        background:
                          "rgba(255,51,68,0.08)",
                        border:
                          "1px solid #ff3344",
                        color: "#ff3344",
                        padding: "13px",
                        fontSize: 11,
                        letterSpacing:
                          "0.2em",
                        cursor: "pointer",
                        borderRadius: 2,
                        ...mono,
                      }}
                    >
                      âœ• REJECT DECISION
                    </button>
                  </div>
                )}
              </div>
            );
          })()}

        {/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            AI CONSOLE
        â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */}
        {view === "chat" && (
          <div
            style={{
              maxWidth: 820,
            }}
          >
            <Panel
              style={{
                marginBottom: 10,
              }}
            >
              <Label>
                AI CONSOLE — IBM GRANITE 4
              </Label>

              <div
                style={{
                  padding:
                    "14px 16px",
                  minHeight: 280,
                  maxHeight: 420,
                  overflowY: "auto",
                }}
              >
                {chatHistory.length ===
                  0 &&
                !loading ? (
                  <div
                    style={{
                      color:
                        "var(--text-dim)",
                      fontSize: 11,
                      letterSpacing:
                        "0.15em",
                      textAlign:
                        "center",
                      marginTop: 70,
                      ...mono,
                    }}
                  >
                    AWAITING OPERATOR INPUT...
                  </div>
                ) : (
                  chatHistory.map(
                    (h, i) => (
                      <div
                        key={i}
                        style={{
                          marginBottom: 18,
                        }}
                      >
                        <div
                          style={{
                            display:
                              "flex",
                            gap: 10,
                            marginBottom: 6,
                          }}
                        >
                          <span
                            style={{
                              color:
                                "var(--accent)",
                              fontSize: 10,
                              letterSpacing:
                                "0.1em",
                              flexShrink:
                                0,
                              marginTop: 2,
                              ...mono,
                            }}
                          >
                            OPERATOR â–¶
                          </span>

                          <span
                            style={{
                              color:
                                "var(--text-primary)",
                              fontSize: 13,
                              ...sans,
                            }}
                          >
                            {h.q}
                          </span>
                        </div>

                        <div
                          style={{
                            display:
                              "flex",
                            gap: 10,
                          }}
                        >
                          <span
                            style={{
                              color:
                                "#00ff88",
                              fontSize: 10,
                              letterSpacing:
                                "0.1em",
                              flexShrink:
                                0,
                              marginTop: 2,
                              ...mono,
                            }}
                          >
                            GRANITE â–¶
                          </span>

                          <span
                            style={{
                              color:
                                "#c8ffd8",
                              fontSize: 13,
                              lineHeight:
                                1.75,
                              whiteSpace:
                                "pre-wrap",
                              ...sans,
                            }}
                          >
                            {h.a}
                          </span>
                        </div>
                      </div>
                    )
                  )
                )}

                {loading && (
                  <div
                    style={{
                      color: "#ffcc00",
                      fontSize: 11,
                      letterSpacing:
                        "0.12em",
                      ...mono,
                    }}
                  >
                    PROCESSING QUERY...
                  </div>
                )}

                <div
                  ref={chatEndRef}
                />
              </div>
            </Panel>

            <Panel>
              <Label>INPUT</Label>

              <div
                style={{
                  padding:
                    "10px 14px",
                  display: "flex",
                  gap: 8,
                }}
              >
                <textarea
                  value={chat}
                  onChange={(e) =>
                    setChat(
                      e.target.value
                    )
                  }
                  onKeyDown={(e) => {
                    if (
                      e.key ===
                        "Enter" &&
                      !e.shiftKey
                    ) {
                      e.preventDefault();
                      sendChat();
                    }
                  }}
                  placeholder="Enter query... (Enter to send)"
                  style={{
                    flex: 1,
                    background:
                      "var(--bg-primary)",
                    border:
                      "1px solid var(--border-bright)",
                    color:
                      "var(--text-primary)",
                    padding:
                      "9px 12px",
                    fontSize: 12,
                    resize: "none",
                    height: 68,
                    outline: "none",
                    borderRadius: 2,
                    ...mono,
                  }}
                />

                <button
                  onClick={sendChat}
                  disabled={loading}
                  style={{
                    background:
                      "rgba(0,170,255,0.08)",
                    border:
                      "1px solid var(--accent)",
                    color:
                      "var(--accent)",
                    padding:
                      "0 22px",
                    fontSize: 10,
                    letterSpacing:
                      "0.15em",
                    cursor: loading
                      ? "not-allowed"
                      : "pointer",
                    opacity: loading
                      ? 0.4
                      : 1,
                    borderRadius: 2,
                    ...mono,
                  }}
                >
                  SEND
                </button>
              </div>
            </Panel>
          </div>
        )}

        {/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            MISSION REPORT
        â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */}
        {view === "report" && report && (
          <div>
            <div
              style={{
                display: "grid",
                gridTemplateColumns:
                  "repeat(4,1fr)",
                gap: 10,
                marginBottom: 14,
              }}
            >
              {[
                {
                  label: "TOTAL DECISIONS",
                  value:
                    report.total_decisions,
                  color:
                    "var(--accent)",
                },
                {
                  label: 'AUTO ROUTE SELECTED',
                  value:
                    report.auto_executed,
                  color: "#00ff88",
                },
                {
                  label: "PENDING APPROVAL",
                  value:
                    report.pending_approval,
                  color: "#ffcc00",
                },
                {
                  label: "ESCALATED",
                  value: report.escalated,
                  color: "#ff3344",
                },
              ].map((s) => (
                <Panel key={s.label}>
                  <Label>{s.label}</Label>

                  <div
                    style={{
                      padding:
                        "14px 18px",
                    }}
                  >
                    <div
                      style={{
                        fontSize: 38,
                        fontWeight: 700,
                        color: s.color,
                        textShadow: `0 0 18px ${s.color}44`,
                        lineHeight: 1,
                      }}
                    >
                      {s.value}
                    </div>
                  </div>
                </Panel>
              ))}
            </div>

            <div
              style={{
                display: "grid",
                gridTemplateColumns:
                  "1fr 340px",
                gap: 14,
              }}
            >
              <Panel>
                <Label>
                  RECENT DECISIONS LOG
                </Label>

                {report.decisions
                  ?.length === 0 && (
                  <div
                    style={{
                      padding: "40px",
                      textAlign:
                        "center",
                      color:
                        "var(--text-dim)",
                      fontSize: 11,
                    }}
                  >
                    NO RECORDS
                  </div>
                )}

                {report.decisions?.map(
                  (
                    d: any,
                    i: number
                  ) => {
                    const sm =
                      STATUS[
                        d.status
                      ] ??
                      STATUS.rejected;

                    return (
                      <div
                        key={
                          d.decision_id
                        }
                        style={{
                          padding:
                            "12px 14px",
                          borderBottom:
                            i <
                            report
                              .decisions
                              .length -
                              1
                              ? "1px solid var(--border)"
                              : "none",
                        }}
                      >
                        <div
                          style={{
                            display:
                              "flex",
                            justifyContent:
                              "space-between",
                            alignItems:
                              "center",
                            marginBottom: 6,
                          }}
                        >
                          <span
                            style={{
                              color:
                                "var(--accent)",
                              fontWeight: 600,
                              fontSize: 12,
                            }}
                          >
                            {d.decision_id}
                          </span>

                          <span
                            style={{
                              display:
                                "flex",
                              alignItems:
                                "center",
                            }}
                          >
                            <Dot
                              color={
                                sm.dot
                              }
                            />

                            <span
                              style={{
                                color:
                                  sm.color,
                                fontSize: 10,
                                letterSpacing:
                                  "0.08em",
                              }}
                            >
                              {
                                sm.label
                              }
                            </span>
                          </span>
                        </div>

                        <p
                          style={{
                            fontSize: 11,
                            color:
                              "var(--text-muted)",
                            lineHeight: 1.6,
                            ...sans,
                          }}
                        >
                          {
                            d.ai_trace_summary
                          }
                        </p>
                      </div>
                    );
                  }
                )}
              </Panel>

              <Panel>
                <Label>
                  MISSION PIPELINE
                </Label>

                <ArchDiagram />
              </Panel>
            </div>
          </div>
        )}

        {/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            VALIDATION
        â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */}
        {view === "validation" && (
          <div>
            <div
              style={{
                display: "flex",
                alignItems:
                  "center",
                justifyContent:
                  "space-between",
                marginBottom: 14,
              }}
            >
              <span
                style={{
                  color:
                    "var(--accent)",
                  fontSize: 11,
                  letterSpacing:
                    "0.15em",
                  ...mono,
                }}
              >
                VALIDATION RESULTS
              </span>

              <button
                onClick={
                  fetchValidation
                }
                style={{
                  background:
                    "rgba(0,170,255,0.08)",
                  border:
                    "1px solid var(--accent)",
                  color:
                    "var(--accent)",
                  padding: "6px 14px",
                  fontSize: 10,
                  letterSpacing:
                    "0.14em",
                  cursor: "pointer",
                  borderRadius: 2,
                  ...mono,
                }}
              >
                RELOAD
              </button>
            </div>

            <Panel
              style={{
                marginBottom: 12,
              }}
            >
              <Label>
                SIMULATION BENCHMARK — FROZEN (do not retrain)
              </Label>

              <div
                style={{
                  padding:
                    "14px 18px",
                }}
              >
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns:
                      "repeat(4,1fr)",
                    gap: 12,
                    marginBottom: 12,
                  }}
                >
                  {[
                    {
                      label:
                        "TEMPORAL PREDICTOR MACRO F1",
                      value: "0.9708",
                      color:
                        "#00ff88",
                    },
                    {
                      label:
                        "CRITICAL RECALL",
                      value: "98.2%",
                      color:
                        "#00ff88",
                    },
                    {
                      label:
                        "LAST-KNOWN-STATE F1",
                      value: "0.9015",
                      color:
                        "#ffcc00",
                    },
                    {
                      label:
                        "KP-ONLY BASELINE F1",
                      value: "0.8977",
                      color:
                        "#ffcc00",
                    },
                  ].map((m) => (
                    <div
                      key={m.label}
                      style={{
                        textAlign:
                          "center",
                      }}
                    >
                      <div
                        style={{
                          fontSize: 9,
                          color:
                            "var(--text-muted)",
                          letterSpacing:
                            "0.12em",
                          marginBottom: 4,
                        }}
                      >
                        {m.label}
                      </div>

                      <div
                        style={{
                          fontSize: 22,
                          fontWeight: 700,
                          color: m.color,
                        }}
                      >
                        {m.value}
                      </div>
                    </div>
                  ))}
                </div>

                <div
                  style={{
                    fontSize: 10,
                    color:
                      "var(--text-dim)",
                    ...sans,
                  }}
                >
                  Evaluated on 1,800 held-out simulation test sequences (chronological 70/15/15 split).
                  Temporal predictor is a supervised GB classifier over 6-step, 30-min look-back windows.
                  Not a predictor of real spacecraft failures.
                </div>
              </div>
            </Panel>

            {!validation ? (
              <div
                style={{
                  textAlign:
                    "center",
                  padding: 40,
                  color:
                    "var(--text-dim)",
                  fontSize: 11,
                  letterSpacing:
                    "0.15em",
                }}
              >
                VALIDATION DATA NOT LOADED — CLICK RELOAD TO FETCH
              </div>
            ) : (
              <div
                style={{
                  display: "flex",
                  flexDirection:
                    "column",
                  gap: 12,
                }}
              >
                {(validation.snapshot_validation ||
                  validation.temporal_validation) && (
                  <Panel>
                    <Label>
                      REAL-DATA VALIDATION (OMNI2 INTERNAL CONSISTENCY CHECK)
                    </Label>

                    <div
                      style={{
                        padding:
                          "10px 18px 14px",
                      }}
                    >
                      <div
                        style={{
                          fontSize: 10,
                          color:
                            "var(--text-dim)",
                          marginBottom: 10,
                          ...sans,
                        }}
                      >
                        Consistency check: do model predictions match HelioMesh labeling rules applied to space-weather inputs?
                        Source:{" "}
                        {validation.snapshot_validation?.data_source ??
                          "—"}{" "}
                        | NOT spacecraft failure prediction.
                      </div>

                      <div
                        style={{
                          display: "grid",
                          gridTemplateColumns:
                            "1fr 1fr",
                          gap: 14,
                        }}
                      >
                        {validation.snapshot_validation && (
                          <div>
                            <div
                              style={{
                                fontSize: 9,
                                color:
                                  "var(--accent)",
                                letterSpacing:
                                  "0.14em",
                                marginBottom:
                                  8,
                              }}
                            >
                              SNAPSHOT RF (CURRENT STATE)
                            </div>

                            <div
                              style={{
                                fontSize: 9,
                                color:
                                  "var(--text-muted)",
                                marginBottom:
                                  4,
                              }}
                            >
                              Records:{" "}
                              {
                                validation
                                  .snapshot_validation
                                  .n_records
                              }
                            </div>

                            <div
                              style={{
                                display:
                                  "flex",
                                gap: 18,
                              }}
                            >
                              <div>
                                <div
                                  style={{
                                    fontSize: 9,
                                    color:
                                      "var(--text-muted)",
                                  }}
                                >
                                  CONSISTENCY
                                </div>

                                <div
                                  style={{
                                    fontSize: 18,
                                    fontWeight: 700,
                                    color:
                                      "#00ff88",
                                  }}
                                >
                                  {(
                                    validation
                                      .snapshot_validation
                                      .consistency_rate *
                                    100
                                  ).toFixed(
                                    1
                                  )}
                                  %
                                </div>
                              </div>

                              <div>
                                <div
                                  style={{
                                    fontSize: 9,
                                    color:
                                      "var(--text-muted)",
                                  }}
                                >
                                  MACRO F1
                                </div>

                                <div
                                  style={{
                                    fontSize: 18,
                                    fontWeight: 700,
                                    color:
                                      "#00ff88",
                                  }}
                                >
                                  {
                                    validation
                                      .snapshot_validation
                                      .macro_f1
                                  }
                                </div>
                              </div>
                            </div>
                          </div>
                        )}

                        {validation.temporal_validation && (
                          <div>
                            <div
                              style={{
                                fontSize: 9,
                                color:
                                  "var(--accent)",
                                letterSpacing:
                                  "0.14em",
                                marginBottom:
                                  8,
                              }}
                            >
                              TEMPORAL GB (30-MIN AHEAD)
                            </div>

                            <div
                              style={{
                                fontSize: 9,
                                color:
                                  "var(--text-muted)",
                                marginBottom:
                                  4,
                              }}
                            >
                              Sequences:{" "}
                              {
                                validation
                                  .temporal_validation
                                  .n_sequences
                              }{" "}
                              | Note:{" "}
                              {
                                validation
                                  .temporal_validation
                                  .step_duration_note
                              }
                            </div>

                            <div
                              style={{
                                display:
                                  "flex",
                                gap: 18,
                              }}
                            >
                              <div>
                                <div
                                  style={{
                                    fontSize: 9,
                                    color:
                                      "var(--text-muted)",
                                  }}
                                >
                                  CONSISTENCY
                                </div>

                                <div
                                  style={{
                                    fontSize: 18,
                                    fontWeight: 700,
                                    color:
                                      "#ffcc00",
                                  }}
                                >
                                  {(
                                    validation
                                      .temporal_validation
                                      .consistency_rate *
                                    100
                                  ).toFixed(
                                    1
                                  )}
                                  %
                                </div>
                              </div>

                              <div>
                                <div
                                  style={{
                                    fontSize: 9,
                                    color:
                                      "var(--text-muted)",
                                  }}
                                >
                                  MACRO F1
                                </div>

                                <div
                                  style={{
                                    fontSize: 18,
                                    fontWeight: 700,
                                    color:
                                      "#ffcc00",
                                  }}
                                >
                                  {
                                    validation
                                      .temporal_validation
                                      .macro_f1
                                  }
                                </div>
                              </div>
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  </Panel>
                )}

                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns:
                      "1fr 1fr",
                    gap: 12,
                  }}
                >
                  {validation.early_warning && (
                    <Panel>
                      <Label>
                        EARLY WARNING EVALUATION
                      </Label>

                      <div
                        style={{
                          padding:
                            "10px 18px 14px",
                        }}
                      >
                        {[
                          {
                            label:
                              "CRITICAL TRANSITIONS",
                            value:
                              validation
                                .early_warning
                                .total_critical_transitions,
                          },
                          {
                            label:
                              "DETECTED EARLY",
                            value: `${validation.early_warning.transitions_detected_early} (${(
                              validation.early_warning
                                .early_detection_rate *
                              100
                            ).toFixed(1)}%)`,
                          },
                          {
                            label: "MISSED",
                            value:
                              validation
                                .early_warning
                                .missed_transitions,
                          },
                          {
                            label:
                              "FALSE EARLY WARNINGS",
                            value:
                              validation
                                .early_warning
                                .false_early_warnings,
                          },
                          {
                            label:
                              "MEDIAN LEAD TIME",
                            value: `${validation.early_warning.median_lead_time_steps} steps (${validation.early_warning.median_lead_time_steps * 5} min)`,
                          },
                        ].map((r) => (
                          <div
                            key={r.label}
                            style={{
                              display:
                                "flex",
                              justifyContent:
                                "space-between",
                              marginBottom:
                                7,
                            }}
                          >
                            <span
                              style={{
                                fontSize: 10,
                                color:
                                  "var(--text-muted)",
                              }}
                            >
                              {r.label}
                            </span>

                            <span
                              style={{
                                fontSize: 10,
                                color:
                                  "var(--text-primary)",
                                fontWeight: 600,
                              }}
                            >
                              {r.value}
                            </span>
                          </div>
                        ))}
                      </div>
                    </Panel>
                  )}

                  {validation.model_agreement && (
                    <Panel>
                      <Label>
                        MODEL AGREEMENT ANALYSIS
                      </Label>

                      <div
                        style={{
                          padding:
                            "10px 18px 14px",
                        }}
                      >
                        {[
                          {
                            label:
                              "SEQUENCES",
                            value:
                              validation
                                .model_agreement
                                .n_sequences,
                          },
                          {
                            label:
                              "AGREEMENT RATE",
                            value: `${(
                              validation
                                .model_agreement
                                .agreement_rate *
                              100
                            ).toFixed(1)}%`,
                          },
                          {
                            label:
                              "DISAGREEMENT RATE",
                            value: `${(
                              validation
                                .model_agreement
                                .disagreement_rate *
                              100
                            ).toFixed(1)}%`,
                          },
                          {
                            label:
                              "RF-NOM / GB-CRIT",
                            value:
                              validation
                                .model_agreement
                                .rf_nominal_gb_critical,
                          },
                          {
                            label:
                              "RF-NON-NOM / GB-NOM",
                            value:
                              validation
                                .model_agreement
                                .rf_non_nominal_gb_nominal,
                          },
                        ].map((r) => (
                          <div
                            key={r.label}
                            style={{
                              display:
                                "flex",
                              justifyContent:
                                "space-between",
                              marginBottom:
                                7,
                            }}
                          >
                            <span
                              style={{
                                fontSize: 10,
                                color:
                                  "var(--text-muted)",
                              }}
                            >
                              {r.label}
                            </span>

                            <span
                              style={{
                                fontSize: 10,
                                color:
                                  "var(--text-primary)",
                                fontWeight: 600,
                              }}
                            >
                              {r.value}
                            </span>
                          </div>
                        ))}
                      </div>
                    </Panel>
                  )}
                </div>

                {validation.policy_tests && (
                  <Panel>
                    <Label>
                      POLICY TEST SUITE
                    </Label>

                    <div
                      style={{
                        padding:
                          "10px 18px 14px",
                      }}
                    >
                      <div
                        style={{
                          display: "flex",
                          gap: 24,
                          marginBottom: 12,
                        }}
                      >
                        <div
                          style={{
                            textAlign:
                              "center",
                          }}
                        >
                          <div
                            style={{
                              fontSize: 9,
                              color:
                                "var(--text-muted)",
                            }}
                          >
                            PASSED
                          </div>

                          <div
                            style={{
                              fontSize: 22,
                              fontWeight: 700,
                              color:
                                "#00ff88",
                            }}
                          >
                            {
                              validation
                                .policy_tests
                                .passed
                            }
                          </div>
                        </div>

                        <div
                          style={{
                            textAlign:
                              "center",
                          }}
                        >
                          <div
                            style={{
                              fontSize: 9,
                              color:
                                "var(--text-muted)",
                            }}
                          >
                            FAILED
                          </div>

                          <div
                            style={{
                              fontSize: 22,
                              fontWeight: 700,
                              color:
                                validation
                                  .policy_tests
                                  .failed >
                                0
                                  ? "#ff3344"
                                  : "#00ff88",
                            }}
                          >
                            {
                              validation
                                .policy_tests
                                .failed
                            }
                          </div>
                        </div>

                        <div
                          style={{
                            textAlign:
                              "center",
                          }}
                        >
                          <div
                            style={{
                              fontSize: 9,
                              color:
                                "var(--text-muted)",
                            }}
                          >
                            CONSISTENCY
                          </div>

                          <div
                            style={{
                              fontSize: 22,
                              fontWeight: 700,
                              color:
                                "#00ff88",
                            }}
                          >
                            {
                              validation
                                .policy_tests
                                .consistency_pct
                            }
                            %
                          </div>
                        </div>
                      </div>

                      <div
                        style={{
                          fontSize: 10,
                          color:
                            "var(--text-dim)",
                          ...sans,
                        }}
                      >
                        {
                          validation
                            .policy_tests
                            .total_scenarios
                        }{" "}
                        deterministic routing
                        scenarios verified against
                        compute_confidence thresholds.
                      </div>
                    </div>
                  </Panel>
                )}

                <Panel>
                  <Label>
                    LIMITATIONS &amp; DISCLAIMERS
                  </Label>

                  <div
                    style={{
                      padding:
                        "12px 18px",
                      fontSize: 11,
                      color:
                        "var(--text-muted)",
                      lineHeight: 1.75,
                      ...sans,
                    }}
                  >
                    <div>
                      â€¢ All training data is
                      simulation-generated.
                      Models learn HelioMesh
                      prototype operational
                      safety rules, not real
                      spacecraft engineering
                      thresholds.
                    </div>

                    <div>
                      â€¢ OMNI2 validation uses
                      static sample data
                      (STATIC_SAMPLE). Live NASA
                      download unavailable in
                      current environment.
                    </div>

                    <div>
                      â€¢ Temporal predictor trained
                      on 5-min step sequences;
                      OMNI2 validation uses hourly
                      steps — results are
                      indicative only.
                    </div>

                    <div>
                      â€¢ Early warning lead times
                      are proxies derived from
                      delta_kp in the window, not
                      actual step-level transition
                      tracking.
                    </div>

                    <div>
                      â€¢ Model does NOT predict real
                      satellite failures. All
                      labels are HelioMesh
                      simulation safety states.
                    </div>
                  </div>
                </Panel>
              </div>
            )}
          </div>
        )}

        {/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            OPS-SAT REAL SPACECRAFT
        â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */}
        {view === "opssat" && (
          <div>
            <Panel
              style={{
                marginBottom: 12,
                borderColor: "#00ff8844",
              }}
            >
              <Label>
                REAL OPS-SAT SPACECRAFT EVIDENCE — ESA OPS-SAT MISSION
              </Label>

              <div
                style={{
                  padding: "12px 18px",
                }}
              >
                <div
                  style={{
                    display: "flex",
                    gap: 20,
                    alignItems:
                      "flex-start",
                    flexWrap: "wrap",
                  }}
                >
                  <div
                    style={{
                      flex: 1,
                      minWidth: 220,
                    }}
                  >
                    <div
                      style={{
                        fontSize: 10,
                        color: "#00ff88",
                        letterSpacing:
                          "0.12em",
                        marginBottom: 6,
                      }}
                    >
                      SOURCE
                    </div>

                    <div
                      style={{
                        fontSize: 12,
                        color:
                          "var(--text-primary)",
                        ...mono,
                      }}
                    >
                      OPS-SAT-AD | Zenodo 12588359 | MIT License
                    </div>

                    <div
                      style={{
                        fontSize: 10,
                        color:
                          "var(--text-dim)",
                        marginTop: 4,
                      }}
                    >
                      Ruszczak et al.,
                      Scientific Data (2025)
                    </div>

                    <div
                      style={{
                        fontSize: 10,
                        color:
                          "var(--text-dim)",
                        marginTop: 2,
                      }}
                    >
                      303,493 real telemetry
                      samples · 9 channels ·
                      2,123 segments
                    </div>
                  </div>

                  <div
                    style={{
                      flex: 2,
                      minWidth: 320,
                      background:
                        "rgba(255,80,50,0.06)",
                      border:
                        "1px solid rgba(255,80,50,0.3)",
                      padding:
                        "8px 12px",
                      borderRadius: 2,
                    }}
                  >
                    <div
                      style={{
                        fontSize: 9,
                        color: "#ff8866",
                        letterSpacing:
                          "0.15em",
                        marginBottom: 4,
                      }}
                    >
                      SEMANTIC BOUNDARY
                    </div>

                    <div
                      style={{
                        fontSize: 10,
                        color:
                          "var(--text-muted)",
                        lineHeight: 1.7,
                        ...sans,
                      }}
                    >
                      OPS-SAT-AD uses{" "}
                      <strong>
                        BINARY
                      </strong>{" "}
                      anomaly labels
                      (0=normal, 1=anomaly).
                      The HelioMesh four-class
                      taxonomy
                      (NOMINAL/STANDBY/SAFE_MODE/CRITICAL_AHEAD)
                      is{" "}
                      <strong>
                        NOT
                      </strong>{" "}
                      derived from or validated
                      by these labels.
                    </div>
                  </div>
                </div>
              </div>
            </Panel>

            {!opssatEvidence ? (
              <div
                style={{
                  textAlign:
                    "center",
                  padding: 40,
                  color:
                    "var(--text-dim)",
                  fontSize: 11,
                  letterSpacing:
                    "0.15em",
                }}
              >
                LOADING OPS-SAT EVIDENCE...
                &nbsp;

                <button
                  onClick={
                    fetchOpssatEvidence
                  }
                  style={{
                    background:
                      "rgba(0,170,255,0.08)",
                    border:
                      "1px solid var(--accent)",
                    color:
                      "var(--accent)",
                    padding: "4px 12px",
                    fontSize: 10,
                    cursor: "pointer",
                    borderRadius: 2,
                    ...mono,
                  }}
                >
                  RELOAD
                </button>
              </div>
            ) : (
              <div
                style={{
                  display: "flex",
                  flexDirection:
                    "column",
                  gap: 12,
                }}
              >
                <Panel>
                  <Label>
                    ANOMALY DETECTION — OFFICIAL TEST PARTITION (529 segments, 113 anomalous)
                  </Label>

                  <div
                    style={{
                      padding:
                        "14px 18px",
                    }}
                  >
                    <div
                      style={{
                        display: "grid",
                        gridTemplateColumns:
                          "repeat(4,1fr)",
                        gap: 12,
                        marginBottom: 12,
                      }}
                    >
                      {[
                        {
                          label:
                            "F1 SCORE",
                          value:
                            opssatEvidence.anomaly_detection?.f1?.toFixed(
                              4
                            ) ?? "—",
                          color:
                            "#00ff88",
                        },
                        {
                          label:
                            "ROC-AUC",
                          value:
                            opssatEvidence.anomaly_detection?.roc_auc?.toFixed(
                              4
                            ) ?? "—",
                          color:
                            "#00ff88",
                        },
                        {
                          label:
                            "PR-AUC",
                          value:
                            opssatEvidence.anomaly_detection?.pr_auc?.toFixed(
                              4
                            ) ?? "—",
                          color:
                            "#00ff88",
                        },
                        {
                          label:
                            "MCC",
                          value:
                            opssatEvidence.anomaly_detection?.mcc?.toFixed(
                              4
                            ) ?? "—",
                          color:
                            "#00ff88",
                        },
                      ].map((m) => (
                        <div
                          key={m.label}
                          style={{
                            textAlign:
                              "center",
                            padding:
                              "8px 0",
                          }}
                        >
                          <div
                            style={{
                              fontSize: 9,
                              color:
                                "var(--text-muted)",
                              letterSpacing:
                                "0.12em",
                              marginBottom: 4,
                            }}
                          >
                            {m.label}
                          </div>

                          <div
                            style={{
                              fontSize: 24,
                              fontWeight: 700,
                              color: m.color,
                              ...mono,
                            }}
                          >
                            {m.value}
                          </div>
                        </div>
                      ))}
                    </div>

                    <div
                      style={{
                        display: "grid",
                        gridTemplateColumns:
                          "repeat(4,1fr)",
                        gap: 10,
                      }}
                    >
                      {[
                        {
                          label:
                            "RECALL",
                          value:
                            opssatEvidence.anomaly_detection?.recall?.toFixed(
                              4
                            ) ?? "—",
                        },
                        {
                          label:
                            "PRECISION",
                          value:
                            opssatEvidence.anomaly_detection?.precision?.toFixed(
                              4
                            ) ?? "—",
                        },
                        {
                          label:
                            "TP / FP",
                          value: `${opssatEvidence.anomaly_detection?.tp ?? "—"} / ${opssatEvidence.anomaly_detection?.fp ?? "—"}`,
                        },
                        {
                          label:
                            "FN / TN",
                          value: `${opssatEvidence.anomaly_detection?.fn ?? "—"} / ${opssatEvidence.anomaly_detection?.tn ?? "—"}`,
                        },
                      ].map((m) => (
                        <div
                          key={m.label}
                          style={{
                            textAlign:
                              "center",
                          }}
                        >
                          <div
                            style={{
                              fontSize: 9,
                              color:
                                "var(--text-muted)",
                              marginBottom: 3,
                            }}
                          >
                            {m.label}
                          </div>

                          <div
                            style={{
                              fontSize: 14,
                              fontWeight: 600,
                              color:
                                "var(--text-primary)",
                              ...mono,
                            }}
                          >
                            {m.value}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </Panel>

                <Panel>
                  <Label>
                    POLICY CALIBRATION — CALIBRATED vs BASELINE
                  </Label>

                  <div
                    style={{
                      padding:
                        "14px 18px",
                    }}
                  >
                    <div
                      style={{
                        display: "grid",
                        gridTemplateColumns:
                          "repeat(3,1fr)",
                        gap: 14,
                        marginBottom: 10,
                      }}
                    >
                      {[
                        {
                          label:
                            "UNSAFE AUTO (BASELINE)",
                          value:
                            opssatEvidence
                              .policy_evaluation
                              ?.calibrated_vs_baseline
                              ?.unsafe_auto_baseline ??
                            14,
                          color:
                            "#ff3344",
                        },
                        {
                          label:
                            "UNSAFE AUTO (CALIBRATED)",
                          value:
                            opssatEvidence
                              .policy_evaluation
                              ?.calibrated_vs_baseline
                              ?.unsafe_auto_calibrated ??
                            9,
                          color:
                            "#00ff88",
                        },
                        {
                          label:
                            "RECALL IMPROVEMENT",
                          value: `+${(
                            (opssatEvidence
                              .policy_evaluation
                              ?.calibrated_vs_baseline
                              ?.recall_calibrated ??
                              0.9204) -
                            (opssatEvidence
                              .policy_evaluation
                              ?.calibrated_vs_baseline
                              ?.recall_baseline ??
                              0.8761)
                          ).toFixed(4)}`,
                          color:
                            "#00ff88",
                        },
                      ].map((m) => (
                        <div
                          key={m.label}
                          style={{
                            textAlign:
                              "center",
                            padding:
                              "6px 0",
                          }}
                        >
                          <div
                            style={{
                              fontSize: 9,
                              color:
                                "var(--text-muted)",
                              letterSpacing:
                                "0.12em",
                              marginBottom: 4,
                            }}
                          >
                            {m.label}
                          </div>

                          <div
                            style={{
                              fontSize: 26,
                              fontWeight: 700,
                              color: m.color,
                              ...mono,
                            }}
                          >
                            {m.value}
                          </div>
                        </div>
                      ))}
                    </div>

                    <div
                      style={{
                        fontSize: 10,
                        color:
                          "var(--text-dim)",
                        ...sans,
                        borderTop:
                          "1px solid var(--border)",
                        paddingTop: 8,
                      }}
                    >
                      Calibrated policy:
                      p_escalate=0.35,
                      p_pending=0.20 Â· Frozen
                      on training partition only Â·
                      Test not seen during calibration
                    </div>
                  </div>
                </Panel>

                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns:
                      "1fr 1fr",
                    gap: 12,
                  }}
                >
                  <Panel>
                    <Label>
                      TEMPORAL EVIDENCE
                    </Label>

                    <div
                      style={{
                        padding:
                          "12px 18px",
                      }}
                    >
                      {[
                        {
                          label:
                            "CLASSIFICATION",
                          value: `${opssatEvidence.temporal?.classification ?? "B"} — ${opssatEvidence.temporal?.classification_label ?? "LIMITED"}`,
                        },
                        {
                          label:
                            "SERIAL DEPENDENCE",
                          value:
                            "chi2=85.08  p<0.001",
                        },
                        {
                          label:
                            "A→A TRANSITION",
                          value: `${(
                            (opssatEvidence
                              .temporal
                              ?.anomaly_to_anomaly_prob ??
                              0.366) *
                            100
                          ).toFixed(
                            1
                          )}% vs ${(
                            (opssatEvidence
                              .temporal
                              ?.normal_to_anomaly_prob ??
                              0.164) *
                            100
                          ).toFixed(
                            1
                          )}% N→A`,
                        },
                        {
                          label:
                            "INTER-SEG GAP",
                          value: `${opssatEvidence.temporal?.median_intersegment_gap_s ?? 1}s median`,
                        },
                        {
                          label:
                            "30-MIN VALID",
                          value:
                            "NO — not supported on OPS-SAT-AD",
                        },
                      ].map((r) => (
                        <div
                          key={r.label}
                          style={{
                            display:
                              "flex",
                            justifyContent:
                              "space-between",
                            marginBottom:
                              6,
                          }}
                        >
                          <span
                            style={{
                              fontSize: 9,
                              color:
                                "var(--text-muted)",
                              letterSpacing:
                                "0.1em",
                            }}
                          >
                            {r.label}
                          </span>

                          <span
                            style={{
                              fontSize: 10,
                              color:
                                r.label ===
                                "30-MIN VALID"
                                  ? "#ff8866"
                                  : "var(--text-primary)",
                              fontWeight: 600,
                              ...mono,
                              maxWidth:
                                "55%",
                              textAlign:
                                "right",
                            }}
                          >
                            {r.value}
                          </span>
                        </div>
                      ))}
                    </div>
                  </Panel>

                  <Panel>
                    <Label>
                      DISAGREEMENT EVIDENCE
                    </Label>

                    <div
                      style={{
                        padding:
                          "12px 18px",
                      }}
                    >
                      {[
                        {
                          label:
                            "EW PRECISION (UNCOND)",
                          value: `${(
                            (opssatEvidence
                              .disagreement
                              ?.early_warning_precision_unconditioned ??
                              0.054) *
                            100
                          ).toFixed(
                            1
                          )}%`,
                        },
                        {
                          label:
                            "EW PRECISION (CUR>0.25)",
                          value: `${(
                            (opssatEvidence
                              .disagreement
                              ?.early_warning_precision_cond_cur025 ??
                              0.387) *
                            100
                          ).toFixed(
                            1
                          )}%`,
                        },
                        {
                          label:
                            "CONDITIONING HELPS",
                          value:
                            opssatEvidence
                              .disagreement
                              ?.conditioning_improves_precision
                              ? "YES"
                              : "NO",
                        },
                      ].map((r) => (
                        <div
                          key={r.label}
                          style={{
                            display:
                              "flex",
                            justifyContent:
                              "space-between",
                            marginBottom:
                              6,
                          }}
                        >
                          <span
                            style={{
                              fontSize: 9,
                              color:
                                "var(--text-muted)",
                              letterSpacing:
                                "0.1em",
                            }}
                          >
                            {r.label}
                          </span>

                          <span
                            style={{
                              fontSize: 10,
                              color:
                                "var(--text-primary)",
                              fontWeight: 600,
                              ...mono,
                            }}
                          >
                            {r.value}
                          </span>
                        </div>
                      ))}

                      <div
                        style={{
                          fontSize: 10,
                          color:
                            "var(--text-dim)",
                          marginTop: 8,
                          ...sans,
                          lineHeight: 1.6,
                        }}
                      >
                        {
                          opssatEvidence
                            .disagreement
                            ?.recommendation
                        }
                      </div>
                    </div>
                  </Panel>
                </div>

                <Panel>
                  <Label>
                    VERIFIED LIMITATIONS
                  </Label>

                  <div
                    style={{
                      padding:
                        "12px 18px",
                      fontSize: 10,
                      color:
                        "var(--text-muted)",
                      lineHeight: 1.8,
                      ...sans,
                    }}
                  >
                    <div>
                      â€¢ 30-minute prediction
                      horizon is NOT validated
                      on OPS-SAT-AD
                    </div>

                    <div>
                      â€¢ Four-class simulation
                      taxonomy is NOT validated
                      by binary OPS-SAT labels
                    </div>

                    <div>
                      â€¢ Temporal early-warning
                      window is near-zero
                      (median inter-segment gap
                      = 1s)
                    </div>

                    <div>
                      â€¢ Decision utility weights
                      are prototype values — not
                      from real mission operators
                    </div>

                    <div>
                      â€¢ Single mission dataset
                      (OPS-SAT, 9 channels, ~5
                      months) — generalizability
                      unknown
                    </div>

                    <div>
                      â€¢ No human operator
                      evaluation of Granite
                      explanations on real data
                    </div>
                  </div>
                </Panel>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}





