"use client";

import {
  Bar,
  BarChart,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import {
  OUTCOME_COLORS,
  OUTCOME_LABELS,
  type OutcomeResponse,
} from "@/lib/api";

interface Row {
  key: string;
  label: string;
  pct: number;
  color: string;
}

function OutcomeTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: Array<{ payload: Row }>;
}) {
  if (!active || !payload?.length) return null;
  const row = payload[0].payload;
  return (
    <div className="rounded-lg border border-ink-200 bg-white px-3 py-2 shadow-lg">
      <p className="text-xs font-semibold text-ink-950">{row.label}</p>
      <p className="mt-0.5 text-xs text-ink-600">
        Probability{" "}
        <span className="font-mono font-semibold text-ink-900">
          {row.pct.toFixed(1)}%
        </span>
      </p>
    </div>
  );
}

export default function OutcomeChart({ result }: { result: OutcomeResponse }) {
  const rows: Row[] = result.probabilities.map((p) => ({
    key: p.outcome,
    label: OUTCOME_LABELS[p.outcome] ?? p.outcome,
    pct: p.probability * 100,
    color: OUTCOME_COLORS[p.outcome] ?? "#65748f",
  }));

  return (
    <div>
      <div style={{ height: rows.length * 42 + 24 }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={rows}
            layout="vertical"
            margin={{ top: 4, right: 48, bottom: 4, left: 4 }}
            barCategoryGap={10}
          >
            <XAxis type="number" domain={[0, 100]} hide />
            <YAxis
              type="category"
              dataKey="label"
              width={124}
              tick={{ fontSize: 12, fill: "#424c60" }}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip
              cursor={{ fill: "rgba(16,24,40,0.04)" }}
              content={<OutcomeTooltip />}
            />
            <Bar
              dataKey="pct"
              radius={[3, 3, 3, 3]}
              isAnimationActive={false}
              label={{
                position: "right",
                formatter: (v: number) => `${v.toFixed(1)}%`,
                fontSize: 11,
                fill: "#505d76",
              }}
            >
              {rows.map((r) => (
                <Cell key={r.key} fill={r.color} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
