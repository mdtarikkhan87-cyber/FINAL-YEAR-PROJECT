"use client";

import {
  Bar,
  BarChart,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { featureLabel, type ExplanationResponse } from "@/lib/api";

const UP = "#b91c1c";
const DOWN = "#15803d";

interface Row {
  label: string;
  value: number;
  raw: string;
  feature: string;
}

function ShapTooltip({
  active,
  payload,
  unitSuffix,
}: {
  active?: boolean;
  payload?: Array<{ payload: Row }>;
  unitSuffix: string;
}) {
  if (!active || !payload?.length) return null;
  const row = payload[0].payload;
  return (
    <div className="rounded-lg border border-ink-200 bg-white px-3 py-2 shadow-lg">
      <p className="text-xs font-semibold text-ink-950">{row.label}</p>
      <p className="mt-1 text-xs text-ink-600">
        Case value: <span className="font-mono text-ink-900">{row.raw}</span>
      </p>
      <p className="mt-0.5 text-xs text-ink-600">
        Contribution:{" "}
        <span
          className="font-mono font-semibold"
          style={{ color: row.value >= 0 ? UP : DOWN }}
        >
          {row.value >= 0 ? "+" : ""}
          {row.value.toFixed(2)} {unitSuffix}
        </span>
      </p>
    </div>
  );
}

export default function ShapChart({
  explanation,
  topN = 8,
}: {
  explanation: ExplanationResponse;
  topN?: number;
}) {
  const isRegression = explanation.task === "regression";
  const unitSuffix = isRegression ? "days" : "log-odds";

  const rows: Row[] = explanation.contributions.slice(0, topN).map((c) => ({
    label: featureLabel(c.feature),
    value: c.shap_value,
    raw: c.value,
    feature: c.feature,
  }));

  const magnitude = Math.max(...rows.map((r) => Math.abs(r.value)), 0.0001);
  const pad = magnitude * 0.15;

  return (
    <div>
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <p className="text-sm text-ink-600">
          Starting point{" "}
          <span className="font-mono font-semibold text-ink-900">
            {explanation.base_value.toFixed(isRegression ? 0 : 2)}
          </span>{" "}
          {isRegression ? "days" : `log-odds of ${explanation.explained_class}`}
          {" → "}
          <span className="font-mono font-semibold text-ink-900">
            {explanation.reconstruction_check.toFixed(isRegression ? 0 : 2)}
          </span>
        </p>
        <div className="flex items-center gap-3 text-xs text-ink-500">
          <span className="inline-flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-sm" style={{ background: UP }} />
            increases
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-sm" style={{ background: DOWN }} />
            decreases
          </span>
        </div>
      </div>

      <div className="mt-4" style={{ height: rows.length * 38 + 28 }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={rows}
            layout="vertical"
            margin={{ top: 4, right: 16, bottom: 4, left: 4 }}
            barCategoryGap={8}
          >
            <XAxis
              type="number"
              domain={[-magnitude - pad, magnitude + pad]}
              tick={{ fontSize: 11, fill: "#65748f" }}
              tickFormatter={(v: number) =>
                isRegression ? `${Math.round(v)}` : v.toFixed(2)
              }
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              type="category"
              dataKey="label"
              width={168}
              tick={{ fontSize: 11, fill: "#424c60" }}
              axisLine={false}
              tickLine={false}
            />
            <ReferenceLine x={0} stroke="#b0b8c9" />
            <Tooltip
              cursor={{ fill: "rgba(16,24,40,0.04)" }}
              content={<ShapTooltip unitSuffix={unitSuffix} />}
            />
            <Bar dataKey="value" radius={[3, 3, 3, 3]} isAnimationActive={false}>
              {rows.map((r) => (
                <Cell key={r.feature} fill={r.value >= 0 ? UP : DOWN} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <p className="mt-3 text-xs leading-relaxed text-ink-500">{explanation.note}</p>
    </div>
  );
}
