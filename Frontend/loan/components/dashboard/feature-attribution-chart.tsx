"use client"

import React from "react"
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, ReferenceLine } from "recharts"
import type { FeatureAttribution } from "@/context/loan-data-context"

interface Props {
  features: FeatureAttribution[]
  title?: string
}

export function FeatureAttributionChart({ features, title = "Feature Attribution" }: Props) {
  const data = features
    .map((f) => ({ name: f.name, value: f.value, fill: f.impact === "positive" ? "var(--success)" : "var(--destructive)" }))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))

  return (
    <div className="rounded-xl border border-border bg-card p-6">
      <h3 className="mb-4 text-sm font-semibold text-foreground">{title}</h3>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ left: 8, right: 16, top: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="var(--border)" />
            <XAxis
              type="number"
              domain={[-0.4, 0.4]}
              tickFormatter={(v: number) => v.toFixed(1)}
              tick={{ fontSize: 11, fill: "var(--muted-foreground)" }}
              axisLine={{ stroke: "var(--border)" }}
            />
            <YAxis
              type="category"
              dataKey="name"
              width={100}
              tick={{ fontSize: 11, fill: "var(--muted-foreground)" }}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip
              contentStyle={{
                background: "var(--card)",
                border: "1px solid var(--border)",
                borderRadius: "8px",
                fontSize: "12px",
                color: "var(--foreground)",
              }}
              formatter={(value: number) => [value.toFixed(3), "Impact"]}
            />
            <ReferenceLine x={0} stroke="var(--muted-foreground)" strokeWidth={1} />
            <Bar dataKey="value" radius={[4, 4, 4, 4]} maxBarSize={20}>
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.fill} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-3 flex items-center justify-center gap-6 text-xs text-muted-foreground">
        <span className="flex items-center gap-1.5"><span className="inline-block h-2.5 w-2.5 rounded-sm bg-success" /> Positive Impact</span>
        <span className="flex items-center gap-1.5"><span className="inline-block h-2.5 w-2.5 rounded-sm bg-destructive" /> Negative Impact</span>
      </div>
    </div>
  )
}
