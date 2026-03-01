"use client"

import React from "react"
import { motion } from "motion/react"

export function ConfidenceGauge({ confidence }: { confidence: number }) {
  const radius = 54
  const circumference = 2 * Math.PI * radius
  const strokeDashoffset = circumference - (confidence / 100) * circumference
  const color =
    confidence >= 85 ? "var(--success)" : confidence >= 65 ? "var(--accent)" : "var(--destructive)"

  return (
    <div className="flex flex-col items-center justify-center rounded-xl border border-border bg-card p-6">
      <p className="mb-3 text-sm font-medium text-muted-foreground">AI Confidence</p>
      <div className="relative flex items-center justify-center">
        <svg width="128" height="128" viewBox="0 0 128 128" className="-rotate-90">
          <circle
            cx="64"
            cy="64"
            r={radius}
            fill="none"
            stroke="var(--border)"
            strokeWidth="10"
          />
          <motion.circle
            cx="64"
            cy="64"
            r={radius}
            fill="none"
            stroke={color}
            strokeWidth="10"
            strokeLinecap="round"
            strokeDasharray={circumference}
            initial={{ strokeDashoffset: circumference }}
            animate={{ strokeDashoffset }}
            transition={{ duration: 1.2, ease: "easeOut" }}
          />
        </svg>
        <div className="absolute flex flex-col items-center">
          <span className="text-2xl font-bold text-foreground">{confidence}%</span>
          <span className="text-xs text-muted-foreground">Confidence</span>
        </div>
      </div>
    </div>
  )
}
