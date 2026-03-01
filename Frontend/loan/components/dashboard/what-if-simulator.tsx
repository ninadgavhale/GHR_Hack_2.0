"use client"

import React from "react"
import { motion } from "motion/react"
import { SlidersHorizontal, TrendingUp } from "lucide-react"
import { Slider } from "@/components/ui/slider"
import { Label } from "@/components/ui/label"
import { useLoanData } from "@/context/loan-data-context"
import { FeatureAttributionChart } from "@/components/dashboard/feature-attribution-chart"

function formatCurrency(value: number) {
  return new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 }).format(value)
}

export function WhatIfSimulator() {
  const { whatIfState, updateWhatIf, whatIfScore, whatIfFeatures } = useLoanData()

  const sliders = [
    {
      key: "income" as const,
      label: "Annual Income",
      min: 20000,
      max: 300000,
      step: 5000,
      value: whatIfState.income,
      format: formatCurrency,
    },
    {
      key: "creditScore" as const,
      label: "Credit Score",
      min: 300,
      max: 850,
      step: 5,
      value: whatIfState.creditScore,
      format: (v: number) => v.toString(),
    },
    {
      key: "dti" as const,
      label: "Debt-to-Income %",
      min: 0,
      max: 60,
      step: 1,
      value: whatIfState.dti,
      format: (v: number) => `${v}%`,
    },
    {
      key: "downPayment" as const,
      label: "Down Payment",
      min: 0,
      max: 200000,
      step: 5000,
      value: whatIfState.downPayment,
      format: formatCurrency,
    },
    {
      key: "loanAmount" as const,
      label: "Loan Amount",
      min: 5000,
      max: 500000,
      step: 5000,
      value: whatIfState.loanAmount,
      format: formatCurrency,
    },
  ]

  const scoreColor =
    whatIfScore >= 70 ? "text-success" : whatIfScore >= 45 ? "text-amber-500" : "text-destructive"
  const scoreLabel =
    whatIfScore >= 70 ? "Likely Approved" : whatIfScore >= 45 ? "Under Review" : "Likely Denied"

  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-center gap-2">
        <SlidersHorizontal className="h-5 w-5 text-accent" />
        <h2 className="text-lg font-semibold text-foreground">What-If Simulator</h2>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Sliders panel */}
        <div className="rounded-xl border border-border bg-card p-6">
          <div className="mb-6 flex items-center justify-between">
            <p className="text-sm font-medium text-muted-foreground">Adjust parameters</p>
            <div className="flex items-center gap-2">
              <TrendingUp className={`h-4 w-4 ${scoreColor}`} />
              <span className={`text-2xl font-bold ${scoreColor}`}>{whatIfScore}</span>
              <span className="text-xs text-muted-foreground">/100</span>
            </div>
          </div>
          <p className={`mb-6 text-center text-sm font-semibold ${scoreColor}`}>{scoreLabel}</p>
          <div className="flex flex-col gap-6">
            {sliders.map((s) => (
              <div key={s.key} className="flex flex-col gap-2">
                <div className="flex items-center justify-between">
                  <Label className="text-xs font-medium text-muted-foreground">{s.label}</Label>
                  <span className="text-xs font-semibold text-foreground">{s.format(s.value)}</span>
                </div>
                <Slider
                  min={s.min}
                  max={s.max}
                  step={s.step}
                  value={[s.value]}
                  onValueChange={([v]) => updateWhatIf({ [s.key]: v })}
                />
              </div>
            ))}
          </div>
        </div>

        {/* Live chart */}
        <motion.div
          key={whatIfScore}
          initial={{ opacity: 0.6 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.3 }}
        >
          <FeatureAttributionChart features={whatIfFeatures} title="Simulated Feature Impact" />
        </motion.div>
      </div>
    </div>
  )
}
