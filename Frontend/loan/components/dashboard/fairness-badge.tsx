"use client"

import React from "react"
import { motion } from "motion/react"
import { ShieldCheck } from "lucide-react"
import { cn } from "@/lib/utils"

export function FairnessScoreBadge({ score }: { score: number }) {
  const level = score >= 95 ? "Excellent" : score >= 85 ? "Good" : score >= 70 ? "Fair" : "Needs Review"
  const colorClass =
    score >= 95 ? "text-success" : score >= 85 ? "text-accent" : score >= 70 ? "text-amber-500" : "text-destructive"
  const bgClass =
    score >= 95 ? "bg-success/10" : score >= 85 ? "bg-accent/10" : score >= 70 ? "bg-amber-500/10" : "bg-destructive/10"

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.4, delay: 0.1 }}
      className="flex flex-col items-center justify-center rounded-xl border border-border bg-card p-6"
    >
      <div className={cn("flex h-12 w-12 items-center justify-center rounded-full", bgClass)}>
        <ShieldCheck className={cn("h-6 w-6", colorClass)} />
      </div>
      <p className="mt-3 text-2xl font-bold text-foreground">{score}%</p>
      <p className={cn("mt-1 text-xs font-semibold", colorClass)}>{level}</p>
      <p className="mt-1 text-xs text-muted-foreground">Fairness Score</p>
    </motion.div>
  )
}
