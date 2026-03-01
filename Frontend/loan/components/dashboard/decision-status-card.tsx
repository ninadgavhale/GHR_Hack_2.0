"use client"

import React from "react"
import { motion } from "motion/react"
import { CheckCircle2, XCircle, Clock, AlertTriangle } from "lucide-react"
import { cn } from "@/lib/utils"
import type { LoanStatus } from "@/context/loan-data-context"

const statusConfig: Record<LoanStatus, { label: string; icon: React.ElementType; colorClass: string; bgClass: string }> = {
  approved: { label: "Approved", icon: CheckCircle2, colorClass: "text-success", bgClass: "bg-success/10" },
  denied: { label: "Denied", icon: XCircle, colorClass: "text-destructive", bgClass: "bg-destructive/10" },
  in_review: { label: "In Review", icon: Clock, colorClass: "text-amber-500", bgClass: "bg-amber-500/10" },
  pending: { label: "Pending", icon: AlertTriangle, colorClass: "text-muted-foreground", bgClass: "bg-muted" },
}

export function DecisionStatusCard({ status, score }: { status: LoanStatus; score: number }) {
  const config = statusConfig[status]
  const Icon = config.icon

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.4 }}
      className="flex flex-col items-center justify-center rounded-xl border border-border bg-card p-6"
    >
      <div className={cn("flex h-14 w-14 items-center justify-center rounded-full", config.bgClass)}>
        <Icon className={cn("h-7 w-7", config.colorClass)} />
      </div>
      <p className={cn("mt-3 text-lg font-bold", config.colorClass)}>{config.label}</p>
      <p className="mt-1 text-xs text-muted-foreground">AI Score: {score}/100</p>
    </motion.div>
  )
}
