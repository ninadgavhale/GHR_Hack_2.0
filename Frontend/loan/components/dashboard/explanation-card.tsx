"use client"

import React from "react"
import { motion } from "motion/react"
import { MessageSquareText } from "lucide-react"

export function ExplanationCard({ explanation }: { explanation: string }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: 0.2 }}
      className="rounded-xl border border-border bg-card p-6"
    >
      <div className="mb-3 flex items-center gap-2">
        <MessageSquareText className="h-4 w-4 text-accent" />
        <h3 className="text-sm font-semibold text-foreground">Plain-English Explanation</h3>
      </div>
      <p className="text-sm text-muted-foreground leading-relaxed">{explanation}</p>
    </motion.div>
  )
}
