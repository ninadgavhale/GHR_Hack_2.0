"use client"

import React from "react"
import { useRouter } from "next/navigation"
import { motion } from "motion/react"
import { FileText, ArrowRight } from "lucide-react"
import { useAuth } from "@/context/auth-context"
import { useLoanData } from "@/context/loan-data-context"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { DecisionStatusCard } from "@/components/dashboard/decision-status-card"
import { ConfidenceGauge } from "@/components/dashboard/confidence-gauge"
import { FairnessScoreBadge } from "@/components/dashboard/fairness-badge"
import { FeatureAttributionChart } from "@/components/dashboard/feature-attribution-chart"
import { ExplanationCard } from "@/components/dashboard/explanation-card"
import { WhatIfSimulator } from "@/components/dashboard/what-if-simulator"
import Link from "next/link"

const fadeIn = {
  hidden: { opacity: 0, y: 16 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.08, duration: 0.4 },
  }),
}

function AuthGate({ children }: { children: React.ReactNode }) {
  const { isAuthenticated } = useAuth()
  const router = useRouter()

  if (!isAuthenticated) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-4 p-8">
        <p className="text-muted-foreground">Please sign in to view your dashboard.</p>
        <Button onClick={() => router.push("/login")} className="bg-accent text-accent-foreground hover:bg-accent/90">
          Go to Login
        </Button>
      </div>
    )
  }

  return <>{children}</>
}

export default function DashboardPage() {
  const { currentApplication, applications } = useLoanData()
  const app = currentApplication

  return (
    <AuthGate>
      <div className="flex flex-col gap-8 p-4 md:p-6 lg:p-8">
        {/* Header */}
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h1 className="text-2xl font-bold text-foreground">XAI Dashboard</h1>
            <p className="text-sm text-muted-foreground">
              Explainable AI analysis for your loan application
            </p>
          </div>
          <div className="flex items-center gap-2">
            {app && (
              <Badge variant="outline" className="text-xs">
                {app.id}
              </Badge>
            )}
            <Link href="/apply">
              <Button size="sm" className="bg-accent text-accent-foreground hover:bg-accent/90 gap-1.5">
                <FileText className="h-3.5 w-3.5" /> New Application
              </Button>
            </Link>
          </div>
        </div>

        {app ? (
          <>
            {/* Bento Grid: Status / Confidence / Fairness */}
            <div className="grid gap-4 sm:grid-cols-3">
              <motion.div initial="hidden" animate="visible" variants={fadeIn} custom={0}>
                <DecisionStatusCard status={app.status} score={app.score} />
              </motion.div>
              <motion.div initial="hidden" animate="visible" variants={fadeIn} custom={1}>
                <ConfidenceGauge confidence={app.confidence} />
              </motion.div>
              <motion.div initial="hidden" animate="visible" variants={fadeIn} custom={2}>
                <FairnessScoreBadge score={app.fairnessScore} />
              </motion.div>
            </div>

            {/* Feature Attribution */}
            <motion.div initial="hidden" animate="visible" variants={fadeIn} custom={3}>
              <FeatureAttributionChart features={app.features} />
            </motion.div>

            {/* Explanation */}
            <motion.div initial="hidden" animate="visible" variants={fadeIn} custom={4}>
              <ExplanationCard explanation={app.explanation} />
            </motion.div>

            {/* Application Details */}
            <motion.div
              initial="hidden"
              animate="visible"
              variants={fadeIn}
              custom={5}
              className="rounded-xl border border-border bg-card p-6"
            >
              <h3 className="mb-4 text-sm font-semibold text-foreground">Application Details</h3>
              <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                {[
                  { label: "Loan Amount", value: `$${app.amount.toLocaleString()}` },
                  { label: "Term", value: `${app.term} years` },
                  { label: "Purpose", value: app.purpose },
                  { label: "Income", value: `$${app.income.toLocaleString()}` },
                  { label: "Credit Score", value: app.creditScore.toString() },
                  { label: "DTI Ratio", value: `${app.dti}%` },
                  { label: "Employment", value: app.employment },
                  { label: "Down Payment", value: `$${app.downPayment.toLocaleString()}` },
                ].map((d) => (
                  <div key={d.label} className="flex flex-col gap-1">
                    <span className="text-xs text-muted-foreground">{d.label}</span>
                    <span className="text-sm font-semibold text-foreground">{d.value}</span>
                  </div>
                ))}
              </div>
            </motion.div>

            {/* What-If Simulator */}
            <motion.div initial="hidden" animate="visible" variants={fadeIn} custom={6}>
              <WhatIfSimulator />
            </motion.div>

            {/* Quick links to other applications */}
            {applications.length > 1 && (
              <motion.div
                initial="hidden"
                animate="visible"
                variants={fadeIn}
                custom={7}
                className="rounded-xl border border-border bg-card p-6"
              >
                <div className="mb-3 flex items-center justify-between">
                  <h3 className="text-sm font-semibold text-foreground">Recent Applications</h3>
                  <Link href="/history" className="flex items-center gap-1 text-xs font-medium text-accent hover:underline">
                    View all <ArrowRight className="h-3 w-3" />
                  </Link>
                </div>
                <div className="flex flex-col gap-2">
                  {applications.slice(0, 3).map((a) => (
                    <button
                      key={a.id}
                      onClick={() => {
                        /* handled by context */
                      }}
                      className="flex items-center justify-between rounded-lg border border-border px-4 py-3 text-left transition-colors hover:bg-secondary"
                    >
                      <div className="flex flex-col gap-0.5">
                        <span className="text-sm font-medium text-foreground">{a.purpose}</span>
                        <span className="text-xs text-muted-foreground">{a.id} - {a.date}</span>
                      </div>
                      <Badge
                        variant={a.status === "approved" ? "default" : a.status === "denied" ? "destructive" : "secondary"}
                        className={
                          a.status === "approved"
                            ? "bg-success text-success-foreground"
                            : a.status === "in_review"
                            ? "bg-amber-500/10 text-amber-600"
                            : ""
                        }
                      >
                        {a.status.replace("_", " ")}
                      </Badge>
                    </button>
                  ))}
                </div>
              </motion.div>
            )}
          </>
        ) : (
          <div className="flex flex-col items-center justify-center gap-4 rounded-xl border border-border bg-card p-12">
            <FileText className="h-10 w-10 text-muted-foreground" />
            <p className="text-muted-foreground">No applications yet.</p>
            <Link href="/apply">
              <Button className="bg-accent text-accent-foreground hover:bg-accent/90 gap-2">
                Start Your Application <ArrowRight className="h-4 w-4" />
              </Button>
            </Link>
          </div>
        )}
      </div>
    </AuthGate>
  )
}
