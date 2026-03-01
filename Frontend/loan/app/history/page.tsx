"use client"

import React, { useState } from "react"
import { useRouter } from "next/navigation"
import { AnimatePresence, motion } from "motion/react"
import { History, Eye, X, FileText, ArrowRight } from "lucide-react"
import { useAuth } from "@/context/auth-context"
import { useLoanData, type LoanApplication } from "@/context/loan-data-context"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { FeatureAttributionChart } from "@/components/dashboard/feature-attribution-chart"
import Link from "next/link"

function StatusBadge({ status }: { status: string }) {
  const config: Record<string, { className: string; label: string }> = {
    approved: { className: "bg-success text-success-foreground", label: "Approved" },
    denied: { className: "bg-destructive text-destructive-foreground", label: "Denied" },
    in_review: { className: "bg-amber-500/10 text-amber-600 border border-amber-500/20", label: "In Review" },
    pending: { className: "bg-muted text-muted-foreground", label: "Pending" },
  }
  const c = config[status] || config.pending
  return <Badge className={c.className}>{c.label}</Badge>
}

function RationaleModal({ app, onClose }: { app: LoanApplication; onClose: () => void }) {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
      onClick={onClose}
    >
      <motion.div
        initial={{ scale: 0.95, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.95, opacity: 0 }}
        onClick={(e) => e.stopPropagation()}
        className="relative w-full max-w-2xl max-h-[80vh] overflow-y-auto rounded-xl border border-border bg-card p-6 shadow-xl"
      >
        <button
          onClick={onClose}
          className="absolute right-4 top-4 text-muted-foreground hover:text-foreground"
        >
          <X className="h-5 w-5" />
          <span className="sr-only">Close</span>
        </button>
        <div className="flex items-center gap-3 mb-4">
          <h2 className="text-lg font-bold text-foreground">Decision Rationale</h2>
          <StatusBadge status={app.status} />
        </div>
        <div className="mb-2 flex items-center gap-4 text-sm text-muted-foreground">
          <span>{app.id}</span>
          <span>{app.date}</span>
          <span>${app.amount.toLocaleString()}</span>
          <span>{app.purpose}</span>
        </div>
        <div className="mb-6 rounded-lg bg-secondary p-4">
          <p className="text-sm text-foreground leading-relaxed">{app.explanation}</p>
        </div>
        <div className="mb-4 grid grid-cols-3 gap-3">
          <div className="flex flex-col items-center rounded-lg border border-border p-3">
            <span className="text-lg font-bold text-foreground">{app.score}</span>
            <span className="text-xs text-muted-foreground">AI Score</span>
          </div>
          <div className="flex flex-col items-center rounded-lg border border-border p-3">
            <span className="text-lg font-bold text-foreground">{app.confidence}%</span>
            <span className="text-xs text-muted-foreground">Confidence</span>
          </div>
          <div className="flex flex-col items-center rounded-lg border border-border p-3">
            <span className="text-lg font-bold text-foreground">{app.fairnessScore}%</span>
            <span className="text-xs text-muted-foreground">Fairness</span>
          </div>
        </div>
        <FeatureAttributionChart features={app.features} title="Feature Impact Analysis" />
      </motion.div>
    </motion.div>
  )
}

function AuthGate({ children }: { children: React.ReactNode }) {
  const { isAuthenticated } = useAuth()
  const router = useRouter()

  if (!isAuthenticated) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-4 p-8">
        <p className="text-muted-foreground">Please sign in to view your history.</p>
        <Button onClick={() => router.push("/login")} className="bg-accent text-accent-foreground hover:bg-accent/90">
          Go to Login
        </Button>
      </div>
    )
  }

  return <>{children}</>
}

export default function HistoryPage() {
  const { applications, setCurrentApplication } = useLoanData()
  const [selectedApp, setSelectedApp] = useState<LoanApplication | null>(null)
  const router = useRouter()

  const viewRationale = (app: LoanApplication) => {
    setSelectedApp(app)
  }

  const viewOnDashboard = (app: LoanApplication) => {
    setCurrentApplication(app)
    router.push("/dashboard")
  }

  return (
    <AuthGate>
      <div className="flex flex-col gap-6 p-4 md:p-6 lg:p-8">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-center gap-3">
            <History className="h-6 w-6 text-accent" />
            <div>
              <h1 className="text-2xl font-bold text-foreground">Application History</h1>
              <p className="text-sm text-muted-foreground">
                {applications.length} application{applications.length !== 1 ? "s" : ""} on record
              </p>
            </div>
          </div>
          <Link href="/apply">
            <Button size="sm" className="bg-accent text-accent-foreground hover:bg-accent/90 gap-1.5">
              <FileText className="h-3.5 w-3.5" /> New Application
            </Button>
          </Link>
        </div>

        {applications.length === 0 ? (
          <div className="flex flex-col items-center justify-center gap-4 rounded-xl border border-border bg-card p-12">
            <FileText className="h-10 w-10 text-muted-foreground" />
            <p className="text-muted-foreground">No applications yet.</p>
            <Link href="/apply">
              <Button className="bg-accent text-accent-foreground hover:bg-accent/90 gap-2">
                Start Your Application <ArrowRight className="h-4 w-4" />
              </Button>
            </Link>
          </div>
        ) : (
          <div className="rounded-xl border border-border bg-card">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Application ID</TableHead>
                  <TableHead>Date</TableHead>
                  <TableHead>Purpose</TableHead>
                  <TableHead className="text-right">Amount</TableHead>
                  <TableHead>Score</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {applications.map((app) => (
                  <TableRow
                    key={app.id}
                    className="transition-colors hover:bg-muted/50"
                  >
                    <TableCell className="font-mono text-xs">{app.id}</TableCell>
                    <TableCell className="text-sm">{app.date}</TableCell>
                    <TableCell className="text-sm">{app.purpose}</TableCell>
                    <TableCell className="text-right text-sm font-medium">
                      ${app.amount.toLocaleString()}
                    </TableCell>
                    <TableCell>
                      <span className="text-sm font-semibold">{app.score}</span>
                      <span className="text-xs text-muted-foreground">/100</span>
                    </TableCell>
                    <TableCell>
                      <StatusBadge status={app.status} />
                    </TableCell>
                    <TableCell className="text-right">
                      <div className="flex items-center justify-end gap-1">
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => viewRationale(app)}
                          className="gap-1 text-xs"
                        >
                          <Eye className="h-3.5 w-3.5" /> Rationale
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => viewOnDashboard(app)}
                          className="gap-1 text-xs"
                        >
                          <ArrowRight className="h-3.5 w-3.5" /> Dashboard
                        </Button>
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        )}
      </div>

      <AnimatePresence>
        {selectedApp && <RationaleModal app={selectedApp} onClose={() => setSelectedApp(null)} />}
      </AnimatePresence>
    </AuthGate>
  )
}
