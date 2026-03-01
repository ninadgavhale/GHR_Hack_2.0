"use client"

import React, { useState } from "react"
import { useRouter } from "next/navigation"
import { motion } from "motion/react"
import {
  User,
  Mail,
  CreditCard,
  Calendar,
  DollarSign,
  Upload,
  FileText,
  CheckCircle2,
  X,
  BrainCircuit,
} from "lucide-react"
import { useAuth } from "@/context/auth-context"
import { useLoanData } from "@/context/loan-data-context"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { cn } from "@/lib/utils"

const fadeIn = {
  hidden: { opacity: 0, y: 12 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.08, duration: 0.4 },
  }),
}

const mockDocuments = [
  { name: "ID_Verification.pdf", status: "uploaded", date: "2024-12-10" },
  { name: "Income_Proof_2024.pdf", status: "uploaded", date: "2024-12-12" },
  { name: "Tax_Returns_2023.pdf", status: "uploaded", date: "2024-12-14" },
]

function AuthGate({ children }: { children: React.ReactNode }) {
  const { isAuthenticated } = useAuth()
  const router = useRouter()

  if (!isAuthenticated) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-4 p-8">
        <p className="text-muted-foreground">Please sign in to view your profile.</p>
        <Button onClick={() => router.push("/login")} className="bg-accent text-accent-foreground hover:bg-accent/90">
          Go to Login
        </Button>
      </div>
    )
  }

  return <>{children}</>
}

export default function ProfilePage() {
  const { user } = useAuth()
  const { currentApplication } = useLoanData()
  const [uploadedFiles, setUploadedFiles] = useState<string[]>([])
  const [dragging, setDragging] = useState(false)

  const app = currentApplication
  const totalPaid = app ? Math.round(app.amount * 0.18) : 0
  const remaining = app ? app.amount - totalPaid : 0
  const paymentProgress = app ? Math.round((totalPaid / app.amount) * 100) : 0
  const monthlyPayment = app ? Math.round(app.amount / (app.term * 12)) : 0

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragging(false)
    const files = Array.from(e.dataTransfer.files)
    setUploadedFiles((prev) => [...prev, ...files.map((f) => f.name)])
  }

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const files = Array.from(e.target.files)
      setUploadedFiles((prev) => [...prev, ...files.map((f) => f.name)])
    }
  }

  return (
    <AuthGate>
      <div className="flex flex-col gap-6 p-4 md:p-6 lg:p-8">
        <div>
          <h1 className="text-2xl font-bold text-foreground">Profile & Loan Info</h1>
          <p className="mt-1 text-sm text-muted-foreground">Your account details and active loan information</p>
        </div>

        <div className="grid gap-6 lg:grid-cols-3">
          {/* Profile Card */}
          <motion.div initial="hidden" animate="visible" variants={fadeIn} custom={0} className="lg:col-span-1">
            <div className="rounded-xl border border-border bg-card p-6">
              <div className="flex flex-col items-center">
                <div className="flex h-16 w-16 items-center justify-center rounded-full bg-accent text-accent-foreground text-xl font-bold">
                  {user?.name.charAt(0).toUpperCase()}
                </div>
                <h2 className="mt-3 text-lg font-bold text-foreground">{user?.name}</h2>
                <p className="text-sm text-muted-foreground">{user?.email}</p>
                <Badge variant="outline" className="mt-2 text-xs">Active Member</Badge>
              </div>
              <div className="mt-6 flex flex-col gap-3">
                <div className="flex items-center gap-3 text-sm">
                  <User className="h-4 w-4 text-muted-foreground shrink-0" />
                  <span className="text-muted-foreground">ID:</span>
                  <span className="font-mono text-xs text-foreground">{user?.id}</span>
                </div>
                <div className="flex items-center gap-3 text-sm">
                  <Mail className="h-4 w-4 text-muted-foreground shrink-0" />
                  <span className="text-muted-foreground">Email:</span>
                  <span className="text-foreground">{user?.email}</span>
                </div>
                <div className="flex items-center gap-3 text-sm">
                  <BrainCircuit className="h-4 w-4 text-muted-foreground shrink-0" />
                  <span className="text-muted-foreground">Member since:</span>
                  <span className="text-foreground">2024</span>
                </div>
              </div>
            </div>
          </motion.div>

          {/* Active Loan Details */}
          <motion.div initial="hidden" animate="visible" variants={fadeIn} custom={1} className="lg:col-span-2">
            <div className="rounded-xl border border-border bg-card p-6">
              <div className="mb-6 flex items-center gap-2">
                <CreditCard className="h-5 w-5 text-accent" />
                <h3 className="text-base font-semibold text-foreground">Active Loan Details</h3>
              </div>

              {app && app.status === "approved" ? (
                <div className="flex flex-col gap-6">
                  <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                    <div className="flex flex-col gap-1 rounded-lg bg-secondary p-3">
                      <span className="text-xs text-muted-foreground">Loan Amount</span>
                      <span className="text-lg font-bold text-foreground">${app.amount.toLocaleString()}</span>
                    </div>
                    <div className="flex flex-col gap-1 rounded-lg bg-secondary p-3">
                      <span className="text-xs text-muted-foreground">Monthly Payment</span>
                      <span className="text-lg font-bold text-foreground">${monthlyPayment.toLocaleString()}</span>
                    </div>
                    <div className="flex flex-col gap-1 rounded-lg bg-secondary p-3">
                      <span className="text-xs text-muted-foreground">Remaining Balance</span>
                      <span className="text-lg font-bold text-foreground">${remaining.toLocaleString()}</span>
                    </div>
                    <div className="flex flex-col gap-1 rounded-lg bg-secondary p-3">
                      <span className="text-xs text-muted-foreground">Term</span>
                      <span className="text-lg font-bold text-foreground">{app.term} years</span>
                    </div>
                  </div>

                  {/* Payment Progress */}
                  <div className="flex flex-col gap-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium text-foreground">Payment Progress</span>
                      <span className="text-sm text-muted-foreground">{paymentProgress}% paid</span>
                    </div>
                    <Progress value={paymentProgress} className="h-3" />
                    <div className="flex items-center justify-between text-xs text-muted-foreground">
                      <span>${totalPaid.toLocaleString()} paid</span>
                      <span>${remaining.toLocaleString()} remaining</span>
                    </div>
                  </div>

                  {/* Loan metadata */}
                  <div className="grid gap-3 sm:grid-cols-2">
                    <div className="flex items-center gap-3 text-sm">
                      <Calendar className="h-4 w-4 text-muted-foreground" />
                      <span className="text-muted-foreground">Start Date:</span>
                      <span className="text-foreground">{app.date}</span>
                    </div>
                    <div className="flex items-center gap-3 text-sm">
                      <FileText className="h-4 w-4 text-muted-foreground" />
                      <span className="text-muted-foreground">Purpose:</span>
                      <span className="text-foreground">{app.purpose}</span>
                    </div>
                    <div className="flex items-center gap-3 text-sm">
                      <DollarSign className="h-4 w-4 text-muted-foreground" />
                      <span className="text-muted-foreground">Interest Rate:</span>
                      <span className="text-foreground">6.5% APR</span>
                    </div>
                    <div className="flex items-center gap-3 text-sm">
                      <CreditCard className="h-4 w-4 text-muted-foreground" />
                      <span className="text-muted-foreground">Loan ID:</span>
                      <span className="font-mono text-xs text-foreground">{app.id}</span>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="flex flex-col items-center gap-3 py-8">
                  <CreditCard className="h-10 w-10 text-muted-foreground" />
                  <p className="text-sm text-muted-foreground">
                    {app ? "Your most recent application is still being processed." : "No active loan on file."}
                  </p>
                </div>
              )}
            </div>
          </motion.div>
        </div>

        {/* Document Upload */}
        <motion.div initial="hidden" animate="visible" variants={fadeIn} custom={2}>
          <div className="rounded-xl border border-border bg-card p-6">
            <div className="mb-4 flex items-center gap-2">
              <Upload className="h-5 w-5 text-accent" />
              <h3 className="text-base font-semibold text-foreground">Documents</h3>
            </div>

            {/* Existing documents */}
            <div className="mb-6 flex flex-col gap-2">
              {[...mockDocuments, ...uploadedFiles.map((name) => ({ name, status: "uploaded", date: new Date().toISOString().split("T")[0] }))].map((doc, i) => (
                <div key={`${doc.name}-${i}`} className="flex items-center justify-between rounded-lg border border-border px-4 py-3">
                  <div className="flex items-center gap-3">
                    <FileText className="h-4 w-4 text-muted-foreground" />
                    <div>
                      <p className="text-sm font-medium text-foreground">{doc.name}</p>
                      <p className="text-xs text-muted-foreground">Uploaded {doc.date}</p>
                    </div>
                  </div>
                  <CheckCircle2 className="h-4 w-4 text-success" />
                </div>
              ))}
            </div>

            {/* Upload area */}
            <div
              onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleDrop}
              className={cn(
                "flex flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed p-8 transition-colors",
                dragging ? "border-accent bg-accent/5" : "border-border"
              )}
            >
              <Upload className="h-8 w-8 text-muted-foreground" />
              <div className="text-center">
                <p className="text-sm font-medium text-foreground">Drop files here to upload</p>
                <p className="text-xs text-muted-foreground mt-1">or click to browse</p>
              </div>
              <label>
                <input
                  type="file"
                  multiple
                  className="hidden"
                  onChange={handleFileSelect}
                />
                <span className="cursor-pointer rounded-lg bg-accent px-4 py-2 text-xs font-medium text-accent-foreground hover:bg-accent/90 transition-colors">
                  Browse Files
                </span>
              </label>
            </div>
          </div>
        </motion.div>
      </div>
    </AuthGate>
  )
}
