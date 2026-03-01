"use client"

import React, { createContext, useContext, useState, useCallback, type ReactNode } from "react"

export type LoanStatus = "approved" | "denied" | "in_review" | "pending"

export interface FeatureAttribution {
  name: string
  value: number
  impact: "positive" | "negative"
}

export interface LoanApplication {
  id: string
  date: string
  amount: number
  term: number
  purpose: string
  status: LoanStatus
  score: number
  confidence: number
  fairnessScore: number
  explanation: string
  features: FeatureAttribution[]
  income: number
  creditScore: number
  dti: number
  employment: string
  downPayment: number
}

export interface WhatIfState {
  income: number
  creditScore: number
  dti: number
  downPayment: number
  loanAmount: number
}

const defaultFeatures: FeatureAttribution[] = [
  { name: "Credit Score", value: 0.32, impact: "positive" },
  { name: "Income Level", value: 0.24, impact: "positive" },
  { name: "DTI Ratio", value: -0.18, impact: "negative" },
  { name: "Employment Yrs", value: 0.15, impact: "positive" },
  { name: "Down Payment", value: 0.12, impact: "positive" },
  { name: "Loan Amount", value: -0.08, impact: "negative" },
  { name: "Recent Inquiries", value: -0.05, impact: "negative" },
  { name: "Account Age", value: 0.09, impact: "positive" },
]

const sampleApplications: LoanApplication[] = [
  {
    id: "LA-2024-001",
    date: "2024-12-15",
    amount: 250000,
    term: 30,
    purpose: "Home Purchase",
    status: "approved",
    score: 82,
    confidence: 94,
    fairnessScore: 97,
    explanation: "Your application was approved based on strong credit history (750+), stable income exceeding 3x monthly payments, and a healthy down payment of 20%. The AI model found no adverse indicators in your financial profile.",
    features: defaultFeatures,
    income: 95000,
    creditScore: 762,
    dti: 28,
    employment: "Full-time (5 years)",
    downPayment: 50000,
  },
  {
    id: "LA-2024-002",
    date: "2024-11-03",
    amount: 35000,
    term: 5,
    purpose: "Auto Loan",
    status: "approved",
    score: 76,
    confidence: 89,
    fairnessScore: 95,
    explanation: "Auto loan approved. Your debt-to-income ratio and credit score indicate manageable risk. The model weighted employment stability and payment history positively.",
    features: [
      { name: "Credit Score", value: 0.28, impact: "positive" },
      { name: "Income Level", value: 0.22, impact: "positive" },
      { name: "DTI Ratio", value: -0.12, impact: "negative" },
      { name: "Employment Yrs", value: 0.20, impact: "positive" },
      { name: "Down Payment", value: 0.08, impact: "positive" },
      { name: "Loan Amount", value: -0.04, impact: "negative" },
    ],
    income: 72000,
    creditScore: 710,
    dti: 22,
    employment: "Full-time (3 years)",
    downPayment: 7000,
  },
  {
    id: "LA-2024-003",
    date: "2024-10-20",
    amount: 15000,
    term: 3,
    purpose: "Personal Loan",
    status: "denied",
    score: 38,
    confidence: 91,
    fairnessScore: 96,
    explanation: "Application declined due to elevated debt-to-income ratio (48%) and recent credit inquiries. The model suggests reducing existing debt before reapplying. This decision was not influenced by demographic factors.",
    features: [
      { name: "Credit Score", value: 0.10, impact: "positive" },
      { name: "Income Level", value: -0.05, impact: "negative" },
      { name: "DTI Ratio", value: -0.35, impact: "negative" },
      { name: "Employment Yrs", value: 0.08, impact: "positive" },
      { name: "Recent Inquiries", value: -0.22, impact: "negative" },
      { name: "Account Age", value: -0.10, impact: "negative" },
    ],
    income: 42000,
    creditScore: 620,
    dti: 48,
    employment: "Part-time (1 year)",
    downPayment: 1000,
  },
  {
    id: "LA-2025-001",
    date: "2025-01-10",
    amount: 180000,
    term: 15,
    purpose: "Home Refinance",
    status: "in_review",
    score: 65,
    confidence: 72,
    fairnessScore: 98,
    explanation: "Your refinance application is currently under review. Initial analysis shows a moderate approval likelihood. The AI model requires additional verification of recent income changes before making a final determination.",
    features: defaultFeatures.map((f) => ({ ...f, value: f.value * 0.85 })),
    income: 85000,
    creditScore: 695,
    dti: 34,
    employment: "Full-time (7 years)",
    downPayment: 0,
  },
]

interface LoanDataContextType {
  applications: LoanApplication[]
  currentApplication: LoanApplication | null
  whatIfState: WhatIfState
  whatIfScore: number
  whatIfFeatures: FeatureAttribution[]
  setCurrentApplication: (app: LoanApplication | null) => void
  addApplication: (app: Omit<LoanApplication, "id" | "date" | "status" | "score" | "confidence" | "fairnessScore" | "explanation" | "features">) => void
  updateWhatIf: (state: Partial<WhatIfState>) => void
}

const LoanDataContext = createContext<LoanDataContextType | undefined>(undefined)

function computeWhatIfScore(state: WhatIfState): number {
  const creditWeight = (state.creditScore - 300) / 550
  const incomeWeight = Math.min(state.income / 150000, 1)
  const dtiWeight = 1 - Math.min(state.dti / 60, 1)
  const dpWeight = Math.min(state.downPayment / (state.loanAmount * 0.25 || 1), 1)
  const raw = creditWeight * 0.35 + incomeWeight * 0.25 + dtiWeight * 0.25 + dpWeight * 0.15
  return Math.round(Math.max(0, Math.min(100, raw * 100)))
}

function computeWhatIfFeatures(state: WhatIfState): FeatureAttribution[] {
  const creditImpact = (state.creditScore - 650) / 200
  const incomeImpact = (state.income - 50000) / 100000
  const dtiImpact = -(state.dti - 30) / 40
  const dpImpact = state.loanAmount > 0 ? (state.downPayment / state.loanAmount - 0.1) : 0
  return [
    { name: "Credit Score", value: parseFloat((creditImpact * 0.35).toFixed(2)), impact: creditImpact >= 0 ? "positive" : "negative" },
    { name: "Income Level", value: parseFloat((incomeImpact * 0.25).toFixed(2)), impact: incomeImpact >= 0 ? "positive" : "negative" },
    { name: "DTI Ratio", value: parseFloat((dtiImpact * 0.25).toFixed(2)), impact: dtiImpact >= 0 ? "positive" : "negative" },
    { name: "Down Payment", value: parseFloat((dpImpact * 0.15).toFixed(2)), impact: dpImpact >= 0 ? "positive" : "negative" },
    { name: "Employment Yrs", value: 0.12, impact: "positive" },
    { name: "Account Age", value: 0.08, impact: "positive" },
  ]
}

export function LoanDataProvider({ children }: { children: ReactNode }) {
  const [applications, setApplications] = useState<LoanApplication[]>(sampleApplications)
  const [currentApplication, setCurrentApplication] = useState<LoanApplication | null>(sampleApplications[0])
  const [whatIfState, setWhatIfState] = useState<WhatIfState>({
    income: 95000,
    creditScore: 762,
    dti: 28,
    downPayment: 50000,
    loanAmount: 250000,
  })

  const whatIfScore = computeWhatIfScore(whatIfState)
  const whatIfFeatures = computeWhatIfFeatures(whatIfState)

  const addApplication = useCallback(
    (app: Omit<LoanApplication, "id" | "date" | "status" | "score" | "confidence" | "fairnessScore" | "explanation" | "features">) => {
      const score = computeWhatIfScore({
        income: app.income,
        creditScore: app.creditScore,
        dti: app.dti,
        downPayment: app.downPayment,
        loanAmount: app.amount,
      })
      const status: LoanStatus = score >= 70 ? "approved" : score >= 45 ? "in_review" : "denied"
      const newApp: LoanApplication = {
        ...app,
        id: `LA-${new Date().getFullYear()}-${String(applications.length + 1).padStart(3, "0")}`,
        date: new Date().toISOString().split("T")[0],
        status,
        score,
        confidence: 75 + Math.round(Math.random() * 20),
        fairnessScore: 90 + Math.round(Math.random() * 10),
        explanation:
          status === "approved"
            ? "Your application shows strong financial indicators. The AI model found positive signals in your credit history and income stability."
            : status === "in_review"
            ? "Your application is under review. Some factors need additional assessment before a final decision."
            : "Application not approved at this time. Key factors include elevated debt ratios. Consider reducing existing debt.",
        features: computeWhatIfFeatures({
          income: app.income,
          creditScore: app.creditScore,
          dti: app.dti,
          downPayment: app.downPayment,
          loanAmount: app.amount,
        }),
      }
      setApplications((prev) => [newApp, ...prev])
      setCurrentApplication(newApp)
    },
    [applications.length]
  )

  const updateWhatIf = useCallback((partial: Partial<WhatIfState>) => {
    setWhatIfState((prev) => ({ ...prev, ...partial }))
  }, [])

  return (
    <LoanDataContext.Provider
      value={{
        applications,
        currentApplication,
        whatIfState,
        whatIfScore,
        whatIfFeatures,
        setCurrentApplication,
        addApplication,
        updateWhatIf,
      }}
    >
      {children}
    </LoanDataContext.Provider>
  )
}

export function useLoanData() {
  const ctx = useContext(LoanDataContext)
  if (!ctx) throw new Error("useLoanData must be used within LoanDataProvider")
  return ctx
}
