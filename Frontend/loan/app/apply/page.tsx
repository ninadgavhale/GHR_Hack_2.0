// "use client"

// import React, { useState } from "react"
// import { useRouter } from "next/navigation"
// import { motion, AnimatePresence } from "motion/react"
// import {
//   ChevronRight,
//   ChevronLeft,
//   CheckCircle2,
//   DollarSign,
//   Briefcase,
//   CreditCard,
//   FileText,
//   Info,
//   Loader2,
// } from "lucide-react"
// import { useAuth } from "@/context/auth-context"
// import { useLoanData } from "@/context/loan-data-context"
// import { Button } from "@/components/ui/button"
// import { Input } from "@/components/ui/input"
// import { Label } from "@/components/ui/label"
// import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
// import { cn } from "@/lib/utils"

// const steps = [
//   { label: "Loan Details", icon: DollarSign },
//   { label: "Employment", icon: Briefcase },
//   { label: "Financial", icon: CreditCard },
//   { label: "Review", icon: FileText },
// ]

// function FieldWithTooltip({
//   label,
//   tooltip,
//   children,
// }: {
//   label: string
//   tooltip: string
//   children: React.ReactNode
// }) {
//   return (
//     <div className="flex flex-col gap-1.5">
//       <div className="flex items-center gap-1.5">
//         <Label className="text-sm font-medium text-foreground">{label}</Label>
//         <Tooltip>
//           <TooltipTrigger asChild>
//             <button type="button" className="text-muted-foreground hover:text-foreground">
//               <Info className="h-3.5 w-3.5" />
//               <span className="sr-only">Info about {label}</span>
//             </button>
//           </TooltipTrigger>
//           <TooltipContent side="top" className="max-w-xs">
//             {tooltip}
//           </TooltipContent>
//         </Tooltip>
//       </div>
//       {children}
//     </div>
//   )
// }

// interface FormData {
//   amount: string
//   term: string
//   purpose: string
//   employmentType: string
//   employmentYears: string
//   employer: string
//   income: string
//   creditScore: string
//   dti: string
//   downPayment: string
//   monthlyDebts: string
// }

// const initialForm: FormData = {
//   amount: "",
//   term: "30",
//   purpose: "Home Purchase",
//   employmentType: "Full-time",
//   employmentYears: "",
//   employer: "",
//   income: "",
//   creditScore: "",
//   dti: "",
//   downPayment: "",
//   monthlyDebts: "",
// }

// function AuthGate({ children }: { children: React.ReactNode }) {
//   const { isAuthenticated } = useAuth()
//   const router = useRouter()

//   if (!isAuthenticated) {
//     return (
//       <div className="flex flex-1 flex-col items-center justify-center gap-4 p-8">
//         <p className="text-muted-foreground">Please sign in to apply for a loan.</p>
//         <Button onClick={() => router.push("/login")} className="bg-accent text-accent-foreground hover:bg-accent/90">
//           Go to Login
//         </Button>
//       </div>
//     )
//   }

//   return <>{children}</>
// }

// export default function ApplyPage() {
//   const [step, setStep] = useState(0)
//   const [form, setForm] = useState<FormData>(initialForm)
//   const [submitting, setSubmitting] = useState(false)
//   const [submitted, setSubmitted] = useState(false)
//   const { addApplication } = useLoanData()
//   const router = useRouter()

//   const update = (key: keyof FormData, val: string) => setForm((p) => ({ ...p, [key]: val }))

//   const canNext = () => {
//     if (step === 0) return !!form.amount && !!form.purpose
//     if (step === 1) return !!form.employmentYears && !!form.income
//     if (step === 2) return !!form.creditScore && !!form.downPayment
//     return true
//   }

//   const handleSubmit = async () => {
//     setSubmitting(true)
//     await new Promise((r) => setTimeout(r, 1500))
//     const dti = form.dti ? parseInt(form.dti) : Math.round((parseInt(form.monthlyDebts || "0") * 12 / parseInt(form.income)) * 100)
//     addApplication({
//       amount: parseInt(form.amount),
//       term: parseInt(form.term),
//       purpose: form.purpose,
//       income: parseInt(form.income),
//       creditScore: parseInt(form.creditScore),
//       dti: isNaN(dti) ? 25 : dti,
//       employment: `${form.employmentType} (${form.employmentYears} years)`,
//       downPayment: parseInt(form.downPayment || "0"),
//     })
//     setSubmitting(false)
//     setSubmitted(true)
//   }

//   if (submitted) {
//     return (
//       <AuthGate>
//         <div className="flex flex-1 flex-col items-center justify-center gap-4 p-8">
//           <motion.div
//             initial={{ scale: 0 }}
//             animate={{ scale: 1 }}
//             transition={{ type: "spring", stiffness: 200, damping: 15 }}
//           >
//             <CheckCircle2 className="h-16 w-16 text-success" />
//           </motion.div>
//           <h2 className="text-xl font-bold text-foreground">Application Submitted</h2>
//           <p className="text-sm text-muted-foreground text-center max-w-md">
//             Your loan application has been processed by our AI system. View the results on your dashboard.
//           </p>
//           <Button onClick={() => router.push("/dashboard")} className="bg-accent text-accent-foreground hover:bg-accent/90 gap-2">
//             View Dashboard <ChevronRight className="h-4 w-4" />
//           </Button>
//         </div>
//       </AuthGate>
//     )
//   }

//   return (
//     <AuthGate>
//       <div className="mx-auto max-w-2xl p-4 md:p-6 lg:p-8">
//         <div className="mb-8">
//           <h1 className="text-2xl font-bold text-foreground">Loan Application</h1>
//           <p className="mt-1 text-sm text-muted-foreground">
//             Complete the form below. Hover the info icons to see how each field affects your AI score.
//           </p>
//         </div>

//         {/* Step indicator */}
//         <div className="mb-8 flex items-center justify-between">
//           {steps.map((s, i) => (
//             <React.Fragment key={s.label}>
//               <div className="flex flex-col items-center gap-1.5">
//                 <div
//                   className={cn(
//                     "flex h-9 w-9 items-center justify-center rounded-full border-2 transition-colors",
//                     i <= step
//                       ? "border-accent bg-accent text-accent-foreground"
//                       : "border-border bg-card text-muted-foreground"
//                   )}
//                 >
//                   {i < step ? (
//                     <CheckCircle2 className="h-4 w-4" />
//                   ) : (
//                     <s.icon className="h-4 w-4" />
//                   )}
//                 </div>
//                 <span
//                   className={cn(
//                     "hidden text-xs font-medium sm:block",
//                     i <= step ? "text-foreground" : "text-muted-foreground"
//                   )}
//                 >
//                   {s.label}
//                 </span>
//               </div>
//               {i < steps.length - 1 && (
//                 <div className={cn("h-0.5 flex-1 mx-2 rounded-full", i < step ? "bg-accent" : "bg-border")} />
//               )}
//             </React.Fragment>
//           ))}
//         </div>

//         {/* Form steps */}
//         <div className="rounded-xl border border-border bg-card p-6">
//           <AnimatePresence mode="wait">
//             <motion.div
//               key={step}
//               initial={{ opacity: 0, x: 20 }}
//               animate={{ opacity: 1, x: 0 }}
//               exit={{ opacity: 0, x: -20 }}
//               transition={{ duration: 0.25 }}
//               className="flex flex-col gap-5"
//             >
//               {step === 0 && (
//                 <>
//                   <FieldWithTooltip label="Loan Amount ($)" tooltip="Higher loan amounts relative to income reduce approval odds. Keep below 5x annual income for best results.">
//                     <Input
//                       type="number"
//                       placeholder="e.g. 250000"
//                       value={form.amount}
//                       onChange={(e) => update("amount", e.target.value)}
//                     />
//                   </FieldWithTooltip>
//                   <FieldWithTooltip label="Loan Term (years)" tooltip="Longer terms lower monthly payments but increase total interest. 15-30 years is standard for mortgages.">
//                     <select
//                       className="flex h-9 w-full rounded-md border border-input bg-background px-3 py-1 text-sm text-foreground shadow-sm transition-colors placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring"
//                       value={form.term}
//                       onChange={(e) => update("term", e.target.value)}
//                     >
//                       <option value="3">3 years</option>
//                       <option value="5">5 years</option>
//                       <option value="10">10 years</option>
//                       <option value="15">15 years</option>
//                       <option value="20">20 years</option>
//                       <option value="30">30 years</option>
//                     </select>
//                   </FieldWithTooltip>
//                   <FieldWithTooltip label="Loan Purpose" tooltip="Purpose affects risk assessment. Home purchases are viewed more favorably than unsecured personal loans.">
//                     <select
//                       className="flex h-9 w-full rounded-md border border-input bg-background px-3 py-1 text-sm text-foreground shadow-sm transition-colors placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring"
//                       value={form.purpose}
//                       onChange={(e) => update("purpose", e.target.value)}
//                     >
//                       <option>Home Purchase</option>
//                       <option>Home Refinance</option>
//                       <option>Auto Loan</option>
//                       <option>Personal Loan</option>
//                       <option>Business Loan</option>
//                       <option>Education</option>
//                     </select>
//                   </FieldWithTooltip>
//                 </>
//               )}

//               {step === 1 && (
//                 <>
//                   <FieldWithTooltip label="Employment Type" tooltip="Full-time employment with 2+ years at the same employer is weighted positively by the AI model.">
//                     <select
//                       className="flex h-9 w-full rounded-md border border-input bg-background px-3 py-1 text-sm text-foreground shadow-sm transition-colors placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring"
//                       value={form.employmentType}
//                       onChange={(e) => update("employmentType", e.target.value)}
//                     >
//                       <option>Full-time</option>
//                       <option>Part-time</option>
//                       <option>Self-employed</option>
//                       <option>Contract</option>
//                       <option>Retired</option>
//                     </select>
//                   </FieldWithTooltip>
//                   <FieldWithTooltip label="Years at Current Employer" tooltip="Longer tenure signals stability. 3+ years has a strong positive impact on approval.">
//                     <Input
//                       type="number"
//                       placeholder="e.g. 5"
//                       value={form.employmentYears}
//                       onChange={(e) => update("employmentYears", e.target.value)}
//                     />
//                   </FieldWithTooltip>
//                   <FieldWithTooltip label="Annual Income ($)" tooltip="Income is the second-highest weighted factor. Higher income relative to the loan amount significantly improves your score.">
//                     <Input
//                       type="number"
//                       placeholder="e.g. 95000"
//                       value={form.income}
//                       onChange={(e) => update("income", e.target.value)}
//                     />
//                   </FieldWithTooltip>
//                   <FieldWithTooltip label="Employer Name (optional)" tooltip="Used for verification purposes only. Does not impact AI scoring.">
//                     <Input
//                       type="text"
//                       placeholder="e.g. Acme Corp"
//                       value={form.employer}
//                       onChange={(e) => update("employer", e.target.value)}
//                     />
//                   </FieldWithTooltip>
//                 </>
//               )}

//               {step === 2 && (
//                 <>
//                   <FieldWithTooltip label="Credit Score" tooltip="The highest-weighted factor. 750+ is excellent, 700+ is good. Scores below 650 significantly reduce approval odds.">
//                     <Input
//                       type="number"
//                       min="300"
//                       max="850"
//                       placeholder="e.g. 750"
//                       value={form.creditScore}
//                       onChange={(e) => update("creditScore", e.target.value)}
//                     />
//                   </FieldWithTooltip>
//                   <FieldWithTooltip label="Debt-to-Income Ratio (%)" tooltip="DTI below 36% is ideal. Above 43% is a strong negative signal. Lower is better.">
//                     <Input
//                       type="number"
//                       min="0"
//                       max="100"
//                       placeholder="e.g. 28"
//                       value={form.dti}
//                       onChange={(e) => update("dti", e.target.value)}
//                     />
//                   </FieldWithTooltip>
//                   <FieldWithTooltip label="Down Payment ($)" tooltip="A larger down payment reduces lender risk. 20%+ of the loan amount is ideal for best scoring.">
//                     <Input
//                       type="number"
//                       placeholder="e.g. 50000"
//                       value={form.downPayment}
//                       onChange={(e) => update("downPayment", e.target.value)}
//                     />
//                   </FieldWithTooltip>
//                   <FieldWithTooltip label="Monthly Debt Payments ($)" tooltip="Total monthly payments on existing debts. Used to verify DTI if not provided.">
//                     <Input
//                       type="number"
//                       placeholder="e.g. 1200"
//                       value={form.monthlyDebts}
//                       onChange={(e) => update("monthlyDebts", e.target.value)}
//                     />
//                   </FieldWithTooltip>
//                 </>
//               )}

//               {step === 3 && (
//                 <div className="flex flex-col gap-4">
//                   <h3 className="text-base font-semibold text-foreground">Review Your Application</h3>
//                   <div className="grid gap-3 sm:grid-cols-2">
//                     {[
//                       { label: "Loan Amount", value: `$${parseInt(form.amount || "0").toLocaleString()}` },
//                       { label: "Term", value: `${form.term} years` },
//                       { label: "Purpose", value: form.purpose },
//                       { label: "Employment", value: `${form.employmentType} (${form.employmentYears} yrs)` },
//                       { label: "Annual Income", value: `$${parseInt(form.income || "0").toLocaleString()}` },
//                       { label: "Credit Score", value: form.creditScore },
//                       { label: "DTI Ratio", value: form.dti ? `${form.dti}%` : "Auto-calculated" },
//                       { label: "Down Payment", value: `$${parseInt(form.downPayment || "0").toLocaleString()}` },
//                     ].map((item) => (
//                       <div key={item.label} className="flex flex-col gap-0.5 rounded-lg bg-secondary p-3">
//                         <span className="text-xs text-muted-foreground">{item.label}</span>
//                         <span className="text-sm font-semibold text-foreground">{item.value}</span>
//                       </div>
//                     ))}
//                   </div>
//                 </div>
//               )}
//             </motion.div>
//           </AnimatePresence>

//           {/* Navigation buttons */}
//           <div className="mt-8 flex items-center justify-between">
//             <Button
//               variant="outline"
//               onClick={() => setStep((s) => s - 1)}
//               disabled={step === 0}
//               className="gap-1.5"
//             >
//               <ChevronLeft className="h-4 w-4" /> Back
//             </Button>
//             {step < 3 ? (
//               <Button
//                 onClick={() => setStep((s) => s + 1)}
//                 disabled={!canNext()}
//                 className="bg-accent text-accent-foreground hover:bg-accent/90 gap-1.5"
//               >
//                 Next <ChevronRight className="h-4 w-4" />
//               </Button>
//             ) : (
//               <Button
//                 onClick={handleSubmit}
//                 disabled={submitting}
//                 className="bg-accent text-accent-foreground hover:bg-accent/90 gap-1.5"
//               >
//                 {submitting ? <Loader2 className="mr-1 h-4 w-4 animate-spin" /> : null}
//                 {submitting ? "Processing..." : "Submit Application"}
//               </Button>
//             )}
//           </div>
//         </div>
//       </div>
//     </AuthGate>
//   )
// }









"use client"

import React, { useState } from "react"
import { useRouter } from "next/navigation"
import { motion, AnimatePresence } from "motion/react"
import {
  ChevronRight,
  ChevronLeft,
  CheckCircle2,
  DollarSign,
  Briefcase,
  CreditCard,
  FileText,
  Info,
  Loader2,
} from "lucide-react"
import { useAuth } from "@/context/auth-context"
import { useLoanData } from "@/context/loan-data-context"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"

const steps = [
  { label: "Loan Details", icon: DollarSign },
  { label: "Employment", icon: Briefcase },
  { label: "Financial", icon: CreditCard },
  { label: "Review", icon: FileText },
]

function FieldWithTooltip({
  label,
  tooltip,
  children,
}: {
  label: string
  tooltip: string
  children: React.ReactNode
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center gap-1.5">
        <Label className="text-sm font-medium text-foreground">{label}</Label>
        <Tooltip>
          <TooltipTrigger asChild>
            <button type="button" className="text-muted-foreground hover:text-foreground">
              <Info className="h-3.5 w-3.5" />
              <span className="sr-only">Info about {label}</span>
            </button>
          </TooltipTrigger>
          <TooltipContent side="top" className="max-w-xs">
            {tooltip}
          </TooltipContent>
        </Tooltip>
      </div>
      {children}
    </div>
  )
}

interface FormData {

  age : string
  maritalStatus : string
  gender : string 
  educationLevel : string 

  amount: string
  term: string
  purpose: string
  employmentType: string
  employmentYears: string
  employer: string
  income: string
  creditScore: string
  dti: string
  downPayment: string
  monthlyDebts: string
}

const initialForm: FormData = {
  age: "",
  maritalStatus: "",
  gender: "",
  educationLevel: "",

  amount: "",
  term: "30",
  purpose: "Home Purchase",
  employmentType: "Full-time",
  employmentYears: "",
  employer: "",
  income: "",
  creditScore: "",
  dti: "",
  downPayment: "",
  monthlyDebts: "",
}

function AuthGate({ children }: { children: React.ReactNode }) {
  const { isAuthenticated } = useAuth()
  const router = useRouter()

  if (!isAuthenticated) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-4 p-8">
        <p className="text-muted-foreground">Please sign in to apply for a loan.</p>
        <Button onClick={() => router.push("/login")} className="bg-accent text-accent-foreground hover:bg-accent/90">
          Go to Login
        </Button>
      </div>
    )
  }

  return <>{children}</>
}

export default function ApplyPage() {
  const [step, setStep] = useState(0)
  const [form, setForm] = useState<FormData>(initialForm)
  const [submitting, setSubmitting] = useState(false)
  const [submitted, setSubmitted] = useState(false)
  const { addApplication } = useLoanData()
  const router = useRouter()

  const update = (key: keyof FormData, val: string) => setForm((p) => ({ ...p, [key]: val }))

  const canNext = () => {
    if (step === 0) return !!form.amount && !!form.purpose
    if (step === 1) return !!form.employmentYears && !!form.income
    if (step === 2) return !!form.creditScore && !!form.downPayment
    return true
  }

  const handleSubmit = async () => {
    setSubmitting(true)
    await new Promise((r) => setTimeout(r, 1500))
    const dti = form.dti ? parseInt(form.dti) : Math.round((parseInt(form.monthlyDebts || "0") * 12 / parseInt(form.income)) * 100)
    addApplication({

      age: parseInt(form.age),
maritalStatus: form.maritalStatus,
gender: form.gender,
educationLevel: form.educationLevel,

      amount: parseInt(form.amount),
      term: parseInt(form.term),
      purpose: form.purpose,
      income: parseInt(form.income),
      creditScore: parseInt(form.creditScore),
      dti: isNaN(dti) ? 25 : dti,
      employment: `${form.employmentType} (${form.employmentYears} years)`,
      downPayment: parseInt(form.downPayment || "0"),
    })
    setSubmitting(false)
    setSubmitted(true)
  }

  if (submitted) {
    return (
      <AuthGate>
        <div className="flex flex-1 flex-col items-center justify-center gap-4 p-8">
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ type: "spring", stiffness: 200, damping: 15 }}
          >
            <CheckCircle2 className="h-16 w-16 text-success" />
          </motion.div>
          <h2 className="text-xl font-bold text-foreground">Application Submitted</h2>
          <p className="text-sm text-muted-foreground text-center max-w-md">
            Your loan application has been processed by our AI system. View the results on your dashboard.
          </p>
          <Button onClick={() => router.push("/dashboard")} className="bg-accent text-accent-foreground hover:bg-accent/90 gap-2">
            View Dashboard <ChevronRight className="h-4 w-4" />
          </Button>
        </div>
      </AuthGate>
    )
  }

  return (
    <AuthGate>
      <div className="mx-auto max-w-2xl p-4 md:p-6 lg:p-8">
        <div className="mb-8">
          <h1 className="text-2xl font-bold text-foreground">Loan Application</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Complete the form below. Hover the info icons to see how each field affects your AI score.
          </p>
        </div>

        {/* Step indicator */}
        <div className="mb-8 flex items-center justify-between">
          {steps.map((s, i) => (
            <React.Fragment key={s.label}>
              <div className="flex flex-col items-center gap-1.5">
                <div
                  className={cn(
                    "flex h-9 w-9 items-center justify-center rounded-full border-2 transition-colors",
                    i <= step
                      ? "border-accent bg-accent text-accent-foreground"
                      : "border-border bg-card text-muted-foreground"
                  )}
                >
                  {i < step ? (
                    <CheckCircle2 className="h-4 w-4" />
                  ) : (
                    <s.icon className="h-4 w-4" />
                  )}
                </div>
                <span
                  className={cn(
                    "hidden text-xs font-medium sm:block",
                    i <= step ? "text-foreground" : "text-muted-foreground"
                  )}
                >
                  {s.label}
                </span>
              </div>
              {i < steps.length - 1 && (
                <div className={cn("h-0.5 flex-1 mx-2 rounded-full", i < step ? "bg-accent" : "bg-border")} />
              )}
            </React.Fragment>
          ))}
        </div>

        {/* Form steps */}
        <div className="rounded-xl border border-border bg-card p-6">
          <AnimatePresence mode="wait">
            <motion.div
              key={step}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.25 }}
              className="flex flex-col gap-5"
            >
              {step === 0 && (
                <>
                  
                  <FieldWithTooltip label="Age" tooltip="Applicant must be 18 years or older. Age impacts long-term repayment risk.">
  <Input
    type="number"
    min="18"
    placeholder="e.g. 25"
    value={form.age}
    onChange={(e) => update("age", e.target.value)}
  />
</FieldWithTooltip>

<FieldWithTooltip label="Marital Status" tooltip="Used for demographic risk modeling.">
  <select
    className="flex h-9 w-full rounded-md border border-input bg-background px-3 py-1 text-sm text-foreground shadow-sm focus:outline-none focus:ring-1 focus:ring-ring"
    value={form.maritalStatus}
    onChange={(e) => update("maritalStatus", e.target.value)}
  >
    <option value="">Select</option>
    <option>Married</option>
    <option>Unmarried</option>
  </select>
</FieldWithTooltip>

<FieldWithTooltip label="Gender" tooltip="Collected for reporting purposes only.">
  <select
    className="flex h-9 w-full rounded-md border border-input bg-background px-3 py-1 text-sm text-foreground shadow-sm focus:outline-none focus:ring-1 focus:ring-ring"
    value={form.gender}
    onChange={(e) => update("gender", e.target.value)}
  >
    <option value="">Select</option>
    <option>M</option>
    <option>F</option>
    <option>O</option>
  </select>
</FieldWithTooltip>

<FieldWithTooltip label="Education Level" tooltip="Higher education levels positively impact AI scoring.">
  <select
    className="flex h-9 w-full rounded-md border border-input bg-background px-3 py-1 text-sm text-foreground shadow-sm focus:outline-none focus:ring-1 focus:ring-ring"
    value={form.educationLevel}
    onChange={(e) => update("educationLevel", e.target.value)}
  >
    <option value="">Select</option>
    <option>10th</option>
    <option>12th</option>
    <option>Graduate</option>
    <option>Post Graduate</option>
  </select>
</FieldWithTooltip>



                  <FieldWithTooltip label="Loan Amount (₹)" tooltip="Higher loan amounts relative to income reduce approval odds. Keep below 5x annual income for best results.">
                    <Input
                      type="number"
                      placeholder="e.g. 250000"
                      value={form.amount}
                      onChange={(e) => update("amount", e.target.value)}
                    />
                  </FieldWithTooltip>
                  <FieldWithTooltip label="Loan Term (years)" tooltip="Longer terms lower monthly payments but increase total interest. 15-30 years is standard for mortgages.">
                    <select
                      className="flex h-9 w-full rounded-md border border-input bg-background px-3 py-1 text-sm text-foreground shadow-sm transition-colors placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring"
                      value={form.term}
                      onChange={(e) => update("term", e.target.value)}
                    >
                      <option value="3">3 years</option>
                      <option value="5">5 years</option>
                      <option value="10">10 years</option>
                      <option value="15">15 years</option>
                      <option value="20">20 years</option>
                      <option value="30">30 years</option>
                    </select>
                  </FieldWithTooltip>
                  <FieldWithTooltip label="Loan Purpose" tooltip="Purpose affects risk assessment. Home purchases are viewed more favorably than unsecured personal loans.">
                    <select
                      className="flex h-9 w-full rounded-md border border-input bg-background px-3 py-1 text-sm text-foreground shadow-sm transition-colors placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring"
                      value={form.purpose}
                      onChange={(e) => update("purpose", e.target.value)}
                    >
                      <option>Home Purchase</option>
                      <option>Home Refinance</option>
                      <option>Auto Loan</option>
                      <option>Personal Loan</option>
                      <option>Business Loan</option>
                      <option>Education</option>
                    </select>
                  </FieldWithTooltip>
                </>
              )}

              {step === 1 && (
                <>
                  <FieldWithTooltip label="Employment Type" tooltip="Full-time employment with 2+ years at the same employer is weighted positively by the AI model.">
                    <select
                      className="flex h-9 w-full rounded-md border border-input bg-background px-3 py-1 text-sm text-foreground shadow-sm transition-colors placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring"
                      value={form.employmentType}
                      onChange={(e) => update("employmentType", e.target.value)}
                    >
                      <option>Full-time</option>
                      <option>Part-time</option>
                      <option>Self-employed</option>
                      <option>Contract</option>
                      <option>Retired</option>
                    </select>
                  </FieldWithTooltip>
                  <FieldWithTooltip label="Years at Current Employer" tooltip="Longer tenure signals stability. 3+ years has a strong positive impact on approval.">
                    <Input
                      type="number"
                      placeholder="e.g. 5"
                      value={form.employmentYears}
                      onChange={(e) => update("employmentYears", e.target.value)}
                    />
                  </FieldWithTooltip>
                  <FieldWithTooltip label="Annual Income (₹)" tooltip="Income is the second-highest weighted factor. Higher income relative to the loan amount significantly improves your score.">
                    <Input
                      type="number"
                      placeholder="e.g. 95000"
                      value={form.income}
                      onChange={(e) => update("income", e.target.value)}
                    />
                  </FieldWithTooltip>
                  <FieldWithTooltip label="Employer Name (optional)" tooltip="Used for verification purposes only. Does not impact AI scoring.">
                    <Input
                      type="text"
                      placeholder="e.g. Acme Corp"
                      value={form.employer}
                      onChange={(e) => update("employer", e.target.value)}
                    />
                  </FieldWithTooltip>
                </>
              )}

              {step === 2 && (
                <>
                  <FieldWithTooltip label="Credit Score" tooltip="The highest-weighted factor. 750+ is excellent, 700+ is good. Scores below 650 significantly reduce approval odds.">
                    <Input
                      type="number"
                      min="300"
                      max="850"
                      placeholder="e.g. 750"
                      value={form.creditScore}
                      onChange={(e) => update("creditScore", e.target.value)}
                    />
                  </FieldWithTooltip>
                  <FieldWithTooltip label="Debt-to-Income Ratio (%)" tooltip="DTI below 36% is ideal. Above 43% is a strong negative signal. Lower is better.">
                    <Input
                      type="number"
                      min="0"
                      max="100"
                      placeholder="e.g. 28"
                      value={form.dti}
                      onChange={(e) => update("dti", e.target.value)}
                    />
                  </FieldWithTooltip>
                  <FieldWithTooltip label="Down Payment (₹)" tooltip="A larger down payment reduces lender risk. 20%+ of the loan amount is ideal for best scoring.">
                    <Input
                      type="number"
                      placeholder="e.g. 50000"
                      value={form.downPayment}
                      onChange={(e) => update("downPayment", e.target.value)}
                    />
                  </FieldWithTooltip>
                  <FieldWithTooltip label="Monthly Debt Payments (₹)" tooltip="Total monthly payments on existing debts. Used to verify DTI if not provided.">
                    <Input
                      type="number"
                      placeholder="e.g. 1200"
                      value={form.monthlyDebts}
                      onChange={(e) => update("monthlyDebts", e.target.value)}
                    />
                  </FieldWithTooltip>
                </>
              )}

              {step === 3 && (
                <div className="flex flex-col gap-4">
                  <h3 className="text-base font-semibold text-foreground">Review Your Application</h3>
                  <div className="grid gap-3 sm:grid-cols-2">
                    {[
                      {
                        
                        label: "Age", value: form.age },
{ label: "Marital Status", value: form.maritalStatus },
{ label: "Gender", value: form.gender },
{ label: "Education Level", value: form.educationLevel },


                      { label: "Loan Amount", value: `₹${parseInt(form.amount || "0").toLocaleString()}` },
                      { label: "Term", value: `${form.term} years` },
                      { label: "Purpose", value: form.purpose },
                      { label: "Employment", value: `${form.employmentType} (${form.employmentYears} yrs)` },
                      { label: "Annual Income", value: `₹${parseInt(form.income || "0").toLocaleString()}` },
                      { label: "Credit Score", value: form.creditScore },
                      { label: "DTI Ratio", value: form.dti ? `${form.dti}%` : "Auto-calculated" },
                      { label: "Down Payment", value: `₹${parseInt(form.downPayment || "0").toLocaleString()}` },
                    ].map((item) => (
                      <div key={item.label} className="flex flex-col gap-0.5 rounded-lg bg-secondary p-3">
                        <span className="text-xs text-muted-foreground">{item.label}</span>
                        <span className="text-sm font-semibold text-foreground">{item.value}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </motion.div>
          </AnimatePresence>

          {/* Navigation buttons */}
          <div className="mt-8 flex items-center justify-between">
            <Button
              variant="outline"
              onClick={() => setStep((s) => s - 1)}
              disabled={step === 0}
              className="gap-1.5"
            >
              <ChevronLeft className="h-4 w-4" /> Back
            </Button>
            {step < 3 ? (
              <Button
                onClick={() => setStep((s) => s + 1)}
                disabled={!canNext()}
                className="bg-accent text-accent-foreground hover:bg-accent/90 gap-1.5"
              >
                Next <ChevronRight className="h-4 w-4" />
              </Button>
            ) : (
              <Button
                onClick={handleSubmit}
                disabled={submitting}
                className="bg-accent text-accent-foreground hover:bg-accent/90 gap-1.5"
              >
                {submitting ? <Loader2 className="mr-1 h-4 w-4 animate-spin" /> : null}
                {submitting ? "Processing..." : "Submit Application"}
              </Button>
            )}
          </div>
        </div>
      </div>
    </AuthGate>
  )
}