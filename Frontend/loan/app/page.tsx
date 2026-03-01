"use client"

import React from "react"
import Link from "next/link"
import { motion } from "motion/react"
import {
  BrainCircuit,
  ShieldCheck,
  SlidersHorizontal,
  ArrowRight,
  Sparkles,
  TrendingUp,
  Scale,
  Eye,
  CheckCircle2,
} from "lucide-react"
import { Button } from "@/components/ui/button"

const fadeUp = {
  hidden: { opacity: 0, y: 24 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.1, duration: 0.5, ease: "easeOut" },
  }),
}

const features = [
  {
    icon: Eye,
    title: "Transparent AI",
    description: "Every decision is explained in plain English with feature-level attribution. No more black-box lending.",
  },
  {
    icon: ShieldCheck,
    title: "Bias-Free Decisions",
    description: "Continuous fairness monitoring ensures equitable outcomes across all demographics.",
  },
  {
    icon: SlidersHorizontal,
    title: "What-If Simulator",
    description: "Explore how changing your financial profile impacts your approval odds in real time.",
  },
]

const stats = [
  { value: "98%", label: "Fairness Score", icon: Scale },
  { value: "< 2s", label: "Decision Time", icon: Sparkles },
  { value: "94%", label: "Model Confidence", icon: TrendingUp },
]

const steps = [
  { step: "01", title: "Submit Application", description: "Provide your financial details through our secure multi-step form." },
  { step: "02", title: "AI Analysis", description: "Our explainable AI evaluates your profile with full transparency." },
  { step: "03", title: "Get Decision", description: "Receive an instant decision with detailed reasoning and next steps." },
]

export default function HomePage() {
  return (
    <div className="flex flex-col">
      {/* Hero Section */}
      <section className="relative overflow-hidden px-4 py-20 md:py-28 lg:py-32">
        <div className="pointer-events-none absolute inset-0 overflow-hidden">
          <div className="absolute -top-40 -right-40 h-80 w-80 rounded-full bg-accent/10 blur-[100px]" />
          <div className="absolute -bottom-40 -left-40 h-80 w-80 rounded-full bg-success/10 blur-[100px]" />
        </div>
        <div className="relative mx-auto max-w-5xl text-center">
          <motion.div
            initial="hidden"
            animate="visible"
            variants={fadeUp}
            custom={0}
            className="mb-4 inline-flex items-center gap-2 rounded-full border border-border bg-card px-4 py-1.5 text-xs font-medium text-muted-foreground"
          >
            <Sparkles className="h-3.5 w-3.5 text-accent" />
            Powered by Explainable AI
          </motion.div>
          <motion.h1
            initial="hidden"
            animate="visible"
            variants={fadeUp}
            custom={1}
            className="text-balance text-4xl font-bold tracking-tight text-foreground md:text-5xl lg:text-6xl"
          >
            Loan decisions you can
            <span className="text-accent"> understand</span> and
            <span className="text-accent"> trust</span>
          </motion.h1>
          <motion.p
            initial="hidden"
            animate="visible"
            variants={fadeUp}
            custom={2}
            className="mx-auto mt-6 max-w-2xl text-pretty text-base text-muted-foreground md:text-lg"
          >
            LoanAI uses explainable artificial intelligence to provide transparent,
            fair, and instant lending decisions. Every factor is visible. Every
            outcome is justified.
          </motion.p>
          <motion.div
            initial="hidden"
            animate="visible"
            variants={fadeUp}
            custom={3}
            className="mt-8 flex flex-wrap items-center justify-center gap-3"
          >
            <Link href="/apply">
              <Button size="lg" className="bg-accent text-accent-foreground hover:bg-accent/90 gap-2 px-6">
                Apply Now <ArrowRight className="h-4 w-4" />
              </Button>
            </Link>
            <Link href="/login">
              <Button size="lg" variant="outline" className="gap-2 px-6">
                Log In
              </Button>
            </Link>
          </motion.div>
        </div>
      </section>

      {/* Stats */}
      <section className="border-y border-border bg-card/50 px-4 py-12">
        <div className="mx-auto grid max-w-4xl grid-cols-1 gap-8 sm:grid-cols-3">
          {stats.map((stat, i) => (
            <motion.div
              key={stat.label}
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true }}
              variants={fadeUp}
              custom={i}
              className="flex flex-col items-center text-center"
            >
              <stat.icon className="mb-3 h-6 w-6 text-accent" />
              <p className="text-3xl font-bold text-foreground">{stat.value}</p>
              <p className="mt-1 text-sm text-muted-foreground">{stat.label}</p>
            </motion.div>
          ))}
        </div>
      </section>

      {/* Features */}
      <section className="px-4 py-20 md:py-24">
        <div className="mx-auto max-w-5xl">
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={fadeUp}
            custom={0}
            className="text-center"
          >
            <p className="text-sm font-semibold uppercase tracking-wider text-accent">Why LoanAI</p>
            <h2 className="mt-2 text-balance text-3xl font-bold text-foreground md:text-4xl">
              AI lending, done right
            </h2>
            <p className="mx-auto mt-4 max-w-2xl text-pretty text-muted-foreground">
              We combine state-of-the-art machine learning with rigorous fairness constraints and full explainability.
            </p>
          </motion.div>
          <div className="mt-14 grid gap-6 md:grid-cols-3">
            {features.map((feature, i) => (
              <motion.div
                key={feature.title}
                initial="hidden"
                whileInView="visible"
                viewport={{ once: true }}
                variants={fadeUp}
                custom={i + 1}
                className="group rounded-xl border border-border bg-card p-6 transition-colors hover:border-accent/30 hover:bg-accent/5"
              >
                <div className="mb-4 flex h-10 w-10 items-center justify-center rounded-lg bg-accent/10 text-accent transition-colors group-hover:bg-accent group-hover:text-accent-foreground">
                  <feature.icon className="h-5 w-5" />
                </div>
                <h3 className="text-base font-semibold text-foreground">{feature.title}</h3>
                <p className="mt-2 text-sm text-muted-foreground leading-relaxed">{feature.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section className="border-t border-border bg-card/50 px-4 py-20 md:py-24">
        <div className="mx-auto max-w-4xl">
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={fadeUp}
            custom={0}
            className="text-center"
          >
            <p className="text-sm font-semibold uppercase tracking-wider text-accent">Process</p>
            <h2 className="mt-2 text-balance text-3xl font-bold text-foreground md:text-4xl">
              Three steps to a decision
            </h2>
          </motion.div>
          <div className="mt-14 grid gap-8 md:grid-cols-3">
            {steps.map((s, i) => (
              <motion.div
                key={s.step}
                initial="hidden"
                whileInView="visible"
                viewport={{ once: true }}
                variants={fadeUp}
                custom={i + 1}
                className="relative flex flex-col items-center text-center"
              >
                <div className="flex h-12 w-12 items-center justify-center rounded-full bg-accent text-accent-foreground text-sm font-bold">
                  {s.step}
                </div>
                <h3 className="mt-4 text-base font-semibold text-foreground">{s.title}</h3>
                <p className="mt-2 text-sm text-muted-foreground leading-relaxed">{s.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="px-4 py-20 md:py-24">
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          variants={fadeUp}
          custom={0}
          className="mx-auto max-w-3xl rounded-2xl border border-accent/20 bg-accent/5 p-10 text-center md:p-14"
        >
          <BrainCircuit className="mx-auto mb-4 h-10 w-10 text-accent" />
          <h2 className="text-balance text-2xl font-bold text-foreground md:text-3xl">
            Ready to experience transparent lending?
          </h2>
          <p className="mx-auto mt-4 max-w-lg text-pretty text-muted-foreground">
            Join thousands of applicants who trust LoanAI for fair, explainable loan decisions.
          </p>
          <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
            <Link href="/apply">
              <Button size="lg" className="bg-accent text-accent-foreground hover:bg-accent/90 gap-2 px-6">
                Start Your Application <ArrowRight className="h-4 w-4" />
              </Button>
            </Link>
          </div>
          <div className="mt-6 flex items-center justify-center gap-4 text-xs text-muted-foreground">
            <span className="flex items-center gap-1"><CheckCircle2 className="h-3.5 w-3.5 text-success" /> No credit impact</span>
            <span className="flex items-center gap-1"><CheckCircle2 className="h-3.5 w-3.5 text-success" /> Instant results</span>
            <span className="flex items-center gap-1"><CheckCircle2 className="h-3.5 w-3.5 text-success" /> Full transparency</span>
          </div>
        </motion.div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border bg-card px-4 py-8">
        <div className="mx-auto flex max-w-5xl flex-col items-center gap-4 sm:flex-row sm:justify-between">
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <BrainCircuit className="h-4 w-4 text-accent" />
            <span className="font-semibold text-foreground">LoanAI</span>
            <span>Explainable Lending</span>
          </div>
          <p className="text-xs text-muted-foreground">
            &copy; {new Date().getFullYear()} LoanAI. All rights reserved. For demonstration purposes only.
          </p>
        </div>
      </footer>
    </div>
  )
}
