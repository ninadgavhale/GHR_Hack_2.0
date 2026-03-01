"use client"

import React from "react"
import { AuthProvider } from "@/context/auth-context"
import { LoanDataProvider } from "@/context/loan-data-context"
import { ThemeProvider } from "@/context/theme-context"
import { AppLayout } from "@/components/app-layout"

export function ClientShell({ children }: { children: React.ReactNode }) {
  return (
    <ThemeProvider>
      <AuthProvider>
        <LoanDataProvider>
          <AppLayout>{children}</AppLayout>
        </LoanDataProvider>
      </AuthProvider>
    </ThemeProvider>
  )
}
