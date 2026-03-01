"use client"

import React, { useState } from "react"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { motion, AnimatePresence } from "motion/react"
import {
  LayoutDashboard,
  FileText,
  History,
  User,
  LogOut,
  Menu,
  X,
  Sun,
  Moon,
  BrainCircuit,
  ChevronRight,
} from "lucide-react"
import { cn } from "@/lib/utils"
import { useAuth } from "@/context/auth-context"
import { useTheme } from "@/context/theme-context"
import { Button } from "@/components/ui/button"
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet"

const navItems = [
  { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { href: "/apply", label: "Apply", icon: FileText },
  { href: "/history", label: "History", icon: History },
  { href: "/profile", label: "Profile", icon: User },
]

function SidebarContent({ onNavigate }: { onNavigate?: () => void }) {
  const pathname = usePathname()
  const { logout } = useAuth()

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-2.5 px-4 py-5">
        <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-accent">
          <BrainCircuit className="h-5 w-5 text-accent-foreground" />
        </div>
        <div>
          <p className="text-sm font-semibold text-foreground leading-none">LoanAI</p>
          <p className="text-xs text-muted-foreground mt-0.5">Explainable AI</p>
        </div>
      </div>
      <nav className="flex-1 px-3 py-2">
        <ul className="flex flex-col gap-1">
          {navItems.map((item) => {
            const isActive = pathname === item.href
            return (
              <li key={item.href}>
                <Link
                  href={item.href}
                  onClick={onNavigate}
                  className={cn(
                    "flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors",
                    isActive
                      ? "bg-accent text-accent-foreground"
                      : "text-muted-foreground hover:bg-secondary hover:text-foreground"
                  )}
                >
                  <item.icon className="h-4 w-4 shrink-0" />
                  {item.label}
                  {isActive && <ChevronRight className="ml-auto h-3.5 w-3.5" />}
                </Link>
              </li>
            )
          })}
        </ul>
      </nav>
      <div className="border-t border-border px-3 py-3">
        <button
          onClick={() => {
            logout()
            onNavigate?.()
          }}
          className="flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium text-muted-foreground transition-colors hover:bg-destructive/10 hover:text-destructive"
        >
          <LogOut className="h-4 w-4" />
          Sign Out
        </button>
      </div>
    </div>
  )
}

export function AppNavbar() {
  const { user, isAuthenticated } = useAuth()
  const { theme, toggleTheme } = useTheme()
  const pathname = usePathname()
  const [mobileOpen, setMobileOpen] = useState(false)
  const isPublicPage = pathname === "/" || pathname === "/login" || pathname === "/register"

  return (
    <header className="sticky top-0 z-50 w-full border-b border-border bg-card/80 backdrop-blur-md">
      <div className={cn("flex h-14 items-center gap-4 px-4", isPublicPage ? "mx-auto max-w-7xl" : "")}>
        {isAuthenticated && !isPublicPage && (
          <Sheet open={mobileOpen} onOpenChange={setMobileOpen}>
            <SheetTrigger asChild>
              <Button variant="ghost" size="icon" className="lg:hidden">
                <Menu className="h-5 w-5" />
                <span className="sr-only">Toggle menu</span>
              </Button>
            </SheetTrigger>
            <SheetContent side="left" className="w-64 p-0">
              <SheetHeader className="sr-only">
                <SheetTitle>Navigation</SheetTitle>
              </SheetHeader>
              <SidebarContent onNavigate={() => setMobileOpen(false)} />
            </SheetContent>
          </Sheet>
        )}
        <Link href="/" className="flex items-center gap-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent">
            <BrainCircuit className="h-4 w-4 text-accent-foreground" />
          </div>
          <span className="text-sm font-bold text-foreground">LoanAI</span>
        </Link>
        <div className="flex-1" />
        <Button variant="ghost" size="icon" onClick={toggleTheme} className="shrink-0">
          <AnimatePresence mode="wait" initial={false}>
            <motion.div
              key={theme}
              initial={{ y: -12, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              exit={{ y: 12, opacity: 0 }}
              transition={{ duration: 0.15 }}
            >
              {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            </motion.div>
          </AnimatePresence>
          <span className="sr-only">Toggle theme</span>
        </Button>
        {isAuthenticated && user ? (
          <div className="flex items-center gap-2">
            <div className="hidden sm:block text-right">
              <p className="text-xs font-medium text-foreground leading-none">{user.name}</p>
              <p className="text-xs text-muted-foreground mt-0.5">{user.email}</p>
            </div>
            <div className="flex h-8 w-8 items-center justify-center rounded-full bg-accent text-accent-foreground text-xs font-semibold">
              {user.name.charAt(0).toUpperCase()}
            </div>
          </div>
        ) : (
          !isPublicPage && null
        )}
        {isPublicPage && !isAuthenticated && (
          <div className="flex items-center gap-2">
            <Link href="/login">
              <Button variant="ghost" size="sm">Log In</Button>
            </Link>
            <Link href="/register">
              <Button size="sm" className="bg-accent text-accent-foreground hover:bg-accent/90">Sign Up</Button>
            </Link>
          </div>
        )}
      </div>
    </header>
  )
}

export function AppSidebar() {
  return (
    <aside className="hidden lg:flex w-60 shrink-0 border-r border-border bg-card">
      <SidebarContent />
    </aside>
  )
}

export function AppLayout({ children }: { children: React.ReactNode }) {
  const { isAuthenticated } = useAuth()
  const pathname = usePathname()
  const isPublicPage = pathname === "/" || pathname === "/login" || pathname === "/register"
  const showSidebar = isAuthenticated && !isPublicPage

  return (
    <div className="flex min-h-screen flex-col">
      <AppNavbar />
      <div className="flex flex-1">
        {showSidebar && <AppSidebar />}
        <main className="flex-1 overflow-y-auto">
          {children}
        </main>
      </div>
    </div>
  )
}
