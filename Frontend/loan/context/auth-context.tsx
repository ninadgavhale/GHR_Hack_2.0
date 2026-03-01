"use client"

import React, { createContext, useContext, useState, useCallback, type ReactNode } from "react"

export interface User {
  id: string
  name: string
  email: string
  avatar?: string
}

interface AuthContextType {
  user: User | null
  isAuthenticated: boolean
  login: (email: string, password: string) => Promise<boolean>
  register: (name: string, email: string, password: string) => Promise<boolean>
  logout: () => void
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null)

  const login = useCallback(async (email: string, _password: string): Promise<boolean> => {
    await new Promise((r) => setTimeout(r, 800))
    setUser({
      id: "usr_001",
      name: email.split("@")[0].replace(/[^a-zA-Z]/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()),
      email,
      avatar: undefined,
    })
    return true
  }, [])

  const register = useCallback(async (name: string, email: string, _password: string): Promise<boolean> => {
    await new Promise((r) => setTimeout(r, 800))
    setUser({ id: "usr_" + Date.now().toString(36), name, email })
    return true
  }, [])

  const logout = useCallback(() => {
    setUser(null)
  }, [])

  return (
    <AuthContext.Provider value={{ user, isAuthenticated: !!user, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error("useAuth must be used within AuthProvider")
  return ctx
}
