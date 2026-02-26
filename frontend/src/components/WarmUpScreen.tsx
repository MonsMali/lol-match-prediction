import { useState, useEffect, type ReactNode } from 'react'

interface WarmUpScreenProps {
  children: ReactNode
}

export function WarmUpScreen({ children }: WarmUpScreenProps) {
  const [ready, setReady] = useState(false)
  const [status, setStatus] = useState('Warming up server...')

  useEffect(() => {
    let cancelled = false

    async function poll() {
      while (!cancelled) {
        try {
          const res = await fetch('/health')
          if (cancelled) break
          const data = await res.json()
          if (data.status === 'ready') {
            if (!cancelled) {
              setReady(true)
            }
            return
          }
          if (!cancelled) {
            setStatus('Loading ML model...')
          }
        } catch {
          if (!cancelled) {
            setStatus('Warming up server...')
          }
        }
        await new Promise((r) => setTimeout(r, 2000))
      }
    }

    poll()

    return () => {
      cancelled = true
    }
  }, [])

  if (ready) {
    return <>{children}</>
  }

  return (
    <div className="warmup-overlay">
      <div className="warmup-content">
        <h1 className="warmup-title">LoL Draft Predictor</h1>
        <div className="warmup-spinner" />
        <p className="warmup-status">{status}</p>
      </div>

      <style>{`
        .warmup-overlay {
          position: fixed;
          inset: 0;
          display: flex;
          align-items: center;
          justify-content: center;
          background: #0a0a0f;
          z-index: 9999;
        }

        .warmup-content {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 1.5rem;
        }

        .warmup-title {
          font-size: 2rem;
          font-weight: 700;
          color: #f0e6d2;
          letter-spacing: 0.025em;
        }

        .warmup-spinner {
          width: 40px;
          height: 40px;
          border: 3px solid #252540;
          border-top-color: #c89b3c;
          border-radius: 50%;
          animation: spin 1s linear infinite;
        }

        .warmup-status {
          font-size: 0.875rem;
          color: #8888aa;
        }

        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  )
}
