import { useDraftStore } from '../store/draftStore'
import type { DraftMode } from '../types'

const TABS: { mode: DraftMode; label: string }[] = [
  { mode: 'live', label: 'Live Draft' },
  { mode: 'bulk', label: 'Quick Entry' },
]

export function ModeToggle() {
  const mode = useDraftStore((s) => s.mode)
  const setMode = useDraftStore((s) => s.setMode)

  return (
    <div className="flex gap-0 border-b border-panel-light">
      {TABS.map((tab) => (
        <button
          key={tab.mode}
          type="button"
          onClick={() => setMode(tab.mode)}
          className={`
            px-6 py-2 text-sm font-medium transition-colors
            ${mode === tab.mode
              ? 'text-gold border-b-2 border-gold'
              : 'text-text-secondary hover:text-text-primary'
            }
          `}
        >
          {tab.label}
        </button>
      ))}
    </div>
  )
}
