import { useState, useRef, useEffect } from 'react'

interface Option {
  value: string
  label: string
  group?: string
}

interface SearchableSelectProps {
  options: Option[]
  value: string | null
  onChange: (value: string) => void
  placeholder?: string
  className?: string
  borderColor?: string
  disabled?: boolean
}

export function SearchableSelect({
  options,
  value,
  onChange,
  placeholder = 'Select...',
  borderColor = 'border-panel-light',
  disabled = false,
}: SearchableSelectProps) {
  const [isOpen, setIsOpen] = useState(false)
  const [query, setQuery] = useState('')
  const containerRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const selectedOption = options.find((o) => o.value === value)

  const filtered = query
    ? options.filter((o) => o.label.toLowerCase().includes(query.toLowerCase()))
    : options

  // Group filtered options
  const groups = new Map<string, Option[]>()
  for (const opt of filtered) {
    const group = opt.group ?? ''
    if (!groups.has(group)) groups.set(group, [])
    groups.get(group)!.push(opt)
  }

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setIsOpen(false)
        setQuery('')
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus()
    }
  }, [isOpen])

  function handleSelect(val: string) {
    onChange(val)
    setIsOpen(false)
    setQuery('')
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === 'Escape') {
      setIsOpen(false)
      setQuery('')
    }
  }

  return (
    <div ref={containerRef} className="relative w-full" onKeyDown={handleKeyDown}>
      <button
        type="button"
        disabled={disabled}
        onClick={() => {
          if (!disabled) {
            setIsOpen(!isOpen)
            setQuery('')
          }
        }}
        className={`w-full px-2 py-1.5 rounded bg-panel text-sm border ${borderColor}
          text-left flex items-center justify-between gap-1
          focus:outline-none transition-colors
          ${disabled ? 'text-text-secondary cursor-not-allowed' : 'text-text-primary cursor-pointer hover:border-gold/50'}`}
      >
        <span className={selectedOption ? 'text-text-primary' : 'text-text-secondary'}>
          {selectedOption?.label ?? placeholder}
        </span>
        <svg
          className={`w-3 h-3 text-text-secondary transition-transform ${isOpen ? 'rotate-180' : ''}`}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          strokeWidth={2}
        >
          <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {isOpen && (
        <div className="absolute z-50 mt-1 w-full bg-panel border border-panel-light rounded shadow-lg shadow-black/40">
          <div className="p-1.5">
            <input
              ref={inputRef}
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search..."
              className="w-full px-2 py-1 rounded bg-panel-light text-text-primary text-sm
                placeholder-text-secondary border border-transparent focus:border-gold/40 focus:outline-none"
            />
          </div>
          <div className="max-h-[240px] overflow-y-auto">
            {filtered.length === 0 && (
              <div className="px-3 py-2 text-xs text-text-secondary text-center">No results</div>
            )}
            {Array.from(groups.entries()).map(([group, opts]) => (
              <div key={group}>
                {group && (
                  <div className="px-2 py-1 text-[10px] font-semibold uppercase tracking-wider text-text-secondary bg-panel-light/50 sticky top-0">
                    {group}
                  </div>
                )}
                {opts.map((opt) => (
                  <button
                    key={opt.value}
                    type="button"
                    onClick={() => handleSelect(opt.value)}
                    className={`w-full text-left px-3 py-1.5 text-sm transition-colors
                      ${opt.value === value
                        ? 'bg-gold/15 text-gold-light'
                        : 'text-text-primary hover:bg-panel-light'
                      }`}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
