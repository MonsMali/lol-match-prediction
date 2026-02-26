import type { ModelMeta } from '../types'

interface ModelBannerProps {
  meta: ModelMeta | null
}

export function ModelBanner({ meta }: ModelBannerProps) {
  if (!meta) return null

  return (
    <div className="w-full bg-yellow-900/30 border border-yellow-700/40 rounded px-3 py-1.5 text-center">
      <p className="text-xs text-yellow-200/90">
        Model trained on <span className="font-semibold">professional match data</span> through{' '}
        <span className="font-mono font-semibold">Patch {meta.training_patch}</span>{' '}
        ({meta.training_year}).
        Predictions reflect professional-level play patterns and may not apply to other skill levels.
      </p>
    </div>
  )
}
