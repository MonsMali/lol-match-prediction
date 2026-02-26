import { useMemo } from 'react'
import { useChampions } from '../api/champions'
import type { ChampionInfo } from '../types'

export function useChampionLookup(): Map<string, ChampionInfo> {
  const { data } = useChampions()

  return useMemo(() => {
    if (!data) return new Map<string, ChampionInfo>()
    const map = new Map<string, ChampionInfo>()
    for (const champion of data) {
      map.set(champion.name, champion)
    }
    return map
  }, [data])
}
