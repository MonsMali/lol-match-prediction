import { useQuery } from '@tanstack/react-query'
import { apiFetch } from './client'
import type { ChampionInfo } from '../types'

interface ChampionsResponse {
  champions: ChampionInfo[]
}

export function useChampions() {
  return useQuery({
    queryKey: ['champions'],
    queryFn: async () => {
      const data = await apiFetch<ChampionsResponse>('/api/champions')
      return data.champions
    },
    staleTime: Infinity,
  })
}
