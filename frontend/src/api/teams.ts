import { useQuery } from '@tanstack/react-query'
import { apiFetch } from './client'

interface TeamsResponse {
  teams: Record<string, string[]>
}

export function useTeams() {
  return useQuery({
    queryKey: ['teams'],
    queryFn: async () => {
      const data = await apiFetch<TeamsResponse>('/api/teams')
      return data.teams
    },
    staleTime: Infinity,
  })
}
