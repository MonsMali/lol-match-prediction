import { useRef } from 'react'
import { useMutation } from '@tanstack/react-query'
import { apiFetch } from './client'
import type { PredictRequest, PredictResponse, SuggestionsResponse } from '../types'

export function usePrediction() {
  const abortRef = useRef<AbortController | null>(null)

  return useMutation({
    mutationFn: async (draft: PredictRequest) => {
      if (abortRef.current) {
        abortRef.current.abort()
      }
      const controller = new AbortController()
      abortRef.current = controller

      return apiFetch<PredictResponse>('/api/predict', {
        method: 'POST',
        body: JSON.stringify(draft),
        signal: controller.signal,
      })
    },
  })
}

export function useSuggestions() {
  const abortRef = useRef<AbortController | null>(null)

  return useMutation({
    mutationFn: async (draft: PredictRequest) => {
      if (abortRef.current) {
        abortRef.current.abort()
      }
      const controller = new AbortController()
      abortRef.current = controller

      return apiFetch<SuggestionsResponse>('/api/suggestions', {
        method: 'POST',
        body: JSON.stringify(draft),
        signal: controller.signal,
      })
    },
  })
}
