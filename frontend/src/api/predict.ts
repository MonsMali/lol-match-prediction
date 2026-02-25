import { useMutation } from '@tanstack/react-query'
import { apiFetch } from './client'
import type { PredictRequest, PredictResponse } from '../types'

export function usePrediction() {
  return useMutation({
    mutationFn: async (draft: PredictRequest) => {
      return apiFetch<PredictResponse>('/api/predict', {
        method: 'POST',
        body: JSON.stringify(draft),
      })
    },
  })
}
