import { DraftBoard } from './components/DraftBoard'
import { ChampionGrid } from './components/ChampionGrid'
import { useDraftStore } from './store/draftStore'

function App() {
  const selectChampion = useDraftStore((s) => s.selectChampion)

  return (
    <div className="bg-background text-text-primary min-h-screen flex flex-col">
      <header className="text-center py-4">
        <h1 className="text-3xl font-bold text-gold-light">LoL Draft Predictor</h1>
        <p className="text-text-secondary text-sm">Match Outcome Prediction</p>
      </header>

      <main className="flex-1 px-4 pb-4 max-w-7xl mx-auto w-full flex flex-col gap-4">
        <DraftBoard />
        <ChampionGrid onSelect={selectChampion} />
      </main>
    </div>
  )
}

export default App
