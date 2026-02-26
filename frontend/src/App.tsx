import { DraftBoard } from './components/DraftBoard'
import { WarmUpScreen } from './components/WarmUpScreen'

function App() {
  return (
    <WarmUpScreen>
      <div className="bg-background text-text-primary min-h-screen flex flex-col">
        <header className="text-center py-3">
          <h1 className="text-2xl font-bold text-gold-light">LoL Draft Predictor</h1>
          <p className="text-text-secondary text-xs">Match Outcome Prediction</p>
        </header>

        <main className="flex-1 px-4 pb-4 max-w-[1400px] mx-auto w-full">
          <DraftBoard />
        </main>
      </div>
    </WarmUpScreen>
  )
}

export default App
