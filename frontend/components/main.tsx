"use client";

import React, { useState, useEffect } from "react";
import { Play, Settings, Trophy, Clock } from "lucide-react";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
const API_URL = process.env.API_URL;

interface Config {
  attributes: string[];
  algorithms: string[];
  heuristics: string[];
}

interface PokemonData {
  [key: string]: string;
}

interface Feedback {
  [key: string]: string;
}

interface SolverStep {
  attempt: number;
  guess_name: string;
  guess_data: PokemonData;
  feedback: Feedback;
  remaining_candidates: number;
  timestamp: number;
}

interface SolverResult {
  secret_name: string;
  success: boolean;
  total_attempts: number;
  steps: SolverStep[];
  execution_time: number;
}

export default function PokedleVisualizer() {
  const [config, setConfig] = useState<Config | null>(null);
  const [selectedAlgorithm, setSelectedAlgorithm] = useState<string>("CSP");
  const [selectedAttributes, setSelectedAttributes] = useState<string[]>([]);
  const [selectedHeuristic, setSelectedHeuristic] = useState<string>("random");
  const [pokemonList, setPokemonList] = useState<string[]>([]);
  const [selectedPokemon, setSelectedPokemon] = useState<string>("");
  const [useRandomPokemon, setUseRandomPokemon] = useState<boolean>(true);
  const [solving, setSolving] = useState<boolean>(false);
  const [result, setResult] = useState<SolverResult | null>(null);
  const [currentStep, setCurrentStep] = useState<number>(0);
  const [isAnimating, setIsAnimating] = useState<boolean>(false);
  const [chartData, setChartData] = useState<
    { attempt: number; candidates: number; fitness?: number }[]
  >([]);

  // Fetch config and pokemon list on mount
  useEffect(() => {
    fetch(`${API_URL}/config`)
      .then((res) => res.json())
      .then((data: Config) => {
        setConfig(data);
        setSelectedAttributes(data.attributes);
      })
      .catch((err) => console.error("Error fetching config:", err));

    fetch(`${API_URL}/pokemon`)
      .then((res) => res.json())
      .then((data: { pokemon: string[] }) => setPokemonList(data.pokemon))
      .catch((err) => console.error("Error fetching pokemon list:", err));
  }, []);

  useEffect(() => {
    if (result) {
      const data = result.steps.map((step) => ({
        attempt: step.attempt,
        candidates: step.remaining_candidates,
        fitness: selectedAlgorithm === "GA" ? 0 : undefined, // Placeholder; update if API provides fitness
      }));
      setChartData(data);
    }
  }, [result, selectedAlgorithm]);

  const handleSolve = async () => {
    setSolving(true);
    setResult(null);
    setCurrentStep(0);

    try {
      const response = await fetch(`${API_URL}/solve`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          algorithm: selectedAlgorithm,
          attributes: selectedAttributes,
          heuristic: selectedHeuristic,
          secret_pokemon: useRandomPokemon ? null : selectedPokemon,
          max_attempts: 10,
        }),
      });

      const data: SolverResult = await response.json();
      setResult(data);

      // Animate through steps
      setIsAnimating(true);
      for (let i = 0; i < data.steps.length; i++) {
        await new Promise((resolve) => setTimeout(resolve, 1000));
        setCurrentStep(i + 1);
      }
      setIsAnimating(false);
    } catch (error) {
      alert("Error: " + (error as Error).message);
    } finally {
      setSolving(false);
    }
  };

  const toggleAttribute = (attr: string) => {
    if (selectedAttributes.includes(attr)) {
      setSelectedAttributes(selectedAttributes.filter((a) => a !== attr));
    } else {
      setSelectedAttributes([...selectedAttributes, attr]);
    }
  };

  const getFeedbackColor = (feedback: string): string => {
    if (feedback === "green") return "bg-green-500";
    if (feedback === "yellow") return "bg-yellow-500";
    if (feedback === "gray") return "bg-gray-400";
    if (feedback === "higher") return "bg-blue-400";
    if (feedback === "lower") return "bg-red-400";
    return "bg-gray-300";
  };

  const getFeedbackLabel = (feedback: string): string => {
    if (feedback === "higher") return "↑";
    if (feedback === "lower") return "↓";
    return feedback[0]?.toUpperCase() || "";
  };

  if (!config) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-indigo-100 to-purple-100 flex items-center justify-center">
        <div className="text-xl">Loading...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-100 to-purple-100 p-8">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-5xl font-bold text-center mb-2 text-indigo-900">
          🎮 Pokedle Solver
        </h1>
        <p className="text-center text-gray-600 mb-8">
          AI Algorithm Visualizer
        </p>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Configuration Panel */}
          <div className="bg-white rounded-lg shadow-lg p-6 space-y-6">
            <div className="flex items-center gap-2 mb-4">
              <Settings className="text-indigo-600" />
              <h2 className="text-2xl font-bold text-gray-800">
                Configuration
              </h2>
            </div>

            {/* Algorithm Selection */}
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                Algorithm
              </label>
              <div className="flex gap-2">
                {config.algorithms.map((algo) => (
                  <button
                    key={algo}
                    onClick={() => setSelectedAlgorithm(algo)}
                    className={`flex-1 py-2 px-4 rounded-lg font-medium transition ${
                      selectedAlgorithm === algo
                        ? "bg-indigo-600 text-white"
                        : "bg-gray-100 text-gray-700 hover:bg-gray-200"
                    }`}
                  >
                    {algo}
                  </button>
                ))}
              </div>
            </div>

            {/* Heuristic Selection */}
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                Heuristic
              </label>
              <select
                value={selectedHeuristic}
                onChange={(e) => setSelectedHeuristic(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
              >
                {config.heuristics.map((h) => (
                  <option key={h} value={h}>
                    {h}
                  </option>
                ))}
              </select>
            </div>

            {/* Attributes Selection */}
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                Attributes ({selectedAttributes.length}/
                {config.attributes.length})
              </label>
              <div className="grid grid-cols-2 gap-2">
                {config.attributes.map((attr) => (
                  <button
                    key={attr}
                    onClick={() => toggleAttribute(attr)}
                    className={`py-2 px-3 rounded-lg text-sm font-medium transition ${
                      selectedAttributes.includes(attr)
                        ? "bg-green-500 text-white"
                        : "bg-gray-100 text-gray-700 hover:bg-gray-200"
                    }`}
                  >
                    {attr}
                  </button>
                ))}
              </div>
            </div>

            {/* Pokemon Selection */}
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                Target Pokémon
              </label>
              <div className="space-y-2">
                <label className="flex items-center gap-2">
                  <input
                    type="radio"
                    checked={useRandomPokemon}
                    onChange={() => setUseRandomPokemon(true)}
                    className="text-indigo-600"
                  />
                  <span className="text-sm">Random Pokémon</span>
                </label>
                <label className="flex items-center gap-2">
                  <input
                    type="radio"
                    checked={!useRandomPokemon}
                    onChange={() => setUseRandomPokemon(false)}
                    className="text-indigo-600"
                  />
                  <span className="text-sm">Select specific</span>
                </label>
                {!useRandomPokemon && (
                  <select
                    value={selectedPokemon}
                    onChange={(e) => setSelectedPokemon(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500"
                  >
                    <option value="">Choose a Pokémon...</option>
                    {pokemonList.map((p) => (
                      <option key={p} value={p}>
                        {p}
                      </option>
                    ))}
                  </select>
                )}
              </div>
            </div>

            {/* Run Button */}
            <button
              onClick={handleSolve}
              disabled={solving || selectedAttributes.length === 0}
              className="w-full bg-indigo-600 text-white py-3 px-6 rounded-lg font-bold text-lg hover:bg-indigo-700 disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center justify-center gap-2 transition"
            >
              {solving ? (
                <>
                  <div className="animate-spin rounded-full h-5 w-5 border-2 border-white border-t-transparent"></div>
                  Solving...
                </>
              ) : (
                <>
                  <Play size={20} />
                  Run Solver
                </>
              )}
            </button>
          </div>

          {/* Visualization Panel */}
          <div className="lg:col-span-2 bg-white rounded-lg shadow-lg p-6">
            {!result ? (
              <div className="flex flex-col items-center justify-center h-full text-gray-400">
                <Trophy size={64} className="mb-4" />
                <p className="text-lg">
                  Configure and run the solver to see visualization
                </p>
              </div>
            ) : (
              <div className="space-y-6">
                {/* Result Header */}
                <div className="bg-gradient-to-r from-indigo-500 to-purple-500 text-white p-6 rounded-lg">
                  <div className="flex items-center justify-between mb-4">
                    <div>
                      <h3 className="text-2xl font-bold">
                        {result.success ? "🎉 Success!" : "❌ Failed"}
                      </h3>
                      <p className="text-indigo-100">
                        Secret: {result.secret_name}
                      </p>
                    </div>
                    <div className="text-right">
                      <div className="text-3xl font-bold">
                        {result.total_attempts}
                      </div>
                      <div className="text-sm text-indigo-100">attempts</div>
                    </div>
                  </div>
                  <div className="flex items-center gap-4 text-sm">
                    <div className="flex items-center gap-1">
                      <Clock size={16} />
                      <span>{result.execution_time.toFixed(3)}s</span>
                    </div>
                    <div>Algorithm: {selectedAlgorithm}</div>
                    <div>Attributes: {selectedAttributes.length}</div>
                  </div>
                </div>

                {/* Steps Visualization */}
                <div className="space-y-4">
                  <h4 className="text-lg font-bold text-gray-800">
                    Solution Steps
                  </h4>
                  {result.steps.slice(0, currentStep).map((step, idx) => (
                    <div
                      key={idx}
                      className={`border-l-4 pl-4 py-3 transition-all ${
                        idx === currentStep - 1 && isAnimating
                          ? "border-indigo-500 bg-indigo-50"
                          : "border-gray-300 bg-gray-50"
                      } rounded`}
                    >
                      <div className="flex items-start gap-4">
                        <img
                          src={step.guess_data.image_url}
                          alt={step.guess_name}
                          className="w-16 h-16 object-contain rounded-lg shadow"
                        />
                        <div className="flex-grow">
                          <div className="flex items-center justify-between mb-2">
                            <span className="font-bold text-gray-800">
                              Attempt {step.attempt}: {step.guess_name}
                            </span>
                            <span className="text-sm text-gray-500">
                              {step.remaining_candidates} candidates
                            </span>
                          </div>
                          <div className="flex flex-wrap gap-2">
                            {selectedAttributes.map((attr) => (
                              <div
                                key={attr}
                                className="flex flex-col items-center"
                              >
                                <div className="text-xs text-gray-600 mb-1">
                                  {attr}
                                </div>
                                <div
                                  className={`${getFeedbackColor(
                                    step.feedback[attr]
                                  )} text-white px-3 py-1 rounded font-medium text-sm min-w-[60px] text-center`}
                                >
                                  {getFeedbackLabel(step.feedback[attr])}
                                </div>
                                <div className="text-xs text-gray-500 mt-1">
                                  {step.guess_data[attr]}
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>

                {/* Charts */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-6">
                  <div className="bg-white p-4 rounded-lg shadow">
                    <h4 className="text-sm font-bold mb-2">
                      Remaining Candidates
                    </h4>
                    <ResponsiveContainer width="100%" height={200}>
                      <LineChart data={chartData}>
                        <XAxis dataKey="attempt" />
                        <YAxis />
                        <Tooltip />
                        <Line
                          type="monotone"
                          dataKey="candidates"
                          stroke="#4f46e5"
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                  {selectedAlgorithm === "GA" && (
                    <div className="bg-white p-4 rounded-lg shadow">
                      <h4 className="text-sm font-bold mb-2">
                        Average Fitness
                      </h4>
                      <ResponsiveContainer width="100%" height={200}>
                        <BarChart data={chartData}>
                          <XAxis dataKey="attempt" />
                          <YAxis />
                          <Tooltip />
                          <Bar dataKey="fitness" fill="#4f46e5" />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  )}
                </div>

                {/* Legend */}
                <div className="border-t pt-4 mt-4">
                  <h5 className="text-sm font-semibold text-gray-700 mb-2">
                    Legend:
                  </h5>
                  <div className="flex flex-wrap gap-3 text-xs">
                    <div className="flex items-center gap-1">
                      <div className="w-4 h-4 bg-green-500 rounded"></div>
                      <span>Exact match</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <div className="w-4 h-4 bg-yellow-500 rounded"></div>
                      <span>Wrong position</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <div className="w-4 h-4 bg-blue-400 rounded"></div>
                      <span>Higher ↑</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <div className="w-4 h-4 bg-red-400 rounded"></div>
                      <span>Lower ↓</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <div className="w-4 h-4 bg-gray-400 rounded"></div>
                      <span>Wrong</span>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
