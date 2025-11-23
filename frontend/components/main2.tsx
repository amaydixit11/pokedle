"use client";

import React, { useState, useEffect } from "react";
import {
  Play,
  RotateCcw,
  Settings,
  ChevronRight,
  ChevronLeft,
  Clock,
  Target,
  Zap,
  Menu,
  X,
  TrendingUp,
  Activity,
  BarChart3,
  GitCompare,
  Info,
} from "lucide-react";
import GAVisualization from "./GAVisualization";
import AStarVisualization from "./AStartVisualization";
import JsonViewer from "./JsonViewer";
import PathVisualizer from "./PathVisualizer";

// Types
interface SolverConfig {
  algorithm: string;
  attributes: string[];
  heuristic: string;
  secret_pokemon: string | null;
  max_attempts: number;
  ga_config?: GAConfig;
  sa_config?: SAConfig;
  astar_config?: AStarConfig;
  csp_config?: CSPConfig;
}

interface CSPConfig {
  variable_heuristic: string;
  value_heuristic: string;
  use_ac3: boolean;
}

interface GAConfig {
  pop_size: number;
  elite_size: number;
  mutation_rate: number;
  crossover_rate: number;
  tournament_size: number;
  crossover_strategy: string;
  generations_per_guess: number;
}

interface SAConfig {
  initial_temp: number;
  cooling_rate: number;
  min_temp: number;
  iterations_per_temp: number;
  reheat_threshold: number;
}

interface AStarConfig {
  max_open_set: number;
  beam_width: number;
  heuristic_weight: number;
}

interface SolverStep {
  attempt: number;
  guess_name: string;
  guess_data: Record<string, string>;
  feedback: Record<string, string>;
  remaining_candidates: number;
  timestamp: number;
  image_url?: string;
  heuristic_info?: Record<string, any>;
  algorithm_state?: Record<string, any>;
}

interface SolverResult {
  secret_name: string;
  secret_image: string;
  success: boolean;
  total_attempts: number;
  steps: SolverStep[];
  execution_time: number;
  algorithm: string;
  heuristic?: string;
  algorithm_config?: Record<string, any>;
  performance_metrics?: Record<string, any>;
}

interface CompareResult {
  secret_pokemon: string;
  results: Record<string, any>;
  winner: string | null;
}

const API_URL = "http://localhost:8000";

export default function PokedleVisualizer() {
  const [config, setConfig] = useState<SolverConfig>({
    algorithm: "CSP",
    attributes: ["Generation", "Type1", "Type2", "Color"],
    heuristic: "entropy",
    secret_pokemon: null,
    max_attempts: 10,
  });

  const [cspConfig, setCspConfig] = useState<CSPConfig>({
    variable_heuristic: "mrv",
    value_heuristic: "lcv",
    use_ac3: true,
  });

  const [gaConfig, setGaConfig] = useState<GAConfig>({
    pop_size: 100,
    elite_size: 20,
    mutation_rate: 0.15,
    crossover_rate: 0.8,
    tournament_size: 7,
    crossover_strategy: "attribute_blend",
    generations_per_guess: 30,
  });

  const [saConfig, setSaConfig] = useState<SAConfig>({
    initial_temp: 100.0,
    cooling_rate: 0.95,
    min_temp: 0.01,
    iterations_per_temp: 50,
    reheat_threshold: 0.1,
  });

  const [astarConfig, setAstarConfig] = useState<AStarConfig>({
    max_open_set: 1000,
    beam_width: 100,
    heuristic_weight: 1.0,
  });

  const [result, setResult] = useState<SolverResult | null>(null);
  const [compareResults, setCompareResults] = useState<CompareResult | null>(
    null
  );
  const [loading, setLoading] = useState(false);
  const [comparing, setComparing] = useState(false);
  const [availableAttrs, setAvailableAttrs] = useState<string[]>([]);
  const [availableAlgorithms, setAvailableAlgorithms] = useState<string[]>([]);
  const [algorithmDescriptions, setAlgorithmDescriptions] = useState<
    Record<string, string>
  >({});
  const [variableHeuristics, setVariableHeuristics] = useState<
    Record<string, string>
  >({});
  const [valueHeuristics, setValueHeuristics] = useState<
    Record<string, string>
  >({});
  const [availableCrossoverStrategies, setAvailableCrossoverStrategies] =
    useState<Record<string, string>>({});
  const [pokemonList, setPokemonList] = useState<
    Array<{ name: string; image_url: string }>
  >([]);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [currentStep, setCurrentStep] = useState(0);
  const [activeTab, setActiveTab] = useState<"solve" | "compare">("solve");

  // Fetch config options
  useEffect(() => {
    fetch(`${API_URL}/config`)
      .then((res) => res.json())
      .then((data) => {
        setAvailableAttrs(data.attributes || []);
        setAvailableAlgorithms(data.algorithms || []);
        setAlgorithmDescriptions(data.algorithm_descriptions || {});

        // Handle new CSP heuristic structure
        if (data.csp_heuristics) {
          setVariableHeuristics(
            data.csp_heuristics.variable_ordering?.descriptions || {}
          );
          setValueHeuristics(
            data.csp_heuristics.value_ordering?.descriptions || {}
          );
        }

        setAvailableCrossoverStrategies(data.crossover_strategies || {});
      })
      .catch((err) => console.error("Failed to fetch config:", err));

    fetch(`${API_URL}/pokemon`)
      .then((res) => res.json())
      .then((data) => setPokemonList(data.pokemon || []))
      .catch((err) => console.error("Failed to fetch Pokemon:", err));
  }, []);

  const runSolver = async () => {
    setLoading(true);
    setResult(null);
    setCurrentStep(0);

    try {
      const configToSend: any = { ...config };

      if (config.algorithm === "CSP") {
        configToSend.csp_config = cspConfig;
      } else if (config.algorithm === "GA") {
        configToSend.ga_config = gaConfig;
      } else if (config.algorithm === "SA") {
        configToSend.sa_config = saConfig;
      } else if (config.algorithm === "ASTAR") {
        configToSend.astar_config = astarConfig;
      }

      const response = await fetch(`${API_URL}/solve`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(configToSend),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Solver failed");
      }

      const data: SolverResult = await response.json();
      setResult(data);
    } catch (err: any) {
      console.error("Error running solver:", err);
      alert(
        err.message ||
          "Failed to run solver. Make sure backend is running on port 8000."
      );
    } finally {
      setLoading(false);
    }
  };

  const runComparison = async () => {
    setComparing(true);
    setCompareResults(null);

    try {
      const response = await fetch(`${API_URL}/compare`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          algorithms: availableAlgorithms,
          attributes: config.attributes,
          secret_pokemon: config.secret_pokemon,
          max_attempts: config.max_attempts,
        }),
      });

      if (!response.ok) {
        throw new Error("Comparison failed");
      }

      const data: CompareResult = await response.json();
      setCompareResults(data);
    } catch (err) {
      console.error("Error running comparison:", err);
      alert("Failed to run comparison. Make sure backend is running.");
    } finally {
      setComparing(false);
    }
  };

  const getFeedbackColor = (status: string): string => {
    switch (status) {
      case "green":
        return "bg-green-500";
      case "yellow":
        return "bg-yellow-500";
      case "gray":
        return "bg-gray-400";
      case "higher":
        return "bg-blue-500";
      case "lower":
        return "bg-red-500";
      default:
        return "bg-gray-300";
    }
  };

  const getFeedbackLabel = (status: string): string => {
    switch (status) {
      case "green":
        return "✓";
      case "yellow":
        return "↔";
      case "gray":
        return "✗";
      case "higher":
        return "↑";
      case "lower":
        return "↓";
      default:
        return "?";
    }
  };

  const getAlgorithmIcon = (algo: string) => {
    switch (algo) {
      case "CSP":
        return <Target className="w-4 h-4" />;
      case "GA":
        return <TrendingUp className="w-4 h-4" />;
      case "ASTAR":
        return <Activity className="w-4 h-4" />;
      case "SA":
        return <Zap className="w-4 h-4" />;
      default:
        return <BarChart3 className="w-4 h-4" />;
    }
  };

  const getAlgorithmColor = (algo: string) => {
    switch (algo) {
      case "CSP":
        return "blue";
      case "GA":
        return "green";
      case "ASTAR":
        return "purple";
      case "SA":
        return "orange";
      default:
        return "gray";
    }
  };

  const renderAlgorithmStateValue = (key: string, value: any) => {
    if (typeof value === "number") {
      return value.toFixed(2);
    }
    if (
      typeof value === "object" &&
      value !== null &&
      (key === "open_set_nodes" ||
        key === "closed_set_nodes" ||
        key === "current_node" ||
        Array.isArray(value))
    ) {
      return (
        <pre className="text-xs bg-gray-100 p-2 rounded mt-1 overflow-auto max-h-40">
          {JSON.stringify(value, null, 2)}
        </pre>
      );
    }
    return String(value);
  };

  useEffect(() => {
    console.log(result);
  }, [result]);

  return (
    <div className="flex h-screen bg-gray-50 overflow-hidden">
      {/* Sidebar */}
      <div
        className={`${
          sidebarOpen ? "w-80" : "w-0"
        } transition-all duration-300 bg-white border-r border-gray-200 flex flex-col overflow-hidden`}
      >
        <div className="p-4 border-b border-gray-200 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Settings className="w-5 h-5 text-gray-700" />
            <h2 className="font-semibold text-gray-900">Configuration</h2>
          </div>
          <button
            onClick={() => setSidebarOpen(false)}
            className="p-1 hover:bg-gray-100 rounded"
          >
            <X className="w-5 h-5 text-gray-500" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-6">
          {/* Algorithm Selection */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Algorithm
            </label>
            <div className="grid grid-cols-2 gap-2">
              {availableAlgorithms.map((algo) => {
                const color = getAlgorithmColor(algo);
                return (
                  <button
                    key={algo}
                    onClick={() => setConfig({ ...config, algorithm: algo })}
                    className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors flex items-center justify-center gap-2 ${
                      config.algorithm === algo
                        ? `bg-${color}-600 text-white`
                        : "bg-gray-100 text-gray-700 hover:bg-gray-200"
                    }`}
                  >
                    {getAlgorithmIcon(algo)}
                    {algo}
                  </button>
                );
              })}
            </div>
            {algorithmDescriptions[config.algorithm] && (
              <div className="mt-2 p-2 bg-blue-50 rounded text-xs text-gray-600 flex items-start gap-2">
                <Info className="w-3 h-3 mt-0.5 flex-shrink-0" />
                <span>{algorithmDescriptions[config.algorithm]}</span>
              </div>
            )}
          </div>

          {/* CSP Heuristics */}
          {config.algorithm === "CSP" && (
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Variable Ordering Heuristic
                </label>
                <select
                  value={cspConfig.variable_heuristic}
                  onChange={(e) =>
                    setCspConfig({
                      ...cspConfig,
                      variable_heuristic: e.target.value,
                    })
                  }
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  {Object.entries(variableHeuristics).map(([key, desc]) => (
                    <option key={key} value={key}>
                      {key.toUpperCase()}
                    </option>
                  ))}
                </select>
                <p className="mt-1 text-xs text-gray-500">
                  {variableHeuristics[cspConfig.variable_heuristic]}
                </p>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Value Ordering Heuristic
                </label>
                <select
                  value={cspConfig.value_heuristic}
                  onChange={(e) =>
                    setCspConfig({
                      ...cspConfig,
                      value_heuristic: e.target.value,
                    })
                  }
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  {Object.entries(valueHeuristics).map(([key, desc]) => (
                    <option key={key} value={key}>
                      {key.toUpperCase()}
                    </option>
                  ))}
                </select>
                <p className="mt-1 text-xs text-gray-500">
                  {valueHeuristics[cspConfig.value_heuristic]}
                </p>
              </div>

              <div>
                <label className="flex items-center gap-2 text-sm cursor-pointer">
                  <input
                    type="checkbox"
                    checked={cspConfig.use_ac3}
                    onChange={(e) =>
                      setCspConfig({
                        ...cspConfig,
                        use_ac3: e.target.checked,
                      })
                    }
                    className="rounded"
                  />
                  <span className="text-gray-700">
                    Use AC-3 Constraint Propagation
                  </span>
                </label>
                <p className="mt-1 ml-6 text-xs text-gray-500">
                  Automatically reduces domains using arc consistency
                </p>
              </div>
            </div>
          )}

          {/* GA Configuration */}
          {config.algorithm === "GA" && (
            <div className="space-y-4">
              <div>
                <label className="block text-xs font-medium text-gray-600 mb-1">
                  Population: {gaConfig.pop_size}
                </label>
                <input
                  type="range"
                  min="50"
                  max="300"
                  step="10"
                  value={gaConfig.pop_size}
                  onChange={(e) =>
                    setGaConfig({
                      ...gaConfig,
                      pop_size: parseInt(e.target.value),
                    })
                  }
                  className="w-full"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-gray-600 mb-1">
                  Elite Size: {gaConfig.elite_size}
                </label>
                <input
                  type="range"
                  min="5"
                  max="50"
                  step="5"
                  value={gaConfig.elite_size}
                  onChange={(e) =>
                    setGaConfig({
                      ...gaConfig,
                      elite_size: parseInt(e.target.value),
                    })
                  }
                  className="w-full"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-gray-600 mb-1">
                  Mutation: {(gaConfig.mutation_rate * 100).toFixed(0)}%
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={gaConfig.mutation_rate}
                  onChange={(e) =>
                    setGaConfig({
                      ...gaConfig,
                      mutation_rate: parseFloat(e.target.value),
                    })
                  }
                  className="w-full"
                />
              </div>
              {/* <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Crossover Strategy
                </label>
                <select
                  value={gaConfig.crossover_strategy}
                  onChange={(e) =>
                    setGaConfig({
                      ...gaConfig,
                      crossover_strategy: e.target.value,
                    })
                  }
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  {Object.keys(availableCrossoverStrategies).map((key) => (
                    <option key={key} value={key}>
                      {key.replace(/_/g, " ")}
                    </option>
                  ))}
                </select>
              </div> */}
              <div>
                <label className="block text-xs font-medium text-gray-600 mb-1">
                  Generations/Guess: {gaConfig.generations_per_guess}
                </label>
                <input
                  type="range"
                  min="10"
                  max="100"
                  step="5"
                  value={gaConfig.generations_per_guess}
                  onChange={(e) =>
                    setGaConfig({
                      ...gaConfig,
                      generations_per_guess: parseInt(e.target.value),
                    })
                  }
                  className="w-full"
                />
              </div>
            </div>
          )}

          {/* A* Configuration */}
          {config.algorithm === "ASTAR" && (
            <div className="space-y-4">
              <div>
                <label className="block text-xs font-medium text-gray-600 mb-1">
                  Beam Width: {astarConfig.beam_width}
                </label>
                <input
                  type="range"
                  min="10"
                  max="200"
                  step="10"
                  value={astarConfig.beam_width}
                  onChange={(e) =>
                    setAstarConfig({
                      ...astarConfig,
                      beam_width: parseInt(e.target.value),
                    })
                  }
                  className="w-full"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-gray-600 mb-1">
                  Heuristic Weight: {astarConfig.heuristic_weight.toFixed(1)}
                </label>
                <input
                  type="range"
                  min="0"
                  max="3"
                  step="0.1"
                  value={astarConfig.heuristic_weight}
                  onChange={(e) =>
                    setAstarConfig({
                      ...astarConfig,
                      heuristic_weight: parseFloat(e.target.value),
                    })
                  }
                  className="w-full"
                />
                <p className="mt-1 text-xs text-gray-500">
                  1.0 = admissible (optimal), &gt;1.0 = faster but not optimal
                </p>
              </div>
            </div>
          )}

          {/* SA Configuration */}
          {config.algorithm === "SA" && (
            <div className="space-y-4">
              <div>
                <label className="block text-xs font-medium text-gray-600 mb-1">
                  Initial Temp: {saConfig.initial_temp}
                </label>
                <input
                  type="range"
                  min="50"
                  max="200"
                  step="10"
                  value={saConfig.initial_temp}
                  onChange={(e) =>
                    setSaConfig({
                      ...saConfig,
                      initial_temp: parseFloat(e.target.value),
                    })
                  }
                  className="w-full"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-gray-600 mb-1">
                  Cooling Rate: {saConfig.cooling_rate.toFixed(2)}
                </label>
                <input
                  type="range"
                  min="0.85"
                  max="0.99"
                  step="0.01"
                  value={saConfig.cooling_rate}
                  onChange={(e) =>
                    setSaConfig({
                      ...saConfig,
                      cooling_rate: parseFloat(e.target.value),
                    })
                  }
                  className="w-full"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-gray-600 mb-1">
                  Iterations/Temp: {saConfig.iterations_per_temp}
                </label>
                <input
                  type="range"
                  min="20"
                  max="100"
                  step="5"
                  value={saConfig.iterations_per_temp}
                  onChange={(e) =>
                    setSaConfig({
                      ...saConfig,
                      iterations_per_temp: parseInt(e.target.value),
                    })
                  }
                  className="w-full"
                />
              </div>
            </div>
          )}

          {/* Attribute Selection */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Attributes ({config.attributes.length})
            </label>
            <div className="space-y-2">
              {availableAttrs.map((attr) => (
                <label
                  key={attr}
                  className="flex items-center gap-2 text-sm cursor-pointer"
                >
                  <input
                    type="checkbox"
                    checked={config.attributes.includes(attr)}
                    onChange={(e) => {
                      if (e.target.checked) {
                        setConfig({
                          ...config,
                          attributes: [...config.attributes, attr],
                        });
                      } else {
                        setConfig({
                          ...config,
                          attributes: config.attributes.filter(
                            (a) => a !== attr
                          ),
                        });
                      }
                    }}
                    className="rounded"
                  />
                  <span className="text-gray-700">{attr}</span>
                </label>
              ))}
            </div>
          </div>

          {/* Secret Pokemon */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Secret Pokemon
            </label>
            <select
              value={config.secret_pokemon || ""}
              onChange={(e) =>
                setConfig({ ...config, secret_pokemon: e.target.value || null })
              }
              className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="">Random</option>
              {pokemonList.map((p) => (
                <option key={p.name} value={p.name}>
                  {p.name}
                </option>
              ))}
            </select>
          </div>

          {/* Max Attempts */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Max Attempts: {config.max_attempts}
            </label>
            <input
              type="range"
              min="5"
              max="20"
              value={config.max_attempts}
              onChange={(e) =>
                setConfig({ ...config, max_attempts: parseInt(e.target.value) })
              }
              className="w-full"
            />
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <div className="bg-white border-b border-gray-200 px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              {!sidebarOpen && (
                <button
                  onClick={() => setSidebarOpen(true)}
                  className="p-2 hover:bg-gray-100 rounded-lg"
                >
                  <Menu className="w-5 h-5 text-gray-600" />
                </button>
              )}
              <div>
                <h1 className="text-2xl font-bold text-gray-900">
                  Enhanced Pokedle AI Solver
                </h1>
                <p className="text-sm text-gray-500">
                  Multi-Algorithm Dashboard v5.0
                </p>
              </div>
            </div>

            <div className="flex items-center gap-3">
              {/* Tab Selector */}
              <div className="flex bg-gray-100 rounded-lg p-1">
                <button
                  onClick={() => setActiveTab("solve")}
                  className={`px-4 py-1.5 rounded-md text-sm font-medium transition-colors ${
                    activeTab === "solve"
                      ? "bg-white text-gray-900 shadow-sm"
                      : "text-gray-600"
                  }`}
                >
                  Solve
                </button>
                <button
                  onClick={() => setActiveTab("compare")}
                  className={`px-4 py-1.5 rounded-md text-sm font-medium transition-colors flex items-center gap-2 ${
                    activeTab === "compare"
                      ? "bg-white text-gray-900 shadow-sm"
                      : "text-gray-600"
                  }`}
                >
                  <GitCompare className="w-4 h-4" />
                  Compare
                </button>
              </div>

              {result && activeTab === "solve" && (
                <button
                  onClick={() => {
                    setResult(null);
                    setCurrentStep(0);
                  }}
                  className="flex items-center gap-2 px-4 py-2 text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
                >
                  <RotateCcw className="w-4 h-4" />
                  Reset
                </button>
              )}

              {activeTab === "solve" ? (
                <button
                  onClick={runSolver}
                  disabled={loading || config.attributes.length === 0}
                  className="flex items-center gap-2 px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed"
                >
                  <Play className="w-4 h-4" />
                  {loading ? "Running..." : "Run Solver"}
                </button>
              ) : (
                <button
                  onClick={runComparison}
                  disabled={comparing || config.attributes.length === 0}
                  className="flex items-center gap-2 px-6 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed"
                >
                  <GitCompare className="w-4 h-4" />
                  {comparing ? "Comparing..." : "Compare All"}
                </button>
              )}
            </div>
          </div>
        </div>

        {/* Content Area */}
        <div className="flex-1 overflow-y-auto p-6">
          {activeTab === "solve" ? (
            result ? (
              <div className="space-y-6">
                {/* Stats Grid */}
                <div className="grid grid-cols-5 gap-4">
                  <div className="bg-white rounded-lg border border-gray-200 p-4">
                    <div className="flex items-center gap-3">
                      <div className="p-2 bg-green-100 rounded-lg">
                        <Target className="w-5 h-5 text-green-600" />
                      </div>
                      <div>
                        <p className="text-xs text-gray-500 font-medium">
                          Result
                        </p>
                        <p className="text-lg font-bold text-gray-900">
                          {result.success ? "Success" : "Failed"}
                        </p>
                      </div>
                    </div>
                  </div>
                  <div className="bg-white rounded-lg border border-gray-200 p-4">
                    <div className="flex items-center gap-3">
                      <div className="p-2 bg-blue-100 rounded-lg">
                        <ChevronRight className="w-5 h-5 text-blue-600" />
                      </div>
                      <div>
                        <p className="text-xs text-gray-500 font-medium">
                          Attempts
                        </p>
                        <p className="text-lg font-bold text-gray-900">
                          {result.total_attempts}
                        </p>
                      </div>
                    </div>
                  </div>
                  <div className="bg-white rounded-lg border border-gray-200 p-4">
                    <div className="flex items-center gap-3">
                      <div className="p-2 bg-purple-100 rounded-lg">
                        <Clock className="w-5 h-5 text-purple-600" />
                      </div>
                      <div>
                        <p className="text-xs text-gray-500 font-medium">
                          Time
                        </p>
                        <p className="text-lg font-bold text-gray-900">
                          {result.execution_time.toFixed(2)}s
                        </p>
                      </div>
                    </div>
                  </div>
                  <div className="bg-white rounded-lg border border-gray-200 p-4">
                    <div className="flex items-center gap-3">
                      <div
                        className={`p-2 bg-${getAlgorithmColor(
                          result.algorithm
                        )}-100 rounded-lg`}
                      >
                        {getAlgorithmIcon(result.algorithm)}
                      </div>
                      <div>
                        <p className="text-xs text-gray-500 font-medium">
                          Algorithm
                        </p>
                        <p className="text-lg font-bold text-gray-900">
                          {result.algorithm}
                        </p>
                      </div>
                    </div>
                  </div>
                  {result.performance_metrics && (
                    <div className="bg-white rounded-lg border border-gray-200 p-4">
                      <div className="flex items-center gap-3">
                        <div className="p-2 bg-orange-100 rounded-lg">
                          <TrendingUp className="w-5 h-5 text-orange-600" />
                        </div>
                        <div>
                          <p className="text-xs text-gray-500 font-medium">
                            Efficiency
                          </p>
                          <p className="text-lg font-bold text-gray-900">
                            {(
                              result.performance_metrics.efficiency * 100
                            ).toFixed(0)}
                            %
                          </p>
                        </div>
                      </div>
                    </div>
                  )}
                </div>

                {/* Secret Pokemon */}
                <div className="bg-gradient-to-r from-yellow-50 to-orange-50 rounded-lg border border-yellow-200 p-6">
                  <div className="flex items-center gap-6">
                    {result.secret_image && (
                      <img
                        src={result.secret_image}
                        alt={result.secret_name}
                        className="w-24 h-24 object-contain bg-white rounded-lg p-2 border border-gray-200"
                      />
                    )}
                    <div>
                      <p className="text-sm text-gray-600 font-medium mb-1">
                        Secret Pokemon
                      </p>
                      <p className="text-3xl font-bold text-gray-900">
                        {result.secret_name}
                      </p>
                      {result.algorithm_config && (
                        <div className="mt-2 flex flex-wrap gap-2">
                          {result.algorithm === "CSP" && (
                            <>
                              <span className="px-2 py-1 bg-blue-100 text-blue-700 text-xs rounded">
                                Var:{" "}
                                {result.algorithm_config.variable_heuristic?.toUpperCase()}
                              </span>
                              <span className="px-2 py-1 bg-blue-100 text-blue-700 text-xs rounded">
                                Val:{" "}
                                {result.algorithm_config.value_heuristic?.toUpperCase()}
                              </span>
                              {result.algorithm_config.use_ac3 && (
                                <span className="px-2 py-1 bg-green-100 text-green-700 text-xs rounded">
                                  AC-3 Enabled
                                </span>
                              )}
                            </>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
                {/* Current Step */}
                <div className="grid grid-cols-3 gap-6">
                  <div className="col-span-2 bg-white rounded-lg border border-gray-200 p-6">
                    <div className="flex items-center justify-between mb-6">
                      <div>
                        <p className="text-sm text-gray-500 font-medium">
                          Attempt #{result.steps[currentStep]?.attempt}
                        </p>
                        <h3 className="text-2xl font-bold text-gray-900">
                          {result.steps[currentStep]?.guess_name}
                        </h3>
                      </div>
                      <div className="flex items-center gap-2">
                        <button
                          onClick={() =>
                            setCurrentStep(Math.max(0, currentStep - 1))
                          }
                          disabled={currentStep === 0}
                          className="p-2 border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                          <ChevronLeft className="w-4 h-4" />
                        </button>
                        <span className="text-sm text-gray-600 px-3">
                          {currentStep + 1} / {result.steps.length}
                        </span>
                        <button
                          onClick={() =>
                            setCurrentStep(
                              Math.min(result.steps.length - 1, currentStep + 1)
                            )
                          }
                          disabled={currentStep === result.steps.length - 1}
                          className="p-2 border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                          <ChevronRight className="w-4 h-4" />
                        </button>
                      </div>
                    </div>

                    {/* Attributes Grid */}
                    <div className="grid grid-cols-3 gap-3 mb-4">
                      {result.steps[currentStep] &&
                        Object.entries(
                          result.steps[currentStep].guess_data
                        ).map(([attr, value]) => {
                          const feedback =
                            result.steps[currentStep].feedback[attr];
                          return (
                            <div
                              key={attr}
                              className="p-3 bg-gray-50 rounded-lg border border-gray-200"
                            >
                              <div className="flex items-center justify-between mb-1">
                                <span className="text-xs text-gray-500 font-medium">
                                  {attr}
                                </span>
                                <span
                                  className={`w-6 h-6 rounded-full ${getFeedbackColor(
                                    feedback
                                  )} flex items-center justify-center text-white text-xs font-bold`}
                                >
                                  {getFeedbackLabel(feedback)}
                                </span>
                              </div>
                              <p className="text-sm font-semibold text-gray-900">
                                {value}
                              </p>
                            </div>
                          );
                        })}
                    </div>

                    {/* Algorithm State */}
                    {result.steps[currentStep]?.algorithm_state && (
                      <div className="p-4 bg-blue-50 rounded-lg border border-blue-200">
                        <p className="text-sm font-medium text-gray-700 mb-2">
                          Algorithm State
                        </p>
<<<<<<< HEAD
                        {result.algorithm === "ASTAR" && result.steps[currentStep]?.algorithm_state.current_node?.path && (
                          <div className="mb-4">
                            <PathVisualizer
                              path={[
                                ...(result.steps[currentStep].algorithm_state.current_node.path || []),
                                result.steps[currentStep].algorithm_state.current_node.pokemon_idx,
                              ]}
                              pokemonList={pokemonList}
                            />
                          </div>
                        )}
                        <div className="w-full">
                          <JsonViewer data={result.steps[currentStep].algorithm_state} />
=======
                        <div className="grid grid-cols-3 gap-3">
                          {Object.entries(
                            result.steps[currentStep].algorithm_state
                          ).map(([key, value]) => (
                            <div key={key}>
                              <span className="text-xs text-gray-500">
                                {key.replace(/_/g, " ")}:
                              </span>
                              <div className="text-sm font-bold text-gray-900">
                                {renderAlgorithmStateValue(key, value)}
                              </div>
                            </div>
                          ))}
>>>>>>> d67f8a73fd7c55a08b0c6a7960031ff938e228fe
                        </div>
                      </div>
                    )}

                    {/* Heuristic Info */}
                    {result.steps[currentStep]?.heuristic_info && (
                      <div className="mt-4 p-4 bg-purple-50 rounded-lg border border-purple-200">
                        <p className="text-sm font-medium text-gray-700 mb-2">
                          Heuristic Info
                        </p>
                        <div className="w-full">
                          <JsonViewer data={result.steps[currentStep].heuristic_info} />
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Pokemon Image & Timeline */}
                  <div className="space-y-4">
                    {result.steps[currentStep]?.image_url && (
                      <div className="bg-white rounded-lg border border-gray-200 p-6 flex items-center justify-center">
                        <img
                          src={result.steps[currentStep].image_url}
                          alt={result.steps[currentStep].guess_name}
                          className="w-48 h-48 object-contain"
                        />
                      </div>
                    )}

                    {/* Timeline */}
                    <div className="bg-white rounded-lg border border-gray-200 p-4 max-h-96 overflow-y-auto">
                      <p className="text-sm font-medium text-gray-700 mb-3">
                        Timeline
                      </p>
                      <div className="space-y-2">
                        {result.steps.map((step, idx) => (
                          <div
                            key={idx}
                            onClick={() => setCurrentStep(idx)}
                            className={`p-3 rounded-lg cursor-pointer transition-colors ${
                              currentStep === idx
                                ? "bg-blue-50 border border-blue-200"
                                : "bg-gray-50 hover:bg-gray-100 border border-gray-200"
                            }`}
                          >
                            <div className="flex items-center justify-between">
                              <div>
                                <p className="text-xs text-gray-500">
                                  #{step.attempt}
                                </p>
                                <p className="text-sm font-semibold text-gray-900">
                                  {step.guess_name}
                                </p>
                              </div>
                              <p className="text-xs text-gray-500">
                                {step.remaining_candidates}
                              </p>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
                {result.algorithm === "GA" &&
                  result.steps[currentStep]?.algorithm_state
                    ?.generation_history && (
                    <div className="col-span-3 mt-6">
                      <GAVisualization
                        generationHistory={
                          result.steps[currentStep].algorithm_state
                            .generation_history
                        }
                      />
                    </div>
                  )}
                {/* {result.algorithm === "ASTAR" &&
                  result.steps[currentStep]?.algorithm_state && (
                    <div className="col-span-3 mt-6">
                      <AStarVisualization
                        algorithmState={
                          result.steps[currentStep].algorithm_state
                        }
                        allSteps={result.steps}
                        currentStepIndex={currentStep}
                      />
                    </div>
                  )} */}
              </div>
            ) : (
              <div className="h-full flex items-center justify-center">
                <div className="text-center">
                  <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
                    <Play className="w-8 h-8 text-gray-400" />
                  </div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">
                    Ready to solve
                  </h3>
                  <p className="text-sm text-gray-500">
                    Configure your settings and click "Run Solver" to begin
                  </p>
                </div>
              </div>
            )
          ) : // Compare Tab
          compareResults ? (
            <div className="space-y-6">
              {/* Winner Banner */}
              {compareResults.winner && (
                <div className="bg-gradient-to-r from-green-50 to-emerald-50 rounded-lg border border-green-200 p-6">
                  <div className="flex items-center gap-4">
                    <div className="p-3 bg-green-100 rounded-full">
                      <Target className="w-8 h-8 text-green-600" />
                    </div>
                    <div>
                      <p className="text-sm text-gray-600 font-medium mb-1">
                        Winner
                      </p>
                      <div className="flex items-center gap-3">
                        <p className="text-3xl font-bold text-gray-900">
                          {compareResults.winner}
                        </p>
                        <span className="px-3 py-1 bg-green-600 text-white text-sm font-medium rounded-full">
                          Best Performance
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Secret Pokemon */}
              <div className="bg-white rounded-lg border border-gray-200 p-4">
                <p className="text-sm text-gray-500 font-medium mb-1">
                  Secret Pokemon
                </p>
                <p className="text-xl font-bold text-gray-900">
                  {compareResults.secret_pokemon}
                </p>
              </div>

              {/* Comparison Grid */}
              <div className="grid grid-cols-2 gap-6">
                {Object.entries(compareResults.results).map(
                  ([algo, data]: [string, any]) => {
                    const color = getAlgorithmColor(algo);
                    const hasError = "error" in data;

                    return (
                      <div
                        key={algo}
                        className={`bg-white rounded-lg border-2 p-6 ${
                          algo === compareResults.winner
                            ? "border-green-500 shadow-lg"
                            : "border-gray-200"
                        }`}
                      >
                        <div className="flex items-center justify-between mb-4">
                          <div className="flex items-center gap-3">
                            <div className={`p-2 bg-${color}-100 rounded-lg`}>
                              {getAlgorithmIcon(algo)}
                            </div>
                            <h3 className="text-xl font-bold text-gray-900">
                              {algo}
                            </h3>
                          </div>
                          {algo === compareResults.winner && (
                            <span className="px-2 py-1 bg-green-100 text-green-700 text-xs font-medium rounded">
                              Winner
                            </span>
                          )}
                        </div>

                        {hasError ? (
                          <div className="p-4 bg-red-50 rounded-lg border border-red-200">
                            <p className="text-sm text-red-600">
                              Error: {data.error}
                            </p>
                          </div>
                        ) : (
                          <div className="space-y-4">
                            <div className="grid grid-cols-2 gap-4">
                              <div className="p-3 bg-gray-50 rounded-lg">
                                <p className="text-xs text-gray-500 font-medium mb-1">
                                  Status
                                </p>
                                <p
                                  className={`text-sm font-bold ${
                                    data.success
                                      ? "text-green-600"
                                      : "text-red-600"
                                  }`}
                                >
                                  {data.success ? "Success ✓" : "Failed ✗"}
                                </p>
                              </div>
                              <div className="p-3 bg-gray-50 rounded-lg">
                                <p className="text-xs text-gray-500 font-medium mb-1">
                                  Attempts
                                </p>
                                <p className="text-sm font-bold text-gray-900">
                                  {data.attempts}
                                </p>
                              </div>
                              <div className="p-3 bg-gray-50 rounded-lg">
                                <p className="text-xs text-gray-500 font-medium mb-1">
                                  Time (s)
                                </p>
                                <p className="text-sm font-bold text-gray-900">
                                  {data.time.toFixed(2)}
                                </p>
                              </div>
                              <div className="p-3 bg-gray-50 rounded-lg">
                                <p className="text-xs text-gray-500 font-medium mb-1">
                                  Efficiency
                                </p>
                                <p className="text-sm font-bold text-gray-900">
                                  {data.metrics
                                    ? (data.metrics.efficiency * 100).toFixed(
                                        0
                                      ) + "%"
                                    : "N/A"}
                                </p>
                              </div>
                            </div>

                            {/* Performance Bar */}
                            <div className="pt-2">
                              <div className="flex items-center justify-between mb-1">
                                <span className="text-xs text-gray-500">
                                  Performance Score
                                </span>
                                <span className="text-xs font-bold text-gray-700">
                                  {data.attempts
                                    ? Math.max(0, 100 - data.attempts * 10)
                                    : 0}
                                  /100
                                </span>
                              </div>
                              <div className="w-full bg-gray-200 rounded-full h-2">
                                <div
                                  className={`bg-${color}-500 h-2 rounded-full transition-all duration-500`}
                                  style={{
                                    width: `${
                                      data.attempts
                                        ? Math.max(0, 100 - data.attempts * 10)
                                        : 0
                                    }%`,
                                  }}
                                />
                              </div>
                            </div>
                          </div>
                        )}
                      </div>
                    );
                  }
                )}
              </div>

              {/* Detailed Metrics Table */}
              <div className="bg-white rounded-lg border border-gray-200 p-6">
                <h3 className="text-lg font-bold text-gray-900 mb-4">
                  Detailed Comparison
                </h3>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-gray-200">
                        <th className="text-left py-3 px-4 font-medium text-gray-700">
                          Algorithm
                        </th>
                        <th className="text-center py-3 px-4 font-medium text-gray-700">
                          Success
                        </th>
                        <th className="text-center py-3 px-4 font-medium text-gray-700">
                          Attempts
                        </th>
                        <th className="text-center py-3 px-4 font-medium text-gray-700">
                          Time (s)
                        </th>
                        <th className="text-center py-3 px-4 font-medium text-gray-700">
                          Avg Time/Guess
                        </th>
                        <th className="text-center py-3 px-4 font-medium text-gray-700">
                          Efficiency
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(compareResults.results)
                        .filter(
                          ([_, data]: [string, any]) => !("error" in data)
                        )
                        .sort((a: any, b: any) => a[1].attempts - b[1].attempts)
                        .map(([algo, data]: [string, any]) => (
                          <tr
                            key={algo}
                            className={`border-b border-gray-100 ${
                              algo === compareResults.winner
                                ? "bg-green-50"
                                : ""
                            }`}
                          >
                            <td className="py-3 px-4">
                              <div className="flex items-center gap-2">
                                {getAlgorithmIcon(algo)}
                                <span className="font-semibold text-gray-900">
                                  {algo}
                                </span>
                                {algo === compareResults.winner && (
                                  <span className="text-xs text-green-600">
                                    👑
                                  </span>
                                )}
                              </div>
                            </td>
                            <td className="text-center py-3 px-4">
                              <span
                                className={`px-2 py-1 rounded text-xs font-medium ${
                                  data.success
                                    ? "bg-green-100 text-green-700"
                                    : "bg-red-100 text-red-700"
                                }`}
                              >
                                {data.success ? "Yes" : "No"}
                              </span>
                            </td>
                            <td className="text-center py-3 px-4 font-semibold text-gray-900">
                              {data.attempts}
                            </td>
                            <td className="text-center py-3 px-4 text-gray-700">
                              {data.time.toFixed(2)}
                            </td>
                            <td className="text-center py-3 px-4 text-gray-700">
                              {data.metrics
                                ? data.metrics.avg_time_per_guess.toFixed(2)
                                : "N/A"}
                            </td>
                            <td className="text-center py-3 px-4">
                              <span className="font-semibold text-gray-900">
                                {data.metrics
                                  ? (data.metrics.efficiency * 100).toFixed(0) +
                                    "%"
                                  : "N/A"}
                              </span>
                            </td>
                          </tr>
                        ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Insights */}
              <div className="bg-blue-50 rounded-lg border border-blue-200 p-6">
                <div className="flex items-start gap-3">
                  <Info className="w-5 h-5 text-blue-600 mt-0.5 flex-shrink-0" />
                  <div>
                    <h4 className="font-semibold text-gray-900 mb-2">
                      Performance Insights
                    </h4>
                    <ul className="space-y-1 text-sm text-gray-700">
                      <li>
                        • <strong>CSP</strong>: Fast and optimal for
                        well-constrained problems with AC-3
                      </li>
                      <li>
                        • <strong>GA</strong>: Good for complex search spaces
                        with population diversity
                      </li>
                      <li>
                        • <strong>A*</strong>: Guaranteed optimal with
                        admissible heuristic
                      </li>
                      <li>
                        • <strong>SA</strong>: Escapes local optima through
                        probabilistic acceptance
                      </li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="h-full flex items-center justify-center">
              <div className="text-center max-w-md">
                <div className="w-16 h-16 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-4">
                  <GitCompare className="w-8 h-8 text-purple-600" />
                </div>
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  Compare All Algorithms
                </h3>
                <p className="text-sm text-gray-500 mb-4">
                  Run all available algorithms on the same Pokemon to compare
                  their performance, speed, and efficiency.
                </p>
                <div className="flex flex-wrap justify-center gap-2">
                  {availableAlgorithms.map((algo) => {
                    const color = getAlgorithmColor(algo);
                    return (
                      <div
                        key={algo}
                        className={`flex items-center gap-2 px-3 py-2 bg-${color}-50 border border-${color}-200 rounded-lg`}
                      >
                        {getAlgorithmIcon(algo)}
                        <span className="text-sm font-medium text-gray-700">
                          {algo}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}