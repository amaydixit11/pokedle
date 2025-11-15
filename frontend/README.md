
---

## 3. Frontend README.md

```markdown
# Pokedle Solver - Frontend

Modern Next.js-based web interface for visualizing and comparing AI algorithms solving the Pokedle game.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Architecture](#architecture)
- [Components](#components)
- [Styling](#styling)
- [Development](#development)
- [Deployment](#deployment)

---

## 🎯 Overview

Interactive web application built with Next.js 16 and React 19 that provides real-time visualization of AI algorithms solving Pokedle puzzles.

### Key Features

- ✅ **Algorithm Configuration**: Intuitive UI for all algorithm parameters
- ✅ **Step-by-Step Playback**: Navigate through solution process
- ✅ **Real-time Visualizations**: D3.js graphs for GA and A*
- ✅ **Algorithm Comparison**: Side-by-side performance metrics
- ✅ **Responsive Design**: Works on desktop, tablet, and mobile
- ✅ **Dark Mode Ready**: Theme-aware components

---

## 💻 Installation

### Prerequisites

- Node.js 18.0 or higher
- npm or yarn

### Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install
# or
yarn install

# Run development server
npm run dev
# or
yarn dev

# Build for production
npm run build
npm start
```

### Environment Variables

Create `.env.local`:

```bash
# API endpoint (optional, defaults to localhost:8000)
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## 🏗️ Architecture

```
frontend/
├── app/
│   ├── layout.tsx           # Root layout with fonts
│   ├── page.tsx             # Home page (renders main component)
│   └── globals.css          # Global styles
├── components/
│   ├── main2.tsx            # Main visualizer component
│   ├── GAVisualization.tsx  # GA generation tracker
│   └── AStarVisualization.tsx  # A* search tree
├── public/                  # Static assets
├── package.json             # Dependencies
├── tsconfig.json            # TypeScript config
├── next.config.ts           # Next.js config
└── tailwind.config.ts       # Tailwind config
```

### Technology Stack

- **Framework**: Next.js 16.0.1 (App Router)
- **UI Library**: React 19.2.0
- **Language**: TypeScript 5
- **Styling**: Tailwind CSS 4
- **Visualizations**: D3.js 7.9.0, Recharts 3.4.1
- **Icons**: Lucide React 0.548.0

---

## 🧩 Components

### 1. Main Visualizer (`components/main2.tsx`)

Primary component managing the entire application state and UI.

**Features:**
- Algorithm selection and configuration
- Attribute selection
- Solver execution
- Result display and navigation
- Algorithm comparison

**State Management:**
```typescript
const [config, setConfig] = useState<SolverConfig>({
  algorithm: 'CSP',
  attributes: ['Generation', 'Type1', 'Type2', 'Color'],
  max_attempts: 10
});

const [result, setResult] = useState<SolverResult | null>(null);
const [currentStep, setCurrentStep] = useState(0);
const [loading, setLoading] = useState(false);
```

**Key Functions:**
```typescript
// Run solver
const runSolver = async () => {
  const response = await fetch(`${API_URL}/solve`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(configToSend)
  });
  const data = await response.json();
  setResult(data);
};

// Compare algorithms
const runComparison = async () => {
  const response = await fetch(`${API_URL}/compare`, {...});
  const data = await response.json();
  setCompareResults(data);
};
```

---

### 2. GA Visualization (`components/GAVisualization.tsx`)

Interactive visualization of genetic algorithm evolution process.

**Features:**
- Generation-by-generation tracking
- Population fitness statistics
- Elite preservation display
- Selection, crossover, mutation details
- Fitness evolution chart

**Props:**
```typescript
interface GAVisualizationProps {
  generationHistory: GenerationData[];
}

interface GenerationData {
  generation_number: number;
  fitness_stats: {
    max: number;
    avg: number;
    median: number;
    min: number;
  };
  elite_preserved: Pokemon[];
  new_population: Pokemon[];
  selection_pairs: SelectionPair[];
  crossover_results: CrossoverResult[];
  mutation_results: MutationResult[];
}
```

**Usage:**
```tsx
<GAVisualization 
  generationHistory={
    result.steps[currentStep].algorithm_state.generation_history
  }
/>
```

---

### 3. A* Visualization (`components/AStarVisualization.tsx`)

Interactive search tree visualization using D3.js.

**Features:**
- Complete search tree display
- Node cost annotations (g, h, f)
- Path highlighting
- Zoom and pan controls
- Open/closed set visualization

**Props:**
```typescript
interface AStarVisualizationProps {
  algorithmState: {
    open_set_size?: number;
    closed_set_size?: number;
    g_cost?: number;
    h_cost?: number;
    f_cost?: number;
    open_set_nodes?: any[];
    closed_set_nodes?: any[];
  };
  allSteps: SolverStep[];
  currentStepIndex: number;
}
```

**D3 Implementation:**
```typescript
// Build tree structure
const hierarchy = d3.hierarchy(root, d => d.children);

// Create tree layout
const treeLayout = d3.tree<AStarNode>()
  .size([800, 600])
  .separation((a, b) => a.parent === b.parent ? 1.5 : 2);

const treeNodes = treeLayout(hierarchy);
```

---

## 🎨 Styling

### Tailwind CSS Configuration

```typescript
// tailwind.config.ts
export default {
  content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        'background': 'var(--background)',
        'foreground': 'var(--foreground)',
      },
    },
  },
};
```

### Global Styles

```css
/* app/globals.css */
@import "tailwindcss";

:root {
  --background: #ffffff;
  --foreground: #171717;
}

@media (prefers-color-scheme: dark) {
  :root {
    --background: #0a0a0a;
    --foreground: #ededed;
  }
}
```

### Component Styling Patterns

**Gradient Backgrounds:**
```tsx
<div className="bg-gradient-to-r from-yellow-50 to-orange-50 rounded-lg border border-yellow-200 p-6">
  {/* Content */}
</div>
```

**Interactive Elements:**
```tsx
<button className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:bg-gray-400">
  Run Solver
</button>
```

**Feedback Colors:**
```tsx
const getFeedbackColor = (status: string) => {
  switch (status) {
    case 'green': return 'bg-green-500';
    case 'yellow': return 'bg-yellow-500';
    case 'gray': return 'bg-gray-400';
    case 'higher': return 'bg-blue-500';
    case 'lower': return 'bg-red-500';
  }
};
```

---

## 🔧 Development

### Running Development Server

```bash
npm run dev
```

Application runs at `http://localhost:3000`

### Building for Production

```bash
# Create optimized production build
npm run build

# Start production server
npm start

# Or export static site
npm run build
npm run export
```

### Linting

```bash
# Run ESLint
npm run lint

# Fix auto-fixable issues
npm run lint -- --fix
```

### Type Checking

```bash
# Check types
npx tsc --noEmit
```

---

## 📱 Responsive Design

### Breakpoints

```typescript
// Tailwind breakpoints
sm: '640px'   // Mobile landscape
md: '768px'   // Tablet
lg: '1024px'  // Desktop
xl: '1280px'  // Large desktop
2xl: '1536px' // Extra large
```

### Responsive Patterns

**Sidebar:**
```tsx
<div className={`${
  sidebarOpen ? 'w-80' : 'w-0'
} lg:relative fixed inset-y-0 left-0 z-30 transition-all`}>
```

**Grid Layouts:**
```tsx
<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
```

**Mobile Menu:**
```tsx
{!sidebarOpen && (
  <button onClick={() => setSidebarOpen(true)} 
          className="lg:hidden p-2">
    <Menu className="w-5 h-5" />
  </button>
)}
```

---

## 🎭 Interactive Features

### Keyboard Navigation

```typescript
useEffect(() => {
  if (!result) return;
  
  const handleKeyPress = (e: KeyboardEvent) => {
    if (e.key === 'ArrowLeft') {
      setCurrentStep(Math.max(0, currentStep - 1));
    } else if (e.key === 'ArrowRight') {
      setCurrentStep(Math.min(result.steps.length - 1, currentStep + 1));
    }
  };
  
  window.addEventListener('keydown', handleKeyPress);
  return () => window.removeEventListener('keydown', handleKeyPress);
}, [result, currentStep]);
```

### Zoom and Pan (A* Visualization)

```typescript
const [zoom, setZoom] = useState(0.8);
const [pan, setPan] = useState({ x: 0, y: 0 });

const handleWheel = (e: React.WheelEvent) => {
  e.preventDefault();
  const delta = e.deltaY > 0 ? 0.9 : 1.1;
  setZoom(prev => Math.min(Math.max(0.1, prev * delta), 3));
};

const handleMouseMove = (e: React.MouseEvent) => {
  if (isDragging) {
    setPan({ x: e.clientX - dragStart.x, y: e.clientY - dragStart.y });
  }
};
```

---

## 🐛 Troubleshooting

### Common Issues

**1. API Connection Refused**
```typescript
// Check API_URL in main2.tsx
const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// Ensure backend is running
// curl http://localhost:8000/health
```

**2. Hydration Errors**
```typescript
// Wrap client-side only code
'use client';

useEffect(() => {
  // Client-side only logic
}, []);
```

**3. Type Errors**
```bash
# Clear Next.js cache
rm -rf .next

# Reinstall dependencies
rm -rf node_modules package-lock.json
npm install
```

**4. Style Not Applying**
```bash
# Rebuild Tailwind
npm run dev

# Check Tailwind content paths in config
content: ['./app/**/*.{js,ts,jsx,tsx}']
```

---

## 📊 Performance Optimization

### Code Splitting

Next.js automatically splits code by route:

```typescript
// Dynamic import for heavy components
const GAVisualization = dynamic(
  () => import('./GAVisualization'),
  { loading: () => <p>Loading...</p> }
);
```

### Memoization

```typescript
import { useMemo, useCallback } from 'react';

const memoizedValue = useMemo(
  () => expensiveCalculation(data),
  [data]
);

const memoizedCallback = useCallback(
  () => { /* function */ },
  [dependency]
);
```

### Image Optimization

```typescript
import Image from 'next/image';

<Image
  src={pokemon.image_url}
  alt={pokemon.name}
  width={200}
  height={200}
  loading="lazy"
/>
```