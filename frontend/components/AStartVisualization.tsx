"use client";

import React, { useState, useEffect, useRef } from 'react';
import { Activity, Zap, Target, TrendingDown, ZoomIn, ZoomOut, Maximize2 } from 'lucide-react';
import * as d3 from 'd3';

interface AStarVisualizationProps {
  algorithmState: {
    open_set_size?: number;
    closed_set_size?: number;
    candidates?: number;
    g_cost?: number;
    h_cost?: number;
    f_cost?: number;
    goal_state?: boolean;
    open_set_nodes?: any[];
    closed_set_nodes?: any[];
    current_node?: any;
  };
  allSteps: SolverStep[];
  currentStepIndex: number;
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

interface AStarNode {
  pokemon_idx: number;
  pokemon_name: string;
  g_cost: number;
  h_cost: number;
  f_cost: number;
  parent?: number;
  state: 'open' | 'closed' | 'current' | 'goal';
  children?: AStarNode[];
}

interface D3Node extends d3.HierarchyPointNode<AStarNode> {
  data: AStarNode;
}

export default function AStarVisualization({ 
  algorithmState, 
  allSteps,
  currentStepIndex 
}: AStarVisualizationProps) {
  const [zoom, setZoom] = useState(0.8);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const svgRef = useRef<SVGSVGElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [treeData, setTreeData] = useState<D3Node[]>([]);
  const [links, setLinks] = useState<Array<{source: D3Node, target: D3Node}>>([]);
  const [nodeCount, setNodeCount] = useState({ open: 0, closed: 0, total: 0 });

  useEffect(() => {
    // Build comprehensive node list from ALL steps up to current
    const nodeMap = new Map<string, AStarNode>();
    const pathNodes = new Set<string>();
    
    // 1. Add all nodes from the actual path (closed set)
    for (let i = 0; i <= currentStepIndex; i++) {
      const step = allSteps[i];
      const state = step.algorithm_state;
      
      pathNodes.add(step.guess_name);
      
      if (!nodeMap.has(step.guess_name)) {
        nodeMap.set(step.guess_name, {
          pokemon_idx: i,
          pokemon_name: step.guess_name,
          g_cost: state?.g_cost || i,
          h_cost: state?.h_cost || 0,
          f_cost: state?.f_cost || i,
          state: i === currentStepIndex ? 'current' : 
                 state?.goal_state ? 'goal' : 'closed',
          parent: i > 0 ? i - 1 : undefined
        });
      }
    }
    
    // 2. Add ALL open set nodes from current step
    const currentState = allSteps[currentStepIndex]?.algorithm_state;
    if (currentState?.open_set_nodes) {
      currentState.open_set_nodes.forEach((openNode: any) => {
        const key = `${openNode.pokemon_name}-${openNode.pokemon_idx}`;
        if (!nodeMap.has(key) && !pathNodes.has(openNode.pokemon_name)) {
          nodeMap.set(key, {
            pokemon_idx: openNode.pokemon_idx,
            pokemon_name: openNode.pokemon_name,
            g_cost: openNode.g_cost,
            h_cost: openNode.h_cost,
            f_cost: openNode.f_cost,
            state: 'open',
            parent: openNode.parent_idx
          });
        }
      });
    }
    
    // 3. Also add open set nodes from previous steps for complete tree
    for (let i = 0; i < currentStepIndex; i++) {
      const state = allSteps[i]?.algorithm_state;
      if (state?.open_set_nodes) {
        state.open_set_nodes.forEach((openNode: any) => {
          const key = `${openNode.pokemon_name}-${openNode.pokemon_idx}`;
          if (!nodeMap.has(key) && !pathNodes.has(openNode.pokemon_name)) {
            nodeMap.set(key, {
              pokemon_idx: openNode.pokemon_idx,
              pokemon_name: openNode.pokemon_name,
              g_cost: openNode.g_cost,
              h_cost: openNode.h_cost,
              f_cost: openNode.f_cost,
              state: 'open',
              parent: openNode.parent_idx
            });
          }
        });
      }
    }
    
    const nodeList = Array.from(nodeMap.values());
    
    // Build tree structure
    const nodesByIdx = new Map<number, AStarNode>();
    nodeList.forEach(node => {
      nodesByIdx.set(node.pokemon_idx, { ...node, children: [] });
    });
    
    // Find roots and build children arrays
    const roots: AStarNode[] = [];
    nodeList.forEach(node => {
      const nodeWithChildren = nodesByIdx.get(node.pokemon_idx)!;
      if (node.parent !== undefined && nodesByIdx.has(node.parent)) {
        const parent = nodesByIdx.get(node.parent)!;
        if (!parent.children) parent.children = [];
        parent.children.push(nodeWithChildren);
      } else {
        roots.push(nodeWithChildren);
      }
    });
    
    // Sort children by f_cost
    nodesByIdx.forEach(node => {
      if (node.children) {
        node.children.sort((a, b) => a.f_cost - b.f_cost);
      }
    });
    roots.sort((a, b) => a.f_cost - b.f_cost);
    
    // Use D3 tree layout for EACH root (handle forest)
    const allNodes: D3Node[] = [];
    const allLinks: Array<{source: D3Node, target: D3Node}> = [];
    let xOffset = 0;
    
    roots.forEach((root, rootIdx) => {
      const hierarchy = d3.hierarchy(root, d => d.children);
      
      // Create tree layout with generous spacing
      const treeLayout = d3.tree<AStarNode>()
        .size([800, 600])
        .separation((a, b) => {
          // More space between siblings, even more between cousins
          return a.parent === b.parent ? 1.5 : 2;
        });
      
      const treeNodes = treeLayout(hierarchy);
      
      // Adjust positions for this root tree
      treeNodes.descendants().forEach((node: any) => {
        node.x += xOffset;
        allNodes.push(node);
      });
      
      treeNodes.links().forEach((link: any) => {
        allLinks.push(link);
      });
      
      // Calculate width of this tree for next offset
      const treeWidth = Math.max(...treeNodes.descendants().map((n: any) => n.x)) - 
                        Math.min(...treeNodes.descendants().map((n: any) => n.x));
      xOffset += treeWidth + 200; // Add spacing between separate trees
    });
    
    setTreeData(allNodes);
    setLinks(allLinks);
    setNodeCount({
      open: nodeList.filter(n => n.state === 'open').length,
      closed: nodeList.filter(n => n.state === 'closed').length,
      total: nodeList.length
    });
  }, [allSteps, currentStepIndex]);

  const getNodeColor = (state: string) => {
    switch (state) {
      case 'current': return '#3b82f6';
      case 'goal': return '#10b981';
      case 'closed': return '#6b7280';
      case 'open': return '#a78bfa';
      default: return '#9ca3af';
    }
  };

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    setZoom(prev => Math.min(Math.max(0.1, prev * delta), 3));
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging) {
      setPan({ x: e.clientX - dragStart.x, y: e.clientY - dragStart.y });
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const resetView = () => {
    setZoom(0.8);
    setPan({ x: 0, y: 0 });
  };

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-purple-100 rounded-lg">
            <Activity className="w-5 h-5 text-purple-600" />
          </div>
          <div>
            <h3 className="text-lg font-bold text-gray-900">A* Search Tree - Complete Graph</h3>
            <p className="text-sm text-gray-500">
              Showing {nodeCount.total} total nodes ({nodeCount.open} in open set, {nodeCount.closed} explored)
            </p>
          </div>
        </div>
        
        {/* Controls */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => setZoom(z => Math.min(z * 1.2, 3))}
            className="p-2 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
            title="Zoom In"
          >
            <ZoomIn className="w-4 h-4" />
          </button>
          <button
            onClick={() => setZoom(z => Math.max(z * 0.8, 0.1))}
            className="p-2 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
            title="Zoom Out"
          >
            <ZoomOut className="w-4 h-4" />
          </button>
          <button
            onClick={resetView}
            className="p-2 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
            title="Reset View"
          >
            <Maximize2 className="w-4 h-4" />
          </button>
          <div className="ml-2 px-3 py-1 bg-blue-100 text-blue-700 text-sm font-medium rounded">
            Zoom: {(zoom * 100).toFixed(0)}%
          </div>
        </div>
      </div>

      {/* Cost Metrics */}
      <div className="grid grid-cols-4 gap-4 mb-6">
        <div className="p-4 bg-gradient-to-br from-blue-50 to-blue-100 rounded-lg border border-blue-200">
          <div className="flex items-center justify-between mb-1">
            <span className="text-xs font-medium text-blue-700">g(n) - Path Cost</span>
            <Activity className="w-4 h-4 text-blue-600" />
          </div>
          <p className="text-2xl font-bold text-blue-900">
            {algorithmState.g_cost?.toFixed(1) || '0'}
          </p>
          <p className="text-xs text-blue-600 mt-1">Guesses so far</p>
        </div>

        <div className="p-4 bg-gradient-to-br from-purple-50 to-purple-100 rounded-lg border border-purple-200">
          <div className="flex items-center justify-between mb-1">
            <span className="text-xs font-medium text-purple-700">h(n) - Heuristic</span>
            <TrendingDown className="w-4 h-4 text-purple-600" />
          </div>
          <p className="text-2xl font-bold text-purple-900">
            {algorithmState.h_cost?.toFixed(2) || '0'}
          </p>
          <p className="text-xs text-purple-600 mt-1">Est. remaining</p>
        </div>

        <div className="p-4 bg-gradient-to-br from-orange-50 to-orange-100 rounded-lg border border-orange-200">
          <div className="flex items-center justify-between mb-1">
            <span className="text-xs font-medium text-orange-700">f(n) = g+h</span>
            <Zap className="w-4 h-4 text-orange-600" />
          </div>
          <p className="text-2xl font-bold text-orange-900">
            {algorithmState.f_cost?.toFixed(2) || '0'}
          </p>
          <p className="text-xs text-orange-600 mt-1">Total cost</p>
        </div>

        <div className="p-4 bg-gradient-to-br from-green-50 to-green-100 rounded-lg border border-green-200">
          <div className="flex items-center justify-between mb-1">
            <span className="text-xs font-medium text-green-700">Open Set</span>
            <Target className="w-4 h-4 text-green-600" />
          </div>
          <p className="text-2xl font-bold text-green-900">
            {algorithmState.open_set_size || 0}
          </p>
          <p className="text-xs text-green-600 mt-1">Nodes to explore</p>
        </div>
      </div>

      {/* Legend */}
      <div className="flex items-center justify-center gap-6 mb-4 p-3 bg-gray-50 rounded-lg">
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-blue-500 rounded-full" />
          <span className="text-xs font-medium text-gray-700">Current Node</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-gray-500 rounded-full" />
          <span className="text-xs font-medium text-gray-700">Explored (Closed Set)</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-purple-400 rounded-full" />
          <span className="text-xs font-medium text-gray-700">Not Chosen (Open Set)</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-green-500 rounded-full" />
          <span className="text-xs font-medium text-gray-700">Goal Found!</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-8 h-1 bg-yellow-500" />
          <span className="text-xs font-medium text-gray-700">Path Taken</span>
        </div>
      </div>

      {/* Interactive Graph */}
      <div 
        className="relative bg-gradient-to-br from-gray-50 to-slate-100 rounded-lg border-2 border-gray-300 overflow-hidden cursor-move"
        style={{ height: '800px' }}
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      >
        <svg
          ref={svgRef}
          className="w-full h-full"
          style={{ 
            cursor: isDragging ? 'grabbing' : 'grab',
          }}
        >
          <defs>
            <pattern id="smallGrid" width="20" height="20" patternUnits="userSpaceOnUse">
              <path d="M 20 0 L 0 0 0 20" fill="none" stroke="gray" strokeWidth="0.5" opacity="0.2"/>
            </pattern>
            <pattern id="grid" width="100" height="100" patternUnits="userSpaceOnUse">
              <rect width="100" height="100" fill="url(#smallGrid)"/>
              <path d="M 100 0 L 0 0 0 100" fill="none" stroke="gray" strokeWidth="1" opacity="0.3"/>
            </pattern>
            <marker id="arrowhead" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
              <polygon points="0 0, 10 3, 0 6" fill="#f59e0b" />
            </marker>
          </defs>
          
          <rect width="100%" height="100%" fill="url(#grid)" />
          
          <g transform={`translate(${pan.x + 100}, ${pan.y + 50}) scale(${zoom})`}>
            {/* Draw edges */}
            {links.map((link, i) => {
              const source = link.source;
              const target = link.target;
              const isOnPath = target.data.state === 'closed' || target.data.state === 'current' || target.data.state === 'goal';
              
              return (
                <g key={`link-${i}`}>
                  {isOnPath && (
                    <line
                      x1={source.x}
                      y1={source.y}
                      x2={target.x}
                      y2={target.y}
                      stroke="#fbbf24"
                      strokeWidth="8"
                      opacity="0.3"
                    />
                  )}
                  <line
                    x1={source.x}
                    y1={source.y}
                    x2={target.x}
                    y2={target.y}
                    stroke={isOnPath ? '#f59e0b' : getNodeColor(target.data.state)}
                    strokeWidth={isOnPath ? 3 : 1.5}
                    opacity={target.data.state === 'open' ? 0.3 : 0.6}
                    markerEnd={isOnPath ? "url(#arrowhead)" : ""}
                  />
                </g>
              );
            })}
            
            {/* Draw nodes */}
            {treeData.map((node, i) => {
              const isOnPath = node.data.state === 'closed' || node.data.state === 'current' || node.data.state === 'goal';
              const radius = node.data.state === 'open' ? 18 : 25;
              
              return (
                <g key={`node-${i}`}>
                  {/* Glow for path nodes */}
                  {isOnPath && (
                    <circle
                      cx={node.x}
                      cy={node.y}
                      r={radius + 10}
                      fill={getNodeColor(node.data.state)}
                      opacity="0.2"
                      filter="blur(8px)"
                    />
                  )}
                  
                  {/* Main node circle */}
                  <circle
                    cx={node.x}
                    cy={node.y}
                    r={radius}
                    fill={getNodeColor(node.data.state)}
                    stroke={isOnPath ? '#fbbf24' : 'white'}
                    strokeWidth={isOnPath ? 3 : 2}
                    opacity={node.data.state === 'open' ? 0.6 : 1}
                    className="transition-all duration-300"
                  >
                    {node.data.state === 'current' && (
                      <animate
                        attributeName="r"
                        values={`${radius};${radius + 5};${radius}`}
                        dur="1.5s"
                        repeatCount="indefinite"
                      />
                    )}
                  </circle>
                  
                  {/* F-cost badge */}
                  <circle
                    cx={node.x + radius - 5}
                    cy={node.y - radius + 5}
                    r="15"
                    fill={node.data.state === 'current' ? '#fbbf24' : 
                          node.data.state === 'goal' ? '#10b981' : 
                          node.data.state === 'open' ? '#a78bfa' : '#6b7280'}
                    stroke="white"
                    strokeWidth="2"
                  />
                  <text
                    x={node.x + radius - 5}
                    y={node.y - radius + 5}
                    textAnchor="middle"
                    dominantBaseline="middle"
                    fill="white"
                    fontSize="10"
                    fontWeight="bold"
                  >
                    {node.data.f_cost.toFixed(1)}
                  </text>
                  
                  {/* Pokemon name */}
                  <text
                    x={node.x}
                    y={node.y + radius + 20}
                    textAnchor="middle"
                    fill={isOnPath ? '#1f2937' : '#6b7280'}
                    fontSize="12"
                    fontWeight={isOnPath ? 'bold' : 'normal'}
                  >
                    {node.data.pokemon_name}
                  </text>
                  
                  {/* Detailed tooltip on hover */}
                  <g opacity="0" className="hover:opacity-100 transition-opacity">
                    <rect
                      x={node.x - 80}
                      y={node.y - radius - 80}
                      width="160"
                      height="70"
                      fill="#1f2937"
                      rx="8"
                      stroke="white"
                      strokeWidth="2"
                    />
                    <text x={node.x} y={node.y - radius - 55} textAnchor="middle" fill="#fbbf24" fontSize="11" fontWeight="bold">
                      {node.data.pokemon_name}
                    </text>
                    <text x={node.x - 70} y={node.y - radius - 38} fill="#9ca3af" fontSize="10">
                      g(n): <tspan fill="#60a5fa" fontWeight="bold">{node.data.g_cost.toFixed(1)}</tspan>
                    </text>
                    <text x={node.x - 70} y={node.y - radius - 24} fill="#9ca3af" fontSize="10">
                      h(n): <tspan fill="#a78bfa" fontWeight="bold">{node.data.h_cost.toFixed(2)}</tspan>
                    </text>
                    <text x={node.x - 70} y={node.y - radius - 10} fill="#9ca3af" fontSize="10">
                      f(n): <tspan fill="#fbbf24" fontWeight="bold">{node.data.f_cost.toFixed(2)}</tspan>
                    </text>
                  </g>
                </g>
              );
            })}
          </g>
        </svg>
      </div>

      {/* Instructions */}
      <div className="mt-4 p-3 bg-blue-50 rounded-lg border border-blue-200">
        <p className="text-xs text-gray-700">
          <strong>💡 Interactive Controls:</strong> Scroll to zoom • Click and drag to pan • Hover over nodes for details • Reset view with the button above
        </p>
      </div>

      {/* Explanation */}
      <div className="mt-4 p-4 bg-purple-50 rounded-lg border border-purple-200">
        <h4 className="text-sm font-semibold text-gray-900 mb-2 flex items-center gap-2">
          <Activity className="w-4 h-4 text-purple-600" />
          Understanding the Graph
        </h4>
        <div className="space-y-2 text-xs text-gray-700">
          <p>
            <strong className="text-purple-700">This shows EVERY node A* evaluated!</strong> The graph displays all {nodeCount.total} nodes in the search tree using D3's tree layout.
          </p>
          <p>
            <strong className="text-gray-700">Gray path:</strong> The nodes A* chose (lowest f(n) at each step)
          </p>
          <p>
            <strong className="text-purple-700">Purple nodes:</strong> Candidates in open set with HIGHER f(n) values - these were rejected!
          </p>
          <p className="pt-2 border-t border-purple-200">
            <strong>Why A* is optimal:</strong> By always choosing the node with lowest f(n) = g(n) + h(n), A* finds the shortest path to the goal!
          </p>
        </div>
      </div>
    </div>
  );
}