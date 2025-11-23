"use client";

import React from "react";
import { ChevronRight } from "lucide-react";

interface Pokemon {
  name: string;
  image_url: string;
}

interface PathVisualizerProps {
  path: number[];
  pokemonList: Pokemon[];
}

const PathVisualizer: React.FC<PathVisualizerProps> = ({ path, pokemonList }) => {
  if (!path || path.length === 0) {
    return (
      <div className="text-sm text-gray-500 italic">
        Path will be shown here after the first guess.
      </div>
    );
  }

  return (
    <div>
      <p className="text-sm font-medium text-gray-700 mb-2">Search Path</p>
      <div className="flex items-center space-x-2 overflow-x-auto p-2 bg-gray-100 rounded-lg">
        {path.map((pokemonId, index) => {
          const pokemon = pokemonList[pokemonId];
          if (!pokemon) return null;

          return (
            <React.Fragment key={index}>
              <div className="flex flex-col items-center flex-shrink-0">
                <img
                  src={pokemon.image_url}
                  alt={pokemon.name}
                  className="w-16 h-16 bg-white rounded-md p-1 border"
                />
                <span className="text-xs mt-1 font-medium text-gray-800">
                  {pokemon.name}
                </span>
              </div>
              {index < path.length - 1 && (
                <ChevronRight className="w-6 h-6 text-gray-400 flex-shrink-0" />
              )}
            </React.Fragment>
          );
        })}
      </div>
    </div>
  );
};

export default PathVisualizer;
