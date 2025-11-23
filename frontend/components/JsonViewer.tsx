"use client";

import React, { useState } from "react";
import { ChevronRight, ChevronDown } from "lucide-react";

interface JsonViewerProps {
  data: any;
  level?: number;
}

const JsonViewer: React.FC<JsonViewerProps> = ({ data, level = 0 }) => {
  if (typeof data !== "object" || data === null) {
    if (typeof data === "string") {
      return <span className="text-green-600">"{data}"</span>;
    }
    if (typeof data === "number") {
      return <span className="text-blue-600">{data}</span>;
    }
    return <span className="text-purple-600">{String(data)}</span>;
  }

  const [isExpanded, setIsExpanded] = useState(level < 1); // Expand first level by default

  const toggleExpand = () => setIsExpanded(!isExpanded);

  const isArray = Array.isArray(data);
  const entries = Object.entries(data);
  const bracketStart = isArray ? "[" : "{";
  const bracketEnd = isArray ? "]" : "}";
  const preview = isArray
    ? `Array(${entries.length})`
    : `Object(${entries.length})`;

  return (
    <div className="font-mono text-sm">
      <div
        className="flex items-center cursor-pointer"
        onClick={toggleExpand}
      >
        {isExpanded ? (
          <ChevronDown className="w-4 h-4 mr-1 flex-shrink-0" />
        ) : (
          <ChevronRight className="w-4 h-4 mr-1 flex-shrink-0" />
        )}
        <span>
          {bracketStart}
          {!isExpanded && <span className="text-gray-500 ml-2">{preview}...</span>}
          {!isExpanded && bracketEnd}
        </span>
      </div>
      {isExpanded && (
        <div className="pl-6 border-l border-gray-200">
          {entries.map(([key, value]) => (
            <div key={key} className="flex">
              <span className="text-red-500 mr-2">
                {isArray ? "" : `"${key}": `}
              </span>
              <JsonViewer data={value} level={level + 1} />
            </div>
          ))}
          <span>{bracketEnd}</span>
        </div>
      )}
    </div>
  );
};

export default JsonViewer;
