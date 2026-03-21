import React, { useMemo } from "react";
import { axialToPixel, getHexPath } from "../utils/hexMath";
import { Player } from "../types";

interface HexCellProps {
    q: number;
    r: number;
    owner?: Player;
    isLastMove?: boolean;
}

const HexCell: React.FC<HexCellProps> = ({ q, r, owner, isLastMove }) => {
    const { x, y } = useMemo(() => axialToPixel(q, r), [q, r]);

    let fill = "#1e293b";
    let stroke = "#334155";
    let strokeWidth = 1;
    let shadow = "none";

    if (owner === 1) {
        fill = "#38bdf8";
        shadow = "drop-shadow(0 0 8px rgba(56, 189, 248, 0.6))";
    } else if (owner === 2) {
        fill = "#f43f5e";
        shadow = "drop-shadow(0 0 8px rgba(244, 63, 94, 0.6))";
    }

    if (isLastMove) {
        strokeWidth = 3;
        stroke = "#ffffff";
    }

    return (
        <g transform={`translate(${x}, ${y})`} style={{ filter: shadow }}>
            <polygon
                points={getHexPath()}
                fill={fill}
                stroke={stroke}
                strokeWidth={strokeWidth}
                style={{ transition: "all 0.3s ease-out" }}
            />
            {owner && (
                <text
                    textAnchor='middle'
                    dy='.3em'
                    fill='white'
                    fontSize='12'
                    fontWeight='bold'
                    style={{ pointerEvents: "none" }}
                >
                    {owner === 1 ? "X" : "O"}
                </text>
            )}
        </g>
    );
};

export default HexCell;
