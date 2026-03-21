import React, { useState, useEffect, useMemo } from "react";

// --- Constants & Math ---
const HEX_SIZE = 25;
const HEX_WIDTH = Math.sqrt(3) * HEX_SIZE;
const HEX_HEIGHT = 2 * HEX_SIZE;

const getHexPath = () => {
    const points = [];
    for (let i = 0; i < 6; i++) {
        const angle = (Math.PI / 180) * (60 * i + 30);
        points.push(`${HEX_SIZE * Math.cos(angle)},${HEX_SIZE * Math.sin(angle)}`);
    }
    return points.join(" ");
};

const axialToPixel = (q, r) => {
    const x = HEX_SIZE * Math.sqrt(3) * (q + r / 2);
    const y = HEX_SIZE * (3 / 2) * r;
    return { x, y };
};

// --- Components ---
const HexCell = ({ q, r, owner, isLastMove }) => {
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

export default function App() {
    const [radius, setRadius] = useState(5);
    const [history, setHistory] = useState([]);
    const [currentIndex, setCurrentIndex] = useState(-1);
    const [isPlaying, setIsPlaying] = useState(false);

    // Fetch Game Data
    useEffect(() => {
        fetch("/game_data.json")
            .then((res) => res.json())
            .then((data) => {
                setRadius(data.radius || 5);
                setHistory(data.history || []);
                setCurrentIndex(-1);
            })
            .catch((err) => {
                console.log("No game_data.json found, using sample.", err);
                const sampleMoves = [
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [1, -1],
                    [-1, 0],
                    [-1, 1],
                    [0, -1],
                ];
                setHistory(sampleMoves);
            });
    }, []);

    // Board State derived from history
    const boardState = useMemo(() => {
        const state = {};
        for (let i = 0; i <= currentIndex; i++) {
            const [q, r] = history[i];
            // Calculate player: 1 then 2,2, 1,1, 2,2...
            // Turn 1: pos 0 -> P1
            // Turn 2: pos 1, 2 -> P2
            // Turn 3: pos 3, 4 -> P1
            let player = 1;
            if (i > 0) {
                player = Math.floor((i + 1) / 2) % 2 === 1 ? 2 : 1;
            }
            state[`${q},${r}`] = player;
        }
        return state;
    }, [history, currentIndex]);

    const allCoords = useMemo(() => {
        const coords = [];
        for (let q = -radius + 1; q < radius; q++) {
            for (let r = -radius + 1; r < radius; r++) {
                if (Math.abs(q + r) < radius) {
                    coords.push({ q, r });
                }
            }
        }
        return coords;
    }, [radius]);

    // Playback Logic
    useEffect(() => {
        let interval;
        if (isPlaying && currentIndex < history.length - 1) {
            interval = setInterval(() => {
                setCurrentIndex((prev) => prev + 1);
            }, 600);
        } else {
            setIsPlaying(false);
        }
        return () => clearInterval(interval);
    }, [isPlaying, currentIndex, history.length]);

    return (
        <div className='dashboard'>
            <header>
                <h1>Hexagonal Tic-Tac-Toe Visualizer</h1>
                <div className='status-bar'>
                    <span>
                        Radius: {radius} | Moves: {history.length}
                    </span>
                    <div className='player-indicator p1'>Player 1: Blue</div>
                    <div className='player-indicator p2'>Player 2: Rose</div>
                </div>
            </header>

            <main className='board-container'>
                <svg viewBox='-300 -300 600 600' width='100%' height='100%' preserveAspectRatio='xMidYMid meet'>
                    {allCoords.map(({ q, r }) => (
                        <HexCell
                            key={`${q},${r}`}
                            q={q}
                            r={r}
                            owner={boardState[`${q},${r}`]}
                            isLastMove={
                                currentIndex >= 0 && history[currentIndex][0] === q && history[currentIndex][1] === r
                            }
                        />
                    ))}
                </svg>
            </main>

            <footer className='controls-pnl'>
                <input
                    type='range'
                    className='timeline'
                    min='-1'
                    max={history.length - 1}
                    value={currentIndex}
                    onChange={(e) => setCurrentIndex(parseInt(e.target.value))}
                />

                <div className='btn-group'>
                    <button onClick={() => setCurrentIndex(-1)}>⏮ Reset</button>
                    <button onClick={() => setCurrentIndex((prev) => Math.max(-1, prev - 1))}>Step Back</button>
                    <button className='primary' onClick={() => setIsPlaying(!isPlaying)}>
                        {isPlaying ? "⏸ Pause" : "▶ Play"}
                    </button>
                    <button onClick={() => setCurrentIndex((prev) => Math.min(history.length - 1, prev + 1))}>
                        Step Forward
                    </button>
                    <button onClick={() => setCurrentIndex(history.length - 1)}>Done ⏭</button>
                </div>
            </footer>
        </div>
    );
}
