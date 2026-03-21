import React, { useState, useEffect, useMemo } from "react";
import HexCell from "./components/HexCell";
import { BoardState, Coord, GameData, Player } from "./types";
import "./App.css";

/**
 * Calculates the player for a given move index based on the 1-2-2 game rules.
 */
const getPlayerForIndex = (index: number): Player => {
    if (index === 0) return 1;
    return Math.floor((index + 1) / 2) % 2 === 1 ? 2 : 1;
};

export default function App() {
    const [radius, setRadius] = useState<number>(5);
    const [history, setHistory] = useState<Coord[]>([]);
    const [currentIndex, setCurrentIndex] = useState<number>(-1);
    const [isPlaying, setIsPlaying] = useState<boolean>(false);
    const [winner, setWinner] = useState<Player | null>(null);

    // Fetch Game Data
    useEffect(() => {
        fetch("/game_data.json")
            .then((res) => res.json())
            .then((data: GameData) => {
                setRadius(data.radius || 5);
                setHistory(data.history || []);
                setWinner(data.winner || null);
                setCurrentIndex(-1);
            })
            .catch((err) => {
                console.warn("No game_data.json found, using sample.", err);
                const sampleMoves: Coord[] = [
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
    const boardState = useMemo<BoardState>(() => {
        const state: BoardState = {};
        for (let i = 0; i <= currentIndex; i++) {
            const [q, r] = history[i];
            state[`${q},${r}`] = getPlayerForIndex(i);
        }
        return state;
    }, [history, currentIndex]);

    const allCoords = useMemo(() => {
        const coords: { q: number; r: number }[] = [];
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
        if (!isPlaying) return;

        // Stop if we're already at the end
        if (currentIndex >= history.length - 1) {
            setIsPlaying(false);
            return;
        }

        const interval = setInterval(() => {
            setCurrentIndex((prev) => {
                const next = prev + 1;
                if (next >= history.length - 1) {
                    setIsPlaying(false);
                }
                return next;
            });
        }, 300); // Slightly faster for better feel

        return () => clearInterval(interval);
    }, [isPlaying, history.length]);

    // Dynamic ViewBox calculation
    const viewSize = useMemo(() => radius * 60, [radius]);
    const viewBox = `-${viewSize / 2} -${viewSize / 2} ${viewSize} ${viewSize}`;

    const currentTurn = useMemo(() => {
        if (currentIndex >= history.length - 1) return null;
        return getPlayerForIndex(currentIndex + 1);
    }, [currentIndex, history.length]);

    return (
        <div className='dashboard'>
            <header>
                <h1>Hexagonal Tic-Tac-Toe Visualizer</h1>
                <div className='status-bar'>
                    <div className='stats-group'>
                        <span>Radius: {radius}</span>
                        <span>Moves: {history.length}</span>
                    </div>

                    <div className='turn-indicator'>
                        {winner && currentIndex === history.length - 1 ?
                            <span className={`winner p${winner}`}>🏆 Player {winner} Wins!</span>
                        :   currentTurn && <span className={`turn p${currentTurn}`}>Turn: Player {currentTurn}</span>}
                    </div>

                    <div className='player-indicators'>
                        <div className={`player-indicator p1 ${currentTurn === 1 ? "active" : ""}`}>P1: Blue</div>
                        <div className={`player-indicator p2 ${currentTurn === 2 ? "active" : ""}`}>P2: Rose</div>
                    </div>
                </div>
            </header>

            <main className='board-container'>
                <svg viewBox={viewBox} width='100%' height='100%' preserveAspectRatio='xMidYMid meet'>
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
                <div className='timeline-container'>
                    <input
                        type='range'
                        className='timeline'
                        min='-1'
                        max={history.length - 1}
                        value={currentIndex}
                        onChange={(e) => setCurrentIndex(parseInt(e.target.value))}
                    />
                </div>

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
