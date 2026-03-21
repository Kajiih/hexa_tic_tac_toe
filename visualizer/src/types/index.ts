export type Player = 1 | 2;
export type Coord = [number, number];

export interface GameData {
    radius: number;
    history: Coord[];
    winner: Player | null;
}

export type BoardState = Record<string, Player>;
