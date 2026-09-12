export enum MetricType {
  L2 = 0,
  InnerProduct = 1,
  Cosine = 2,
}

export enum QuantType {
  None = 0,
  SQ8 = 1,
  FP16 = 2,
  Bit = 3,
  PQ8 = 4,
}

export interface DatabaseOptions {
  path: string;
  dim: number;
  shards?: number;
  metric?: MetricType;
  quantType?: QuantType;
  memoryBudgetBytes?: number;
}

export interface PutOptions {
  membrane?: string;
  timestamp?: number;
  payload?: Uint8Array | Buffer;
}

export interface SearchOptions {
  filterJson?: string;
  asOfTs?: number;
  asOfLsn?: number;
  membrane?: string;
  startTime?: number;
  endTime?: number;
}

export interface SearchHit {
  id: number;
  score: number;
}

export interface RecordView {
  id: number;
  vector: number[];
  dim: number;
  timestamp: number;
  payload: Buffer | null;
  membrane: string | null;
}

export class Database {
  constructor(handle: any);
  static open(options: DatabaseOptions): Database;
  close(): void;
  put(id: number | bigint, vector: number[], options?: PutOptions): void;
  get(id: number | bigint, membrane?: string | null): RecordView | null;
  exists(id: number | bigint, membrane?: string | null): boolean;
  delete(id: number | bigint, membrane?: string | null): void;
  search(queryVector: number[], topK?: number, options?: SearchOptions): SearchHit[];
  flush(): void;
  freeze(membrane?: string | null): void;
  compact(membrane?: string | null): void;
  createMembrane(name: string, dim: number, shardCount?: number): void;
  dropMembrane(name: string): void;
  openMembrane(name: string): void;
  closeMembrane(name: string): void;
  listMembranes(): string[];
  getStats(): Record<string, any>;
}

export default Database;