import type { ChatStreamEvent } from "./types";

export function parseSseChunk<T = ChatStreamEvent>(raw: string): T | null {
  const eventLine = raw.split("\n").find((line) => line.startsWith("event:"));
  const dataLine = raw.split("\n").find((line) => line.startsWith("data:"));
  if (!eventLine || !dataLine) return null;

  const event = eventLine.slice(6).trim();
  const data = JSON.parse(dataLine.slice(5).trim() || "{}");
  return { event, data } as T;
}
