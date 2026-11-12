"use client";

import { useCallback, useEffect, useRef, useState } from "react";

export interface PollingState<T> {
  data: T | undefined;
  error: Error | undefined;
  loading: boolean;
  refresh: () => Promise<void>;
}

/**
 * Fetch on mount and re-fetch every `intervalMs` while `shouldPoll(data)` is true. Requests never
 * overlap, and stale responses from a previous `key` are discarded.
 */
export function usePolling<T>(
  key: string,
  fetcher: () => Promise<T>,
  { intervalMs = 1500, shouldPoll = () => false }: { intervalMs?: number; shouldPoll?: (data: T) => boolean } = {},
): PollingState<T> {
  const [data, setData] = useState<T>();
  const [error, setError] = useState<Error>();
  const [loading, setLoading] = useState(true);
  const fetcherRef = useRef(fetcher);
  const shouldPollRef = useRef(shouldPoll);
  const generation = useRef(0);
  const inFlight = useRef(false);

  useEffect(() => {
    fetcherRef.current = fetcher;
    shouldPollRef.current = shouldPoll;
  });

  const run = useCallback(async (gen: number): Promise<T | undefined> => {
    if (inFlight.current) return undefined;
    inFlight.current = true;
    try {
      const result = await fetcherRef.current();
      if (gen !== generation.current) return undefined;
      setData(result);
      setError(undefined);
      return result;
    } catch (err) {
      if (gen === generation.current) setError(err instanceof Error ? err : new Error(String(err)));
      return undefined;
    } finally {
      inFlight.current = false;
      if (gen === generation.current) setLoading(false);
    }
  }, []);

  useEffect(() => {
    const gen = ++generation.current;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let cancelled = false;

    const loop = async () => {
      const result = await run(gen);
      if (cancelled) return;
      const keepGoing = result === undefined || shouldPollRef.current(result);
      if (keepGoing) timer = setTimeout(loop, intervalMs);
    };
    void loop();

    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [key, intervalMs, run]);

  const refresh = useCallback(async () => {
    await run(generation.current);
  }, [run]);

  return { data, error, loading, refresh };
}
