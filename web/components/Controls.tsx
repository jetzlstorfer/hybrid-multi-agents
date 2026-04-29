'use client';

import { useState } from 'react';

interface Props {
  running: boolean;
  onRun: (opts: { audio: File | null; forceViolation: boolean }) => void;
}

export function Controls({ running, onRun }: Props) {
  const [forceViolation, setForceViolation] = useState(false);
  const [audio, setAudio] = useState<File | null>(null);

  return (
    <div className="mb-6 flex flex-wrap items-center gap-3 rounded-lg bg-slate-900 p-4">
      <button
        disabled={running || !audio}
        onClick={() => onRun({ audio, forceViolation })}
        className="rounded bg-edge px-4 py-2 font-medium text-white disabled:opacity-50"
      >
        {running ? 'Running…' : 'Run pipeline'}
      </button>
      <input
        type="file"
        accept="audio/*"
        disabled={running}
        onChange={(e) => setAudio(e.target.files?.[0] ?? null)}
        className="text-sm text-slate-300"
      />
      <label className="flex items-center gap-2 text-sm text-slate-300">
        <input
          type="checkbox"
          checked={forceViolation}
          onChange={(e) => setForceViolation(e.target.checked)}
          className="h-4 w-4"
        />
        Force violation (inject identifier before policy gate)
      </label>
    </div>
  );
}
