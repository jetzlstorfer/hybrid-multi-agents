'use client';

import { useState } from 'react';

interface Props {
  running: boolean;
  onRun: (opts: { audio: File | null; transcript: string; forceViolation: boolean }) => void;
}

export function Controls({ running, onRun }: Props) {
  const [forceViolation, setForceViolation] = useState(false);
  const [audio, setAudio] = useState<File | null>(null);
  const [transcript, setTranscript] = useState('');
  const [mode, setMode] = useState<'audio' | 'transcript'>('audio');

  return (
    <div className="mb-6 flex flex-wrap items-center gap-3 rounded-lg bg-slate-900 p-4">
      <button
        disabled={running || (mode === 'audio' ? !audio : !transcript.trim())}
        onClick={() => onRun({ audio: mode === 'audio' ? audio : null, transcript: mode === 'transcript' ? transcript : '', forceViolation })}
        className="rounded bg-edge px-4 py-2 font-medium text-white disabled:opacity-50"
      >
        {running ? 'Running…' : 'Run pipeline'}
      </button>

      <div className="flex items-center gap-2">
        <label className="flex items-center gap-1 text-sm text-slate-300">
          <input
            type="radio"
            checked={mode === 'audio'}
            onChange={() => setMode('audio')}
            disabled={running}
          />
          Audio file
        </label>
        <label className="flex items-center gap-1 text-sm text-slate-300">
          <input
            type="radio"
            checked={mode === 'transcript'}
            onChange={() => setMode('transcript')}
            disabled={running}
          />
          Paste transcript
        </label>
      </div>

      {mode === 'audio' ? (
        <input
          type="file"
          accept="audio/*"
          disabled={running}
          onChange={(e) => setAudio(e.target.files?.[0] ?? null)}
          className="max-w-full rounded border border-slate-700 bg-slate-800 px-2 py-1 text-sm text-slate-300 file:mr-3 file:rounded file:border-0 file:bg-edge file:px-3 file:py-1.5 file:text-sm file:font-medium file:text-white hover:file:brightness-110 disabled:cursor-not-allowed disabled:opacity-60"
        />
      ) : (
        <textarea
          placeholder="Paste transcript text here (German)…"
          value={transcript}
          onChange={(e) => setTranscript(e.target.value)}
          disabled={running}
          className="flex-1 rounded bg-slate-800 px-3 py-2 text-sm text-slate-100 placeholder-slate-500"
          rows={2}
        />
      )}
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
