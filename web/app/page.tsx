'use client';

import { useState } from 'react';
import { StagePanels, StageEvent } from '@/components/StagePanels';
import { Controls } from '@/components/Controls';

export default function Home() {
  const [events, setEvents] = useState<Record<string, StageEvent>>({});
  const [running, setRunning] = useState(false);
  const [workflowId, setWorkflowId] = useState<string | null>(null);

  async function runWorkflow(opts: { audio: File | null; transcript: string; forceViolation: boolean }) {
    setEvents({});
    setRunning(true);
    setWorkflowId(null);

    if (!opts.audio && !opts.transcript.trim()) {
      setRunning(false);
      setEvents({ error: { stage: 'error', payload: { message: 'Please select an audio file or paste a transcript.' } } });
      return;
    }

    const form = new FormData();
    if (opts.audio) {
      form.append('audio', opts.audio);
    }
    if (opts.transcript.trim()) {
      form.append('transcript', opts.transcript.trim());
    }
    form.append('force_violation', String(opts.forceViolation));
    form.append('language_hint', 'de-AT');

    let workflow_id: string;
    try {
      const res = await fetch('/api/run', { method: 'POST', body: form });
      if (!res.ok) {
        const errPayload = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
        setEvents({ error: { stage: 'error', payload: { message: errPayload.error ?? `HTTP ${res.status}` } } });
        setRunning(false);
        return;
      }
      ({ workflow_id } = await res.json());
      setWorkflowId(workflow_id);
    } catch {
      setEvents({ error: { stage: 'error', payload: { message: 'Failed to start workflow. Backend unavailable or network error.' } } });
      setRunning(false);
      return;
    }

    const es = new EventSource(`/api/events/${workflow_id}`);
    let completed = false;
    const stages = [
      'progress',
      'transcript',
      'entities',
      'redacted',
      'handover',
      'policy_gate',
      'blocked',
      'research',
      'explanation',
      'final',
      'error'
    ];
    stages.forEach((stage) => {
      es.addEventListener(`stage.${stage}`, (e: MessageEvent) => {
        const payload = JSON.parse(e.data);
        setEvents((prev) => ({ ...prev, [stage]: { stage, payload } }));
      });
    });
    es.addEventListener('done', () => {
      completed = true;
      es.close();
      setRunning(false);
    });
    es.onerror = () => {
      if (!completed) {
        setEvents((prev) => ({
          ...prev,
          error: {
            stage: 'error',
            payload: {
              message: 'Event stream interrupted. Attempting to reconnect…'
            }
          }
        }));
      }
    };
  }

  return (
    <main className="mx-auto max-w-5xl p-6">
      <header className="mb-8">
        <h1 className="text-3xl font-bold">Hybrid Multi-Agent Demo</h1>
        <p className="mt-2 text-slate-400">
          Edge SLMs handle identity and redaction. The cloud LLM only ever sees
          a pseudonymised handover package.
        </p>
        {workflowId && (
          <p className="mt-1 text-xs text-slate-500">workflow_id: {workflowId}</p>
        )}
      </header>

      <Controls running={running} onRun={runWorkflow} />
      <StagePanels events={events} />
    </main>
  );
}
