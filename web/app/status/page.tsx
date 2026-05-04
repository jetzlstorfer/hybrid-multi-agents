'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';

interface StatusResponse {
  overall: 'ok' | 'degraded' | 'not_configured';
  models?: ModelStatus[];
}

interface ModelStatus {
  role: string;
  label: string;
  model: string;
  status: 'loaded' | 'cached' | 'not_cached' | 'sdk_missing' | 'error';
  is_cached: boolean;
  is_loaded: boolean;
  path: string | null;
  detail: string;
}

const STATUS_COLORS: Record<string, string> = {
  ok: 'bg-ok/20 text-ok border-ok',
  configured: 'bg-ok/20 text-ok border-ok',
  loaded: 'bg-ok/20 text-ok border-ok',
  cached: 'bg-emerald-700/20 text-emerald-300 border-emerald-600',
  not_cached: 'bg-amber-700/20 text-amber-400 border-amber-600',
  sdk_missing: 'bg-amber-700/20 text-amber-400 border-amber-600',
  unreachable: 'bg-block/20 text-block border-block',
  not_configured: 'bg-amber-700/20 text-amber-400 border-amber-600',
  error: 'bg-block/20 text-block border-block',
  degraded: 'bg-block/20 text-block border-block',
};

const STATUS_ICON: Record<string, string> = {
  ok: '✓',
  configured: '✓',
  loaded: '✓',
  cached: '⬇',
  not_cached: '○',
  sdk_missing: '⚠',
  unreachable: '✗',
  not_configured: '⚠',
  error: '✗',
  degraded: '✗',
};

function StatusBadge({ status }: { status: string }) {
  const cls = STATUS_COLORS[status] ?? 'bg-slate-800 text-slate-400 border-slate-600';
  const icon = STATUS_ICON[status] ?? '?';
  return (
    <span className={`inline-flex items-center gap-1 rounded border px-2 py-0.5 text-xs font-mono ${cls}`}>
      {icon} {status}
    </span>
  );
}

export default function StatusPage() {
  const [data, setData] = useState<StatusResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastChecked, setLastChecked] = useState<Date | null>(null);

  async function refresh() {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('/api/status');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setData(await res.json());
      setLastChecked(new Date());
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  }

  // Initial load + auto-refresh every 10 seconds.
  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 10_000);
    return () => clearInterval(id);
  }, []);

  return (
    <main className="mx-auto max-w-3xl p-6">
      <header className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Service Status</h1>
          {lastChecked && (
            <p className="mt-0.5 text-xs text-slate-500">
              Last checked: {lastChecked.toLocaleTimeString()} · auto-refreshes every 10 s
            </p>
          )}
        </div>
        <div className="flex items-center gap-3">
          {data && (
            <StatusBadge status={data.overall} />
          )}
          <button
            onClick={refresh}
            disabled={loading}
            className="rounded bg-slate-700 px-3 py-1.5 text-sm hover:bg-slate-600 disabled:opacity-50"
          >
            {loading ? 'Checking…' : 'Refresh'}
          </button>
        </div>
      </header>

      {error && (
        <div className="mb-4 rounded border border-block bg-block/10 p-4 text-block text-sm">
          Could not reach backend: {error}
        </div>
      )}

      {data && (
        <div className="space-y-6">
          {data.models && (
            <section>
              <h2 className="mb-2 text-lg font-semibold">Edge Model Cache</h2>
              <div className="space-y-3">
                {data.models.map((m) => (
                  <section
                    key={m.role}
                    className={`rounded-lg border-2 bg-slate-900 p-4 ${
                      m.status === 'loaded'
                        ? 'border-ok'
                        : m.status === 'cached'
                          ? 'border-emerald-600'
                          : m.status === 'not_cached' || m.status === 'sdk_missing'
                            ? 'border-amber-600'
                            : 'border-block'
                    }`}
                  >
                    <div className="flex items-start justify-between gap-4">
                      <div>
                        <h3 className="font-semibold">{m.label}</h3>
                        <p className="mt-0.5 font-mono text-xs text-slate-400 break-all">{m.model}</p>
                        <p className="mt-1 text-sm text-slate-300">{m.detail}</p>
                        {m.path && (
                          <p className="mt-1 font-mono text-xs text-slate-500 break-all">cache path: {m.path}</p>
                        )}
                      </div>
                      <StatusBadge status={m.status} />
                    </div>
                  </section>
                ))}
              </div>
            </section>
          )}
        </div>
      )}

      {!data && !loading && !error && (
        <p className="text-slate-500">No data yet.</p>
      )}

      <div className="mt-6">
        <Link href="/" className="text-sm text-slate-400 hover:text-slate-200">
          ← Back to demo
        </Link>
      </div>
    </main>
  );
}
