'use client';

export interface StageEvent {
  stage: string;
  payload: unknown;
}

interface EntityPayload {
  type: string;
  value: string;
}

interface EntitiesStagePayload {
  entities?: EntityPayload[];
}

interface FinalStagePayload {
  patient_display_name?: string;
  summary_for_clinician?: string;
  suggested_questions?: string[];
  safety_note?: string;
}

interface BlockedStagePayload {
  workflow_id?: string;
  violations?: string[];
}

interface ErrorStagePayload {
  message?: string;
}

interface PolicyGateStagePayload {
  cloud_allowed?: boolean;
  violations?: string[];
}

interface Props {
  events: Record<string, StageEvent>;
}

const STAGE_META: { key: string; label: string; runtime: 'edge' | 'cloud' | 'gate' }[] = [
  { key: 'transcript', label: '1. Raw transcript (edge)', runtime: 'edge' },
  { key: 'entities', label: '2. Detected entities (edge SLM)', runtime: 'edge' },
  { key: 'redacted', label: '3. Redacted transcript (edge)', runtime: 'edge' },
  { key: 'handover', label: '4. Handover package (edge SLM)', runtime: 'edge' },
  { key: 'policy_gate', label: '5. Policy gate', runtime: 'gate' },
  { key: 'research', label: '6. Cloud research', runtime: 'cloud' },
  { key: 'explanation', label: '7. Cloud explanation', runtime: 'cloud' },
  { key: 'final', label: '8. Final response (edge rehydration)', runtime: 'edge' }
];

export function StagePanels({ events }: Props) {
  return (
    <div className="space-y-4">
      {STAGE_META.map((stage) => (
        <Panel key={stage.key} stage={stage} event={events[stage.key]} />
      ))}
      {events.blocked && <BlockedPanel payload={events.blocked.payload} />}
      {events.error && <ErrorPanel payload={events.error.payload} />}
    </div>
  );
}

function runtimeBadge(runtime: 'edge' | 'cloud' | 'gate') {
  if (runtime === 'edge')
    return <span className="rounded bg-edge px-2 py-0.5 text-xs">EDGE</span>;
  if (runtime === 'cloud')
    return <span className="rounded bg-cloud px-2 py-0.5 text-xs">CLOUD</span>;
  return <span className="rounded bg-amber-600 px-2 py-0.5 text-xs">GATE</span>;
}

function Panel({
  stage,
  event
}: {
  stage: { key: string; label: string; runtime: 'edge' | 'cloud' | 'gate' };
  event: StageEvent | undefined;
}) {
  const isPolicyGate = stage.key === 'policy_gate';
  const gatePayload = (event?.payload ?? null) as PolicyGateStagePayload | null;
  const cloudAllowed = isPolicyGate ? Boolean(gatePayload?.cloud_allowed) : false;
  const violations: string[] = isPolicyGate ? gatePayload?.violations ?? [] : [];

  let border = 'border-slate-800';
  if (isPolicyGate && event) {
    border = cloudAllowed ? 'border-ok' : 'border-block';
  } else if (event) {
    border = stage.runtime === 'cloud' ? 'border-cloud' : 'border-edge';
  }

  return (
    <section className={`rounded-lg border-2 ${border} bg-slate-900 p-4`}>
      <header className="mb-3 flex items-center justify-between">
        <h2 className="text-lg font-semibold">{stage.label}</h2>
        <div className="flex items-center gap-2">
          {runtimeBadge(stage.runtime)}
        </div>
      </header>

      {!event ? (
        <p className="text-sm text-slate-500">waiting…</p>
      ) : isPolicyGate ? (
        <PolicyGateBody allowed={cloudAllowed} violations={violations} />
      ) : stage.key === 'entities' ? (
        <EntitiesBody payload={event.payload} />
      ) : stage.key === 'final' ? (
        <FinalBody payload={event.payload} />
      ) : (
        <pre className="overflow-x-auto whitespace-pre-wrap break-words rounded bg-slate-950 p-3 text-xs text-slate-300">
          {JSON.stringify(event.payload, null, 2)}
        </pre>
      )}
    </section>
  );
}

function PolicyGateBody({ allowed, violations }: { allowed: boolean; violations: string[] }) {
  if (allowed) {
    return (
      <div className="rounded bg-ok/20 p-4 text-ok">
        <p className="text-xl font-bold">✓ cloud_allowed = true</p>
        <p className="mt-1 text-sm">Handover package passed policy validation.</p>
      </div>
    );
  }
  return (
    <div className="rounded bg-block/20 p-4 text-block">
      <p className="text-xl font-bold">✗ cloud_allowed = false</p>
      <p className="mt-2 text-sm font-medium">Violations:</p>
      <ul className="mt-1 list-disc pl-6 text-sm">
        {violations.map((v, i) => (
          <li key={i}>{v}</li>
        ))}
      </ul>
    </div>
  );
}

function EntitiesBody({ payload }: { payload: unknown }) {
  const entities = (payload as EntitiesStagePayload)?.entities ?? [];
  const highRiskTypes = new Set([
    'PERSON_NAME',
    'DATE_OF_BIRTH',
    'ADDRESS',
    'PHONE_NUMBER',
    'EMAIL',
    'INSURANCE_ID',
    'RELATIVE_NAME',
    'EMPLOYER',
    'FREE_TEXT_IDENTIFIER'
  ]);
  const clinicalTypes = new Set(['MEDICAL_CONDITION', 'MEDICATION', 'SYMPTOM', 'PROCEDURE']);

  return (
    <div className="flex flex-wrap gap-2">
      {entities.map((e) => (
        <span
          key={`${e.type}-${e.value}`}
          title={`${e.type}`}
          className={`rounded px-2 py-1 text-xs ${
            highRiskTypes.has(e.type)
              ? 'bg-block/40'
              : clinicalTypes.has(e.type)
                ? 'bg-ok/30'
                : 'bg-amber-700/40'
          }`}
        >
          {e.type}: {e.value}
        </span>
      ))}
      {entities.length === 0 && <p className="text-sm text-slate-500">no entities</p>}
    </div>
  );
}

function FinalBody({ payload }: { payload: unknown }) {
  const finalPayload = payload as FinalStagePayload;
  return (
    <div className="space-y-3">
      {finalPayload.patient_display_name && (
        <div className="rounded bg-slate-800 p-3">
          <p className="text-xs uppercase text-slate-400">Patient (rehydrated locally)</p>
          <p className="text-lg font-medium">{finalPayload.patient_display_name}</p>
        </div>
      )}
      <div>
        <p className="text-xs uppercase text-slate-400">Summary</p>
        <p>{finalPayload.summary_for_clinician}</p>
      </div>
      {finalPayload.suggested_questions?.length ? (
        <div>
          <p className="text-xs uppercase text-slate-400">Suggested questions</p>
          <ul className="list-disc pl-6 text-sm">
            {finalPayload.suggested_questions.map((q, i) => (
              <li key={i}>{q}</li>
            ))}
          </ul>
        </div>
      ) : null}
      <p className="text-xs italic text-slate-500">{finalPayload.safety_note}</p>
    </div>
  );
}

function BlockedPanel({ payload }: { payload: unknown }) {
  const blockedPayload = payload as BlockedStagePayload;
  return (
    <section className="rounded-lg border-2 border-block bg-block/10 p-4">
      <h2 className="text-lg font-bold text-block">Workflow blocked at policy gate</h2>
      <p className="mt-1 text-sm">
        Cloud agents were not invoked. workflow_id: {blockedPayload.workflow_id}
      </p>
      <ul className="mt-2 list-disc pl-6 text-sm">
        {(blockedPayload.violations ?? []).map((v, i) => (
          <li key={i}>{v}</li>
        ))}
      </ul>
    </section>
  );
}

function ErrorPanel({ payload }: { payload: unknown }) {
  const errorPayload = payload as ErrorStagePayload;
  return (
    <section className="rounded-lg border-2 border-block bg-block/10 p-4">
      <h2 className="text-lg font-bold text-block">Pipeline error</h2>
      <p className="mt-1 text-sm">{errorPayload.message}</p>
    </section>
  );
}
