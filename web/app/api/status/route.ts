import { NextRequest } from 'next/server';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const BACKEND_URL = process.env.BACKEND_URL || 'http://backend:8000';

export async function GET(req: NextRequest) {
  const upstream = await fetch(`${BACKEND_URL}/api/status`, {
    method: 'GET',
    headers: { Accept: 'application/json' },
    signal: req.signal,
    cache: 'no-store',
  });

  const contentType = upstream.headers.get('content-type') || 'application/json';
  const bodyText = await upstream.text();

  return new Response(bodyText, {
    status: upstream.status,
    headers: { 'Content-Type': contentType },
  });
}
