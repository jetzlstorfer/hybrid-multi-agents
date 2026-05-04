// Route Handler for POST /api/run – proxies multipart file uploads to the
// backend without letting Next.js parse the body first. Relying solely on
// next.config.js rewrites for multipart POSTs causes Next.js to consume the
// body before forwarding, resulting in HTTP 500.

import { NextRequest } from 'next/server';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

export async function POST(req: NextRequest) {
  const upstream = await fetch(`${BACKEND_URL}/api/run`, {
    method: 'POST',
    headers: req.headers,
    body: req.body,
    // @ts-expect-error – Node fetch requires this to stream a ReadableStream body
    duplex: 'half',
  });

  const data = await upstream.json();
  return new Response(JSON.stringify(data), {
    status: upstream.status,
    headers: { 'Content-Type': 'application/json' },
  });
}
