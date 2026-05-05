// Streaming proxy for the backend SSE endpoint.
//
// Next.js' built-in `rewrites()` proxy buffers upstream responses, which
// breaks Server-Sent Events: the browser only sees events once the upstream
// response ends (workflow completes or fails). This Route Handler forwards
// the upstream stream chunk-by-chunk so per-stage events arrive in real time.

import { NextRequest } from 'next/server';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';
// Disable any framework-level revalidation/caching of this stream.
export const revalidate = 0;
export const fetchCache = 'force-no-store';

const BACKEND_URL = process.env.BACKEND_URL || 'http://backend:8000';

export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ workflowId: string }> }
) {
  const { workflowId } = await params;

  const upstream = await fetch(`${BACKEND_URL}/api/events/${workflowId}`, {
    method: 'GET',
    headers: { Accept: 'text/event-stream' },
    // Forward client disconnects so the backend can stop streaming.
    signal: req.signal,
    cache: 'no-store',
  });

  if (!upstream.ok || !upstream.body) {
    return new Response(
      `event: error\ndata: ${JSON.stringify({ message: `upstream HTTP ${upstream.status}` })}\n\n`,
      {
        status: upstream.status || 502,
        headers: { 'Content-Type': 'text/event-stream' },
      }
    );
  }

  // Manually pump the upstream stream into a fresh ReadableStream and call
  // `controller.enqueue` per chunk. This avoids any framework-level buffering
  // that can occur when returning the upstream body directly in `next dev`.
  const reader = upstream.body.getReader();
  const stream = new ReadableStream<Uint8Array>({
    async pull(controller) {
      try {
        const { value, done } = await reader.read();
        if (done) {
          controller.close();
          return;
        }
        if (value) controller.enqueue(value);
      } catch (err) {
        controller.error(err);
      }
    },
    cancel(reason) {
      reader.cancel(reason).catch(() => { });
    },
  });

  return new Response(stream, {
    status: 200,
    headers: {
      'Content-Type': 'text/event-stream; charset=utf-8',
      'Cache-Control': 'no-cache, no-transform',
      Connection: 'keep-alive',
      // Prevent compression-based buffering (gzip/deflate buffers small chunks).
      'Content-Encoding': 'identity',
      // Disable proxy/CDN buffering (nginx, etc.) along the path.
      'X-Accel-Buffering': 'no',
    },
  });
}
