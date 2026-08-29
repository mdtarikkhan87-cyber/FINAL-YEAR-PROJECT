/**
 * Same-origin proxy from the browser to the FastAPI service.
 *
 * Why this exists rather than pointing the browser straight at the API:
 *
 * `NEXT_PUBLIC_*` variables are inlined into the client bundle **at build
 * time**. In a container that is the wrong lifecycle — baking
 * `http://api:8000` into the bundle would ship a hostname only resolvable
 * inside the Docker network, and the user's browser would fail every request.
 * Baking `http://localhost:8000` instead would work locally and break the
 * moment the stack is deployed anywhere else, forcing an image rebuild to
 * change a URL.
 *
 * So the browser always calls this route on its own origin, and the Next
 * server — which *can* resolve the internal hostname — forwards the request to
 * `API_INTERNAL_URL`, read **at request time**. One environment variable,
 * changeable without rebuilding, correct in both settings. It also means the
 * browser never makes a cross-origin request, so CORS is not in the path at
 * all (the API still sets it, for callers that do go direct).
 *
 * `NEXT_PUBLIC_API_BASE` remains supported for `npm run dev`, where talking to
 * the API directly is simpler; see web/lib/api.ts.
 */

import { NextRequest, NextResponse } from "next/server";

// Read per-request, never at module scope, so the value cannot be captured at
// build time.
function target(): string {
  return (
    process.env.API_INTERNAL_URL ??
    process.env.NEXT_PUBLIC_API_BASE ??
    "http://127.0.0.1:8000"
  ).replace(/\/+$/, "");
}

export const dynamic = "force-dynamic";

async function forward(
  request: NextRequest,
  segments: string[],
  body?: string,
): Promise<NextResponse> {
  const search = request.nextUrl.search ?? "";
  const url = `${target()}/${segments.join("/")}${search}`;

  let upstream: Response;
  try {
    upstream = await fetch(url, {
      method: request.method,
      headers: { "Content-Type": "application/json" },
      body,
      cache: "no-store",
    });
  } catch {
    return NextResponse.json(
      {
        detail:
          `The prediction API is unreachable at ${target()}. ` +
          "If running under Docker Compose, check that the 'api' service is " +
          "healthy; otherwise start it with: uvicorn api.main:app --port 8000",
      },
      { status: 502 },
    );
  }

  // Pass the upstream status and body through untouched: the frontend relies on
  // FastAPI's 422 validation payloads and 404 detail strings to render errors.
  const text = await upstream.text();
  return new NextResponse(text, {
    status: upstream.status,
    headers: {
      "Content-Type": upstream.headers.get("content-type") ?? "application/json",
      "Cache-Control": "no-store",
    },
  });
}

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> },
) {
  const { path } = await params;
  return forward(request, path);
}

export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> },
) {
  const { path } = await params;
  return forward(request, path, await request.text());
}
