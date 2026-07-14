import { readFile } from "node:fs/promises";
import path from "node:path";
import { parse } from "yaml";

export const dynamic = "force-dynamic";

async function backendBaseUrl() {
  const candidates = [
    process.env.BACKEND_CONFIG_PATH,
    path.resolve(process.cwd(), "../backend/config.yaml"),
    "/app/backend/config.yaml",
  ].filter(Boolean) as string[];

  for (const candidate of candidates) {
    try {
      const config = parse(await readFile(candidate, "utf8")) as { app?: { port?: number } };
      const port = config.app?.port;
      if (port) return `http://127.0.0.1:${port}`;
    } catch {
      // Thử path tiếp theo; lỗi cuối được báo rõ bên dưới.
    }
  }
  throw new Error("Không đọc được app.port từ backend/config.yaml");
}

async function proxy(request: Request, params: { path: string[] }) {
  try {
    const baseUrl = await backendBaseUrl();
    const incoming = new URL(request.url);
    const target = `${baseUrl}/${params.path.join("/")}${incoming.search}`;
    const headers = new Headers(request.headers);
    headers.delete("host");
    headers.delete("content-length");

    const response = await fetch(target, {
      method: request.method,
      headers,
      body: request.method === "GET" || request.method === "HEAD" ? undefined : request.body,
      duplex: "half",
      cache: "no-store",
    } as RequestInit & { duplex: "half" });

    const responseHeaders = new Headers(response.headers);
    responseHeaders.delete("content-length");
    return new Response(response.body, {
      status: response.status,
      statusText: response.statusText,
      headers: responseHeaders,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Backend proxy lỗi";
    return Response.json({ detail: message }, { status: 502 });
  }
}

export async function GET(request: Request, context: { params: { path: string[] } }) {
  return proxy(request, context.params);
}

export async function POST(request: Request, context: { params: { path: string[] } }) {
  return proxy(request, context.params);
}
