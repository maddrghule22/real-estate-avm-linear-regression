// Cloudflare Pages Function: GET /api/health
export async function onRequestGet() {
  return new Response(
    JSON.stringify({
      status: "healthy",
      service: "Residential Automated Valuation Model (AVM) API",
      model_loaded: true,
      model_version: "1.0.0-ridge-cv",
      runtime: "Cloudflare Pages Edge Serverless",
      timestamp: new Date().toISOString()
    }),
    {
      headers: {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
        "Cache-Control": "no-cache"
      }
    }
  );
}
