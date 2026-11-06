// Cloudflare Pages Function: GET /api/metadata
import modelWeights from "./model_weights.json";

export async function onRequestGet() {
  return new Response(
    JSON.stringify({
      model_architecture: "Ridge Regression with Log-Transformed Target",
      hyperparameter_alpha: 5.0,
      training_samples: 1166,
      holdout_test_samples: 292,
      holdout_test_rmse: 24441.97,
      holdout_test_mae: 15443.99,
      holdout_test_mape: 9.31,
      holdout_test_r2: 0.9221,
      holdout_test_adjusted_r2: 0.8926,
      features_total: modelWeights.coefficients.length,
      platform: "Cloudflare Pages Edge Serverless",
      version: modelWeights.version
    }),
    {
      headers: {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
        "Cache-Control": "public, max-age=3600"
      }
    }
  );
}
