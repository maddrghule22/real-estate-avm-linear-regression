// Cloudflare Pages Function: POST /api/predict
import modelWeights from "./model_weights.json";

export async function onRequestOptions() {
  return new Response(null, {
    headers: {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type"
    }
  });
}

export async function onRequestPost({ request }) {
  const corsHeaders = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type"
  };

  try {
    const body = await request.json();

    // Required fields check
    if (body.OverallQual === undefined || body.GrLivArea === undefined || body.YearBuilt === undefined) {
      return new Response(
        JSON.stringify({
          error: "Missing required fields: OverallQual, GrLivArea, YearBuilt",
          status: "validation_error"
        }),
        { status: 422, headers: corsHeaders }
      );
    }

    const overallQual = Math.min(Math.max(Number(body.OverallQual) || 5, 1), 10);
    const overallCond = Math.min(Math.max(Number(body.OverallCond) || 5, 1), 10);
    const yrSold = Number(body.YrSold) || 2010;
    const yearBuilt = Number(body.YearBuilt) || 2000;
    const yearRemodAdd = Number(body.YearRemodAdd) || yearBuilt;

    const bsmtSF = Math.max(Number(body.TotalBsmtSF) || 0, 0);
    const flr1SF = Math.max(Number(body["1stFlrSF"] || body.FirstFlrSF) || 0, 0);
    const flr2SF = Math.max(Number(body["2ndFlrSF"] || body.SecondFlrSF) || 0, 0);
    const grLivArea = Math.max(Number(body.GrLivArea) || 0, 0);

    const fullBath = Number(body.FullBath) || 1;
    const halfBath = Number(body.HalfBath) || 0;
    const bsmtFullBath = Number(body.BsmtFullBath) || 0;
    const bsmtHalfBath = Number(body.BsmtHalfBath) || 0;
    const totRmsAbvGrd = Number(body.TotRmsAbvGrd) || 6;
    const fireplaces = Number(body.Fireplaces) || 0;

    const garageCars = Number(body.GarageCars) || 1;
    const garageArea = Math.max(Number(body.GarageArea) || 0, 0);
    const lotArea = Math.max(Number(body.LotArea) || 7500, 0);

    // 1. Domain Feature Engineering
    const totalSF = bsmtSF + flr1SF + flr2SF;
    const totalBaths = fullBath + 0.5 * halfBath + bsmtFullBath + 0.5 * bsmtHalfBath;
    const houseAge = Math.max(0, yrSold - yearBuilt);
    const remodAge = Math.max(0, yrSold - yearRemodAdd);

    const hasGarage = garageArea > 0 ? 1.0 : 0.0;
    const hasBasement = bsmtSF > 0 ? 1.0 : 0.0;
    const hasFireplace = fireplaces > 0 ? 1.0 : 0.0;
    const qualTimesSF = overallQual * totalSF;

    const numVals = [
      overallQual,
      overallCond,
      grLivArea,
      lotArea,
      totRmsAbvGrd,
      garageCars,
      garageArea,
      totalSF,
      totalBaths,
      houseAge,
      remodAge,
      hasGarage,
      hasBasement,
      hasFireplace,
      qualTimesSF
    ];

    // 2. Ridge Linear Regression Dot Product in Log Space
    let logPred = modelWeights.intercept;

    for (let i = 0; i < numVals.length; i++) {
      const mean = modelWeights.scaler_means[i];
      const scale = modelWeights.scaler_scales[i];
      const scaledVal = (numVals[i] - mean) / scale;
      logPred += scaledVal * modelWeights.coefficients[i];
    }

    // 3. One-Hot Encoded Categorical Features (starting at offset index 15)
    let offset = 15;
    for (const [catName, categories] of Object.entries(modelWeights.cat_categories)) {
      const inputCatVal = String(body[catName] || "Missing");
      for (const catLevel of categories) {
        if (inputCatVal === catLevel) {
          logPred += 1.0 * modelWeights.coefficients[offset];
        }
        offset++;
      }
    }

    // 4. Invert Log Transform (expm1)
    const predictedPrice = Math.max(0, Math.expm1(logPred));

    // 5. 95% Confidence / Prediction Interval Calculation
    const se = modelWeights.residual_std_log || 0.125;
    const lowerBound = predictedPrice * Math.exp(-1.96 * se);
    const upperBound = predictedPrice * Math.exp(1.96 * se);

    return new Response(
      JSON.stringify({
        predicted_price_usd: Math.round(predictedPrice * 100) / 100,
        confidence_interval: {
          lower_bound_usd: Math.round(lowerBound * 100) / 100,
          upper_bound_usd: Math.round(upperBound * 100) / 100,
          confidence_level: "95%"
        },
        derived_features: {
          TotalSF: totalSF,
          TotalBaths: totalBaths,
          HouseAge: houseAge,
          RemodAge: remodAge
        },
        model_version: modelWeights.version,
        platform: "Cloudflare Pages Edge Serverless",
        status: "success"
      }),
      { status: 200, headers: corsHeaders }
    );
  } catch (err) {
    return new Response(
      JSON.stringify({
        error: "Failed to process valuation request",
        details: err.message,
        status: "server_error"
      }),
      { status: 500, headers: corsHeaders }
    );
  }
}
