// todo allow model to be selected with cmd args
const fs = require("fs");
const produceEstimate = require("../models/model5/estimator.js");

const srcCsv = process.argv[2];
const stainColour = process.argv[3];

const rows = String(fs.readFileSync(srcCsv)).split("\n");
predictions = [];

for (const row of rows) {
  rowArr = row.split(",");

  if (rowArr[0] === "Division") continue;
  if (rowArr[0] === "") continue;

  const div = rowArr[0];
  const dis = rowArr[1];
  const upa = rowArr[2];
  const uni = rowArr[3];
  const mou = rowArr[4];
  const depth = rowArr[5];
  const colour = stainColour;
  const utensil = "";
  const flood = "";

  const divisions = JSON.parse(
    fs.readFileSync(`./models/model5/aggregate-data/${div}-${dis}.json`)
  );

  const estimate = produceEstimate(
    divisions,
    div,
    dis,
    upa,
    uni,
    mou,
    depth,
    colour,
    utensil,
    flood
  );

  if (estimate.severity == null) {
    predictions.push(estimate.message);
  } else {
    predictions.push(estimate.severity);
  }
}

// add header
predictions.unshift("prediction");

const outFilename = `./prediction_data/model5-${stainColour}-${Math.floor(
  new Date().getTime() / 1000
)}.csv`;
fs.writeFileSync(outFilename, predictions.join("\n"));
console.log(outFilename);
