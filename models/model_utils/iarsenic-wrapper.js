const fs = require("fs");
const model = (process.argv[4] == null) ? 'model5' : process.argv[4]
const kFold = process.argv[5]
const produceEstimate = require(`../${model}/model/k${kFold}/estimator.js`);

const srcCsv = process.argv[2];
const stainColour = process.argv[3];

const rows = String(fs.readFileSync(srcCsv)).split("\n");

predictions = [];

for (const row of rows) {
  rowArr = row.split(",");

  // skip headers & \n at EOF
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

  const divisions = (() => {
    if (model === 'model5') {
      return JSON.parse(
        fs.readFileSync(`./models/${model}/model/k${kFold}/aggregate-data/${div}-${dis}.json`)
      );
    } else {
      // for models with a singe aggregate datafile, this must only be retrieved once
      // however because model5 splits the data into difference files it must be updated
      // to the correct file every loop. This could likely be sensibly optimized
      const raw = fs.readFileSync(`./models/${model}/model/k${kFold}/aggregate-data.js`, 'utf-8')
      return JSON.parse(raw.slice(raw.indexOf('=') + 1))

    }
  })()

  const estimate = (() => {
    if (model === 'model5') {
      return produceEstimate(
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
    } else {
      return produceEstimate(
        divisions,
        div,
        dis,
        upa,
        uni,
        depth,
        colour,
        utensil
      );
    }
  })()

  if (estimate.severity == null) {
    predictions.push(estimate.message);
  } else {
    predictions.push(estimate.severity);
  }
}

// add header
predictions.unshift("Prediction");

const outFilename = `./prediction_data/${model}-k${kFold}-${stainColour}-${Math.floor(
  new Date().getTime() / 1000
)}.csv`;

fs.writeFileSync(outFilename, predictions.join("\n"));
console.log(outFilename);
