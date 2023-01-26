const fs = require("fs");
const produceEstimate = require("./estimator.js");

const div = 'Khulna';
const dis = 'Narail';
const upa = 'Lohagara';
const uni = 'Dighalia';
const mou = 'Dighalia';
const depth = 4.572;
const colour = 'Red';
const utensil = 'Red';
const flood = "no";

const divisions = JSON.parse(
  fs.readFileSync(`./aggregate-data/${div}-${dis}.json`)
);

console.log(produceEstimate(
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
));
