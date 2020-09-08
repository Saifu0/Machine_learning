// // function Tensor() {
// //   // tf.tensor(data,shape,datatype) e.g tf.tensor([1,2,3,4],[2,2],'int32')
// //   //   const data = tf.tensor(
// //   //     [
// //   //       [1, 2, 3],
// //   //       [4, 5, 6],
// //   //       [7, 8, 9],
// //   //     ],
// //   //     "int32"
// //   //   );
// //   //   data.print();
// //   //   console.log(data);

// //   const val = [];
// //   for (let i = 0; i < 15; i++) {
// //     val.push(i + 10);
// //   }

// //   const shape = [5, 3];
// //   const data = tf.tensor(val, shape);

// //   //   console.log(data.dataSync());

// //   //   data.print();
// //   console.log(data.toString());
// // }

// // Define a model for linear regression.

// function LinearRegression() {
//   const model = tf.sequential();
//   model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

//   model.compile({ loss: "meanSquaredError", optimizer: "sgd" });

//   // Generate some synthetic data for training.
//   const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
//   const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);

//   // Train the model using the data.
//   model.fit(xs, ys, { epochs: 10 }).then(() => {
//     // Use the model to do inference on a data point the model hasn't seen before:
//     const y_pred = model.predict(tf.tensor2d([5], [1, 1]));
//     console.log(y_pred.dataSync());
//     // Open the browser devtools to see the output
//     const accuracy = tf.metrics.meanAbsolutePercentageError([3], y_pred);
//     accuracy.print();
//   });
// }

// function LogisticRegression() {
//   const data = [
//     2.78,
//     2.55,
//     3.39,
//     4.4,
//     1.38,
//     1.85,
//     3.06,
//     3.0,
//     7.67,
//     2.75,
//     5.33,
//     2.08,
//     8.67,
//     0.24,
//     7.67,
//     3.5,
//   ];
//   const X_test = tf.tensor([1.46, 2.36, 6.92, 1.77], [2, 2]);
//   const y_test = tf.tensor([0, 1]);

//   const dataY = [0, 0, 0, 0, 1, 1, 1, 1];

//   const shape = [8, 2];

//   const X = tf.tensor(data, shape);

//   const Y = tf.tensor(dataY);

//   const model = tf.sequential();

//   model.add(tf.layers.dense({ units: 1, inputShape: [2] }));

//   const Loss = tf.losses.logLoss;

//   model.compile({
//     loss: "meanSquaredError",
//     optimizer: "sgd",
//     // metrics: "accuracy",
//   });

//   // X_test.print();
//   // y_test.print();

//   X.print();
//   Y.print();

//   model.fit(X, Y, { epochs: 3 }).then(() => {
//     const y_pred = model.predict(X_test);
//     y_pred.print();
//   });
// }

// Daniel Shiffman
// http://codingtra.in
// http://patreon.com/codingtrain

// Linear Regression with TensorFlow.js
// Video: https://www.youtube.com/watch?v=dLp10CFIvxI

let x_vals = [];
let y_vals = [];

let m, b;

const learningRate = 0.5;
const optimizer = tf.train.sgd(learningRate);

function setup() {
  createCanvas(400, 400);
  m = tf.variable(tf.scalar(random(1)));
  b = tf.variable(tf.scalar(random(1)));
}

function loss(pred, labels) {
  return pred.sub(labels).square().mean();
}

function predict(x) {
  const xs = tf.tensor1d(x);
  // y = mx + b;
  const ys = xs.mul(m).add(b);
  return ys;
}

function mousePressed() {
  let x = map(mouseX, 0, width, 0, 1);
  let y = map(mouseY, 0, height, 1, 0);
  x_vals.push(x);
  y_vals.push(y);
}

function draw() {
  tf.tidy(() => {
    if (x_vals.length > 0) {
      const ys = tf.tensor1d(y_vals);
      optimizer.minimize(() => loss(predict(x_vals), ys));
    }
  });

  background(0);

  stroke(255);
  strokeWeight(8);
  for (let i = 0; i < x_vals.length; i++) {
    let px = map(x_vals[i], 0, 1, 0, width);
    let py = map(y_vals[i], 0, 1, height, 0);
    point(px, py);
  }

  const lineX = [0, 1];

  const ys = tf.tidy(() => predict(lineX));
  let lineY = ys.dataSync();
  ys.dispose();

  let x1 = map(lineX[0], 0, 1, 0, width);
  let x2 = map(lineX[1], 0, 1, 0, width);

  let y1 = map(lineY[0], 0, 1, height, 0);
  let y2 = map(lineY[1], 0, 1, height, 0);

  strokeWeight(2);
  line(x1, y1, x2, y2);

  console.log(tf.memory().numTensors);
  //noLoop();
}
