// === Hilfsfunktionen ===
function unpackData(data) {
  return {
    x: data.map(p => p.x),
    y: data.map(p => p.y),
  };
}

function linspace(start, end, num) {
  const step = (end - start) / (num - 1);
  return Array.from({ length: num }, (_, i) => start + i * step);
}

function drawComparisonPlot(canvasId, xData, yData, xLine, yLine, label) {
  const ctx = document.getElementById(canvasId).getContext('2d');
  new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: label,
          data: xData.map((x, i) => ({ x, y: yData[i] })),
          backgroundColor: 'blue',
        },
        {
          label: 'Modellvorhersage',
          data: xLine.map((x, i) => ({ x, y: yLine[i] })),
          borderColor: 'red',
          fill: false,
          showLine: true,
          pointRadius: 0,
        },
      ],
    },
    options: {
      plugins: { title: { display: true, text: label + ' vs. Modell' } },
      scales: {
        x: { title: { display: true, text: 'x' }, min: -2, max: 2 },
        y: { title: { display: true, text: 'y' } },
      },
    },
  });
}

async function predictModelLine(model, xMin = -2, xMax = 2, numPoints = 200) {
  const xValues = linspace(xMin, xMax, numPoints);
  const xTensor = tf.tensor2d(xValues, [xValues.length, 1]);
  const yPredTensor = model.predict(xTensor);
  const yPred = await yPredTensor.data();
  tf.dispose([xTensor, yPredTensor]);
  return { xValues, yPred };
}

async function evaluateModel(model, x, y) {
  const xTensor = tf.tensor2d(x, [x.length, 1]);
  const yTensor = tf.tensor2d(y, [y.length, 1]);
  const lossTensor = model.evaluate(xTensor, yTensor);
  const loss = (await lossTensor.data())[0];
  tf.dispose([xTensor, yTensor, lossTensor]);
  return loss;
}

// === Modell-Architekturen ===
function cleanModel() {
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 100, activation: 'relu', inputShape: [1] }));
  model.add(tf.layers.dense({ units: 100, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 100, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 1, activation: 'linear' }));
  model.compile({ optimizer: tf.train.adam(0.01), loss: 'meanSquaredError' });
  return model;
}

function bestFitModel() {
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 100, activation: 'relu', inputShape: [1] }));
  model.add(tf.layers.dense({ units: 100, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 100, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 1, activation: 'linear' }));
  model.compile({ optimizer: tf.train.adam(0.01), loss: 'meanSquaredError' });
  return model;
}

function overfitModel() {
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 100, activation: 'relu', inputShape: [1] }));
  model.add(tf.layers.dense({ units: 100, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 100, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 1, activation: 'linear' }));
  model.compile({ optimizer: tf.train.adam(0.01), loss: 'meanSquaredError' });
  return model;
}

// === Training + Speicherung ===
async function trainAndSaveModel(name, model, xTrain, yTrain, xTest, yTest, epochs) {
  const xTrainTensor = tf.tensor2d(xTrain, [xTrain.length, 1]);
  const yTrainTensor = tf.tensor2d(yTrain, [yTrain.length, 1]);
  const xTestTensor = tf.tensor2d(xTest, [xTest.length, 1]);
  const yTestTensor = tf.tensor2d(yTest, [yTest.length, 1]);

  await model.fit(xTrainTensor, yTrainTensor, {
    epochs,
    batchSize: 32,
    validationData: [xTestTensor, yTestTensor],
    verbose: 0,
  });

  await model.save(`indexeddb://${name}`);
  tf.dispose([xTrainTensor, yTrainTensor, xTestTensor, yTestTensor]);
}

// === Plot für gespeichertes Modell + Loss-Anzeige ===
async function loadAndPlotModel(name, xTrain, yTrain, xTest, yTest, canvasTrain, canvasTest, label) {
  const model = await tf.loadLayersModel(`indexeddb://${name}`);
  const { xValues, yPred } = await predictModelLine(model);

  drawComparisonPlot(canvasTrain, xTrain, yTrain, xValues, yPred, `Trainingsdaten`);
  drawComparisonPlot(canvasTest, xTest, yTest, xValues, yPred, `Testdaten`);

  const trainLoss = await evaluateModel(model, xTrain, yTrain);
  const testLoss = await evaluateModel(model, xTest, yTest);

  const lossDisplay = document.createElement('div');
  lossDisplay.innerHTML = `
    <strong>${label}</strong><br>
    MSE (Trainingsdaten): ${trainLoss.toFixed(5)}<br>
    MSE (Testdaten): ${testLoss.toFixed(5)}
  `;
  lossDisplay.style.marginTop = '10px';
  lossDisplay.style.fontSize = '14px';

  const canvasElem = document.getElementById(canvasTest);
  canvasElem.parentNode.appendChild(lossDisplay);
}

// === Hauptfunktion ===
async function main() {
  try {
    const dataset = window.fixedDataset;
    if (!dataset) {
      console.error("❌ Dataset nicht verfügbar.");
      return;
    }

    const { x: xTrainClean, y: yTrainClean } = unpackData(dataset.clean.train);
    const { x: xTestClean, y: yTestClean } = unpackData(dataset.clean.test);
    const { x: xTrainNoisy, y: yTrainNoisy } = unpackData(dataset.noisy.train);
    const { x: xTestNoisy, y: yTestNoisy } = unpackData(dataset.noisy.test);

    const models = await tf.io.listModels();

    if (!models['indexeddb://clean-model']) {
      const model = cleanModel();
      await trainAndSaveModel('clean-model', model, xTrainClean, yTrainClean, xTestClean, yTestClean, 500);
    }

    if (!models['indexeddb://bestfit-model']) {
      const model = bestFitModel();
      await trainAndSaveModel('bestfit-model', model, xTrainNoisy, yTrainNoisy, xTestNoisy, yTestNoisy, 385);
    }

    if (!models['indexeddb://overfit-model']) {
      const model = overfitModel();
      await trainAndSaveModel('overfit-model', model, xTrainNoisy, yTrainNoisy, xTestNoisy, yTestNoisy, 2500);
    }

    await loadAndPlotModel('clean-model', xTrainClean, yTrainClean, xTestClean, yTestClean, 'canvasCleanTrain', 'canvasCleanTest', 'Clean Model');
    await loadAndPlotModel('bestfit-model', xTrainNoisy, yTrainNoisy, xTestNoisy, yTestNoisy, 'canvasBestFitTrain', 'canvasBestFitTest', 'Best Fit Model');
    await loadAndPlotModel('overfit-model', xTrainNoisy, yTrainNoisy, xTestNoisy, yTestNoisy, 'canvasOverfitTrain', 'canvasOverfitTest', 'Overfit Model');

  } catch (error) {
    console.error("Fehler beim Laden oder Zeichnen der Modelle:", error);
  }
}

window.addEventListener('DOMContentLoaded', main);
