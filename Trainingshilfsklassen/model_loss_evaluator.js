// === Hilfsfunktionen ===
  function unpackData(data) {
    return {
      x: data.map(p => p.x),
      y: data.map(p => p.y),
    };
  }
  
  // === Modellstruktur ===
  function evalModel() {
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 100, activation: 'relu', inputShape: [1] }));
    model.add(tf.layers.dense({ units: 100, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1, activation: 'linear' }));
    model.compile({ optimizer: tf.train.adam(0.01), loss: 'meanSquaredError' });
    return model;
  }
  
  // === Verlust-Plot ===
  function drawLossPlot(trainLoss, valLoss) {
    const ctx = document.getElementById('lossCanvas').getContext('2d');
  
    new Chart(ctx, {
      type: 'line',
      data: {
        labels: trainLoss.map((_, i) => i + 1),
        datasets: [
          {
            label: 'Train Loss',
            data: trainLoss,
            borderColor: 'blue',
            fill: false,
          },
          {
            label: 'Test Loss',
            data: valLoss,
            borderColor: 'red',
            fill: false,
          },
        ],
      },
      options: {
        responsive: true,
        plugins: {
          title: {
            display: true,
            text: 'Trainings- vs. Test-Loss (logarithmisch)',
          },
        },
        scales: {
          x: { title: { display: true, text: 'Epoche' } },
          y: {
            title: { display: true, text: 'Loss' },
            type: 'logarithmic',
            min: Math.min(...trainLoss, ...valLoss),
            max: Math.max(...trainLoss, ...valLoss),
          },
          
        },
      },
    });
  }
  
  // === Hauptfunktion ===
  async function runLossEvaluation() {
    const raw = localStorage.getItem("regressionDataset");
    if (!raw) {
      console.error("Kein Datensatz im localStorage gefunden!");
      return;
    }
  
    const dataset = JSON.parse(raw);
    const { x: xTrain, y: yTrain } = unpackData(dataset.noisy.train);
    const { x: xTest, y: yTest } = unpackData(dataset.noisy.test);
  
    const xTrainTensor = tf.tensor2d(xTrain, [xTrain.length, 1]);
    const yTrainTensor = tf.tensor2d(yTrain, [yTrain.length, 1]);
    const xTestTensor = tf.tensor2d(xTest, [xTest.length, 1]);
    const yTestTensor = tf.tensor2d(yTest, [yTest.length, 1]);
  
    const model = evalModel();
  
    const history = await model.fit(xTrainTensor, yTrainTensor, {
      epochs: 500, 
      batchSize: 32,
      validationData: [xTestTensor, yTestTensor],
      verbose: 0,
    });
  
    tf.dispose([xTrainTensor, yTrainTensor, xTestTensor, yTestTensor]);
  
    drawLossPlot(history.history.loss, history.history.val_loss);
  }
  
  window.addEventListener('DOMContentLoaded', runLossEvaluation);
  