// === Datengenerierung ===

// ground Truth
function groundTruth(x) {
    return 0.5 * (x + 0.8) * (x + 1.8) * (x - 0.2) * (x - 0.3) * (x - 1.9) + 1;
}

// Zufällige, gleichverteilte x-Werte im Intervall [-2, 2]
function generateXValues(N) {
    const xs = [];
    for (let i = 0; i < N; i++) {
        const x = Math.random() * 4 - 2;
        xs.push(x);
    }
    return xs;
}

// Gaussian Noise
function gaussianNoise(stdDev) {
    const u1 = Math.random();
    const u2 = Math.random();
    const z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    return z * stdDev;
}

// Datengenerierung
function generateData(N = 100, noiseVar = 0.05) {
    const xs = generateXValues(N);
    const ysClean = xs.map(x => groundTruth(x));
    const ysNoisy = ysClean.map(y => y + gaussianNoise(Math.sqrt(noiseVar)));

    const cleanData = xs.map((x, i) => ({ x, y: ysClean[i] }));
    const noisyData = xs.map((x, i) => ({ x, y: ysNoisy[i] }));

    const indices = [...Array(N).keys()];

    const split = N / 2;
    const cleanTrain = indices.slice(0, split).map(i => cleanData[i]);
    const cleanTest = indices.slice(split).map(i => cleanData[i]);

    const noisyTrain = indices.slice(0, split).map(i => noisyData[i]);
    const noisyTest = indices.slice(split).map(i => noisyData[i]);

    return {
        clean: { train: cleanTrain, test: cleanTest },
        noisy: { train: noisyTrain, test: noisyTest },
    };
} 

// Helferfunktion zum Entpacken in x-/y-Arrays
function unpackData(data) {
    return {
        x: data.map(p => p.x),
        y: data.map(p => p.y),
    };
}

// === Datensätze generieren und vorbereiten ===
const dataset = generateData();

const { x: xTrainClean, y: yTrainClean } = unpackData(dataset.clean.train);
const { x: xTestClean, y: yTestClean } = unpackData(dataset.clean.test);

const { x: xTrainNoisy, y: yTrainNoisy } = unpackData(dataset.noisy.train);
const { x: xTestNoisy, y: yTestNoisy } = unpackData(dataset.noisy.test);

// === Speicherfunktionen ===
function saveDatasetToLocalStorage(dataset) {
    localStorage.setItem("regressionDataset", JSON.stringify(dataset));
}

function loadDatasetFromLocalStorage() {
    return JSON.parse(localStorage.getItem("regressionDataset"));
}

// === Hauptfunktion ===
function prepareAndStoreDataIfNeeded() {
    let dataset = loadDatasetFromLocalStorage();
    if (!dataset) {
        dataset = generateData();
        saveDatasetToLocalStorage(dataset);
        console.log("Datensatz neu erzeugt und gespeichert.");
    } else {
        console.log("Datensatz aus localStorage geladen.");
    }
    return dataset;
}

// === Datensatz global verfügbar machen ===
window.dataset = prepareAndStoreDataIfNeeded();


function drawDataPlotsFromDataset(dataset) {

    const xTrainClean = dataset.clean.train.map(p => p.x);
    const yTrainClean = dataset.clean.train.map(p => p.y);
    const xTestClean = dataset.clean.test.map(p => p.x);
    const yTestClean = dataset.clean.test.map(p => p.y);
  
    const xTrainNoisy = dataset.noisy.train.map(p => p.x);
    const yTrainNoisy = dataset.noisy.train.map(p => p.y);
    const xTestNoisy = dataset.noisy.test.map(p => p.x);
    const yTestNoisy = dataset.noisy.test.map(p => p.y);
  
    const ctxClean = document.getElementById('dataPlotClean').getContext('2d');
    const ctxNoisy = document.getElementById('dataPlotNoisy').getContext('2d');
  
    new Chart(ctxClean, {
      type: 'scatter',
      data: {
        datasets: [
          {
            label: 'Train Clean',
            data: xTrainClean.map((x, i) => ({ x: x, y: yTrainClean[i] })),
            backgroundColor: 'blue',
          },
          {
            label: 'Test Clean',
            data: xTestClean.map((x, i) => ({ x: x, y: yTestClean[i] })),
            backgroundColor: 'lightblue',
          },
        ],
      },
      options: {
        plugins: { title: { display: true, text: 'Daten ohne Rauschen' } },
        scales: {
          x: { title: { display: true, text: 'x' } },
          y: { title: { display: true, text: 'y' } },
        },
      },
    });
  
    new Chart(ctxNoisy, {
      type: 'scatter',
      data: {
        datasets: [
          {
            label: 'Train Noisy',
            data: xTrainNoisy.map((x, i) => ({ x: x, y: yTrainNoisy[i] })),
            backgroundColor: 'red',
          },
          {
            label: 'Test Noisy',
            data: xTestNoisy.map((x, i) => ({ x: x, y: yTestNoisy[i] })),
            backgroundColor: 'orange',
          },
        ],
      },
      options: {
        plugins: { title: { display: true, text: 'Daten mit Rauschen' } },
        scales: {
          x: { title: { display: true, text: 'x' } },
          y: { title: { display: true, text: 'y' } },
        },
      },
    });
  }

  window.addEventListener('DOMContentLoaded', () => {
    drawDataPlotsFromDataset(window.dataset);
});
