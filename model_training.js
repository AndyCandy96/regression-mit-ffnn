// === Datengenerierung ===

// Funktion y(x) = 0.5(x+0.8)(x+1.8)(x−0.2)(x−0.3)(x−1.9) + 1
function groundTruth(x) {
    return 0.5 * (x + 0.8) * (x + 1.8) * (x - 0.2) * (x - 0.3) * (x - 1.9) + 1;
  }
  
  // Zufällige, gleichverteilte x-Werte im Intervall [-2, 2]
  function generateXValues(N) {
    const xs = [];
    for (let i = 0; i < N; i++) {
      const x = Math.random() * 4 - 2; // [0,4) → [-2,2)
      xs.push(x);
    }
    return xs;
  }
  
  // Gaussian Noise Generator (Box-Muller Transform)
  function gaussianNoise(stdDev) {
    const u1 = Math.random();
    const u2 = Math.random();
    const z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    return z * stdDev;
  }
  
  // Generiert Datensätze: sauber & verrauscht
  function generateData(N = 100, noiseVar = 0.05) {
    const xs = generateXValues(N);
    const ysClean = xs.map(x => groundTruth(x));
    const ysNoisy = ysClean.map(y => y + gaussianNoise(Math.sqrt(noiseVar)));
  
    // Kombinieren
    const cleanData = xs.map((x, i) => ({ x, y: ysClean[i] }));
    const noisyData = xs.map((x, i) => ({ x, y: ysNoisy[i] }));
  
    // Zufällig mischen (für saubere Trennung)
    const indices = [...Array(N).keys()];
    for (let i = indices.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }
  
    // Aufteilen in 50 Train / 50 Test
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
  
  // Beispielnutzung:
  const dataset = generateData(100, 0.05);
  console.log("Train Clean Sample:", dataset.clean.train[0]);
  console.log("Train Noisy Sample:", dataset.noisy.train[0]);
  