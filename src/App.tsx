import { Component, createSignal, createEffect, onMount, onCleanup, Show } from 'solid-js';
import * as tf from '@tensorflow/tfjs';
import styles from './App.module.css';

const App: Component = () => {
  let videoRef: HTMLVideoElement | undefined;
  let beforeImageRef: HTMLImageElement | undefined;
  let model: any;

  const [beforeImageCapture, setBeforeImageCapture] = createSignal<HTMLCanvasElement | null>(null);
  const [afterImageCapture, setAfterImageCapture] = createSignal<HTMLCanvasElement | null>(null);
  const [currentFacingMode, setCurrentFacingMode] = createSignal('user');
  const [similarity, setSimilarity] = createSignal(0);
  const [errorMessage, setErrorMessage] = createSignal('');
  const [isMobileDevice, setIsMobileDevice] = createSignal(false);
  const [showLiveFeed, setShowLiveFeed] = createSignal(true);
  const [isModelLoading, setIsModelLoading] = createSignal(true);

  const [isPortrait, setIsPortrait] = createSignal(window.innerHeight > window.innerWidth);

  const handleResize = () => {
    setIsPortrait(window.innerHeight > window.innerWidth);
  };

  const isMobile = () => {
    return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
  };

  const setupCamera = async () => {
    try {
      const constraints = {
        video: {
          facingMode: currentFacingMode(),
          aspectRatio: isPortrait() ? 3 / 4 : 4 / 3
        }
      };
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      if (videoRef) {
        videoRef.srcObject = stream;
        videoRef.onloadedmetadata = () => {
          if (videoRef) {
            videoRef.play();
          }
        };
      }
      setIsMobileDevice(isMobile());
    } catch (error: any) {
      setErrorMessage('Error accessing camera: ' + error.message);
    }
  };

  const loadModel = async () => {
    try {
      setIsModelLoading(true);
      model = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
      setIsModelLoading(false);
    } catch (error: any) {
      setErrorMessage('Error loading model: ' + error.message);
      setIsModelLoading(false);
    }
  };

  const captureImage = () => {
    if (!videoRef) return null;
    const canvas = document.createElement('canvas');
    canvas.width = videoRef.videoWidth;
    canvas.height = videoRef.videoHeight;
    canvas.getContext('2d')?.drawImage(videoRef, 0, 0);
    return canvas;
  };

  const getImageEmbedding = async (imageElement: HTMLCanvasElement | HTMLImageElement) => {
    const tfimg = tf.browser.fromPixels(imageElement);
    const normalized = tfimg.toFloat().div(tf.scalar(255));
    const batched = normalized.expandDims(0);
    const resized = tf.image.resizeBilinear(batched as tf.Tensor4D, [224, 224]);
    const embedding = model.predict(resized);
    const embeddingData = await embedding.data();
    return embeddingData;
  };

  const cosineSimilarity = (a: number[], b: number[]) => {
    const dotProduct = a.reduce((sum, _, i) => sum + a[i] * b[i], 0);
    const magnitudeA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
    const magnitudeB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
    return dotProduct / (magnitudeA * magnitudeB);
  };

  const updateSimilarityScore = async () => {
    const before = beforeImageCapture();
    const current = captureImage();
    if (!before || !current) return;

    const beforeEmbedding = await getImageEmbedding(before);
    const currentEmbedding = await getImageEmbedding(current);
    const similarityValue = cosineSimilarity(beforeEmbedding, currentEmbedding);
    setSimilarity(similarityValue);
  };

  const captureBefore = () => {
    const captured = captureImage();
    if (captured) {
      setBeforeImageCapture(captured);
      if (beforeImageRef) {
        beforeImageRef.src = captured.toDataURL('image/png');
        beforeImageRef.style.display = 'block';
      }
    }
  };

  const captureAfter = () => {
    if (!beforeImageCapture() || !model) return;
    const captured = captureImage();
    if (captured) {
      setAfterImageCapture(captured);
      updateSimilarityScore();
      setShowLiveFeed(false);
    }
  };

  const reset = () => {
    setBeforeImageCapture(null);
    setAfterImageCapture(null);
    if (beforeImageRef) beforeImageRef.style.display = 'none';
    setSimilarity(0);
    setShowLiveFeed(true);
  };

  const switchCamera = async () => {
    setCurrentFacingMode(prev => prev === 'user' ? 'environment' : 'user');
    await setupCamera();
  };

  createEffect(() => {
    if (beforeImageCapture() && showLiveFeed()) {
      const intervalId = setInterval(updateSimilarityScore, 300);
      onCleanup(() => clearInterval(intervalId));
    }
  });

  onMount(() => {
    window.addEventListener('resize', handleResize);
    setupCamera();
    loadModel();
  });

  onCleanup(() => {
    window.removeEventListener('resize', handleResize);
  });

  return (
    <div class={styles.container}>
      <h1>Before and After Image Comparison</h1>
      {isModelLoading() ? (
        <div class={styles.loadingIndicator}>Loading model... Please wait.</div>
      ) : (
        <>
          <div class={`${styles.imageContainer} ${isPortrait() ? styles.portrait : ''}`}>
            <video
              ref={videoRef}
              autoplay
              playsinline
              style={{ display: showLiveFeed() ? 'block' : 'none' }}
            ></video>
            <img
              ref={beforeImageRef}
              style={{
                display: beforeImageCapture() ? 'block' : 'none',
                opacity: 0.5,
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: '100%',
                "object-fit": "cover"
              }}
            />
            {afterImageCapture() && (
              <img
                src={afterImageCapture()?.toDataURL('image/png')}
                style={{
                  display: 'block',
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  width: '100%',
                  height: '100%',
                  "object-fit": "cover"
                }}
              />
            )}
          </div>
          <div class={styles.buttonContainer}>
            <button onClick={captureBefore} disabled={!!beforeImageCapture() || isModelLoading()}>Capture Before</button>
            <button onClick={captureAfter} disabled={!beforeImageCapture() || !!afterImageCapture() || isModelLoading()}>Capture After</button>
            <button onClick={reset} disabled={!beforeImageCapture() || isModelLoading()}>Reset</button>
            {isMobileDevice() && <button onClick={switchCamera} disabled={isModelLoading()}>Switch Camera</button>}
          </div>
          <Show when={beforeImageCapture()}>
            <div class={styles.similarityContainer}>
              <div
                class={styles.similarityIndicator}
                style={{
                  "background-color": similarity() <= 0.5 ? '#d93025' : similarity() <= 0.7 ? '#f9ab00' : '#1e8e3e'
                }}
              ></div>
              <span class={styles.similarityScore}>Similarity: {Math.round(similarity() * 100)}%</span>
            </div>
          </Show>
        </>
      )}
      {errorMessage() && <div class={styles.errorMessage}>{errorMessage()}</div>}
    </div>
  );
};

export default App;