import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Settings, Activity, Info } from 'lucide-react';
import logoUbb from './assets/v-escudo-color-gradiente.png';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:5000';

const ZERNIKE_MODES = [
  { id: 'Z1', name: 'Z₁ — Pistón (Piston)', min: -Math.PI, max: Math.PI, step: 0.01 },
  { id: 'Z2', name: 'Z₂ — Tip (Tilt X)', min: -6.0, max: 6.0, step: 0.05 },
  { id: 'Z3', name: 'Z₃ — Tilt (Tilt Y)', min: -6.0, max: 6.0, step: 0.05 },
  { id: 'Z4', name: 'Z₄ — Desfoco (Defocus)', min: -5.0, max: 5.0, step: 0.05 },
  { id: 'Z5', name: 'Z₅ — Astigmatismo Oblicuo 45°', min: -5.0, max: 5.0, step: 0.05 },
  { id: 'Z6', name: 'Z₆ — Astigmatismo Vertical 0°', min: -5.0, max: 5.0, step: 0.05 },
  { id: 'Z7', name: 'Z₇ — Coma Horizontal X', min: -5.0, max: 5.0, step: 0.05 },
  { id: 'Z8', name: 'Z₈ — Coma Vertical Y', min: -5.0, max: 5.0, step: 0.05 },
  { id: 'Z9', name: 'Z₉ — Trébol Oblicuo', min: -5.0, max: 5.0, step: 0.05 },
  { id: 'Z10', name: 'Z₁₀ — Trébol Vertical', min: -5.0, max: 5.0, step: 0.05 },
  { id: 'Z11', name: 'Z₁₁ — Aberración Esférica', min: -5.0, max: 5.0, step: 0.05 },
];

function App() {
  const [method, setMethod] = useState('1'); // '1', '2', '3', or '4'
  const [activeModel, setActiveModel] = useState('phase_diversity');
  const [activeTab, setActiveTab] = useState('simulation'); // 'simulation' | 'camera'
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraFps, setCameraFps] = useState(0.0);
  const [cameraCentroid, setCameraCentroid] = useState([640, 480]);
  const [cameraShutter, setCameraShutter] = useState(33.3);
  const [cameraGain, setCameraGain] = useState(0.0);
  const [cameraRoiSize, setCameraRoiSize] = useState(96);

  const [zernikes, setZernikes] = useState({
    Z1: 0.0,
    Z2: 0.0,
    Z3: 0.0,
    Z4: 0.0,
    Z5: 0.0,
    Z6: 0.0,
    Z7: 0.0,
    Z8: 0.0,
    Z9: 0.0,
    Z10: 0.0,
    Z11: 0.0,
  });
  const [d_r0, setDR0] = useState(1.0);
  const [windSpeed, setWindSpeed] = useState(0.5);
  // Solo necesitamos la imagen del panel CNN (PSF); el mapa SLM va directo al canvas GPU
  const [psfImage, setPsfImage] = useState(`${API_BASE}/image/psf?t=${Date.now()}`);
  const [loading, setLoading] = useState(false);
  const [simOnline, setSimOnline] = useState(false);
  const [fps, setFps] = useState(0);
  const [fpsSLM, setFpsSLM] = useState(0);
  const [fpsTurbPsf, setFpsTurbPsf] = useState(0);
  const [fpsCnnPhase, setFpsCnnPhase] = useState(0);
  const [fpsReconPsf, setFpsReconPsf] = useState(0);
  const [fpsKalmanPsf, setFpsKalmanPsf] = useState(0);
  const [avgAccuracy, setAvgAccuracy] = useState(100.0);
  const [avgKalmanAccuracy, setAvgKalmanAccuracy] = useState(100.0);
  const [isModelSwitching, setIsModelSwitching] = useState(false);

  // Parámetros del controlador Kalman/LQG (valores por defecto del modelo AR(1))
  const [kalmanQ, setKalmanQ] = useState(0.002);
  const [kalmanR, setKalmanR] = useState(0.0025);
  const [kalmanDelay, setKalmanDelay] = useState(1);
  const [kalmanUncertainty, setKalmanUncertainty] = useState(0.0);

  const handleKalmanQChange = (val) => {
    setKalmanQ(val);
    axios.post(`${API_BASE}/config`, { kalman_q: val })
      .catch(err => console.error('Error al actualizar kalman_q:', err));
  };

  const handleKalmanRChange = (val) => {
    setKalmanR(val);
    axios.post(`${API_BASE}/config`, { kalman_r: val })
      .catch(err => console.error('Error al actualizar kalman_r:', err));
  };

  const handleKalmanDelayChange = (val) => {
    setKalmanDelay(val);
    axios.post(`${API_BASE}/config`, { kalman_delay: val })
      .catch(err => console.error('Error al actualizar kalman_delay:', err));
  };

  const handleShutterChange = (val) => {
    setCameraShutter(val);
    axios.post(`${API_BASE}/config`, { camera_shutter: parseFloat(val) })
      .catch(err => console.error('Error al actualizar shutter:', err));
  };

  const handleGainChange = (val) => {
    setCameraGain(val);
    axios.post(`${API_BASE}/config`, { camera_gain: parseFloat(val) })
      .catch(err => console.error('Error al actualizar gain:', err));
  };

  const handleRoiSizeChange = (val) => {
    setCameraRoiSize(val);
    axios.post(`${API_BASE}/config`, { camera_roi_size: parseInt(val) })
      .catch(err => console.error('Error al actualizar zoom:', err));
  };

  // ── SISTEMA DE LOGS DESHABILITADO ──────────────────────────────────────────
  const [debugLogs, setDebugLogs] = useState([]);
  const logBufferRef = useRef([]);
  const logFlushTimerRef = useRef(null);

  const addLog = (level, msg) => {
    // Deshabilitado para mejorar rendimiento de la UI
  };

  const debounceRef = useRef(null);
  const slmFramesCountRef = useRef(0);
  const turbPsfFramesCountRef = useRef(0);
  const cnnPhaseFramesCountRef = useRef(0);
  const reconPsfFramesCountRef = useRef(0);
  const kalmanPsfFramesCountRef = useRef(0);
  const accuracyListRef = useRef([]);
  const kalmanAccuracyListRef = useRef([]);

  const canvasRef = useRef(null);        // Canvas GPU: mapa de fase SLM (corrección perfecta)
  const psfCanvasRef = useRef(null);     // Canvas GPU: PSF con turbulencia
  const cnnPhaseCanvasRef = useRef(null); // Canvas GPU: mapa de fase estimado por CNN
  const reconPsfCanvasRef = useRef(null); // Canvas GPU: PSF reconstruida por CNN
  const kalmanPsfCanvasRef = useRef(null); // Canvas GPU: PSF reconstruida por CNN + Kalman/LQG
  const drawRef = useRef(null);          // Función de dibujo del SLM (ref para recursividad)
  const drawPsfRef = useRef(null);       // Función de dibujo de la PSF con turbulencia
  const drawCnnPhaseRef = useRef(null);  // Función de dibujo del mapa de fase CNN
  const drawReconPsfRef = useRef(null);  // Función de dibujo de la PSF reconstruida
  const drawKalmanPsfRef = useRef(null);  // Función de dibujo de la PSF filtrada Kalman
  const loopIdRef = useRef(null);        // ID de requestAnimationFrame del bucle SLM
  const loopPsfIdRef = useRef(null);     // ID de requestAnimationFrame del bucle PSF
  const loopCnnPhaseIdRef = useRef(null);
  const loopReconPsfIdRef = useRef(null);
  const loopKalmanPsfIdRef = useRef(null);
  // Flags anti-acumulación: evitan lanzar un nuevo fetch si el anterior aún está en vuelo
  const cnnPhaseInFlightRef = useRef(false);
  const reconPsfInFlightRef = useRef(false);
  const kalmanPsfInFlightRef = useRef(false);

  const methodRef = useRef(method);
  useEffect(() => {
    methodRef.current = method;
  }, [method]);

  const isModelSwitchingRef = useRef(isModelSwitching);
  useEffect(() => {
    isModelSwitchingRef.current = isModelSwitching;
  }, [isModelSwitching]);

  const activeTabRef = useRef(activeTab);
  useEffect(() => {
    activeTabRef.current = activeTab;
  }, [activeTab]);

  const cameraCentroidRef = useRef(cameraCentroid);
  useEffect(() => {
    cameraCentroidRef.current = cameraCentroid;
  }, [cameraCentroid]);

  const cameraCanvasRef = useRef(null);
  const drawCameraRef = useRef(null);
  const loopCameraIdRef = useRef(null);

  // ── RENDER GPU: fetch bytes raw del simulador y pinta en canvas ──────────────
  // La física (Prysm) vive en el simulador. El browser solo recibe los píxeles.
  // Los cuatro canvas usan el mismo patrón: fetch raw → putImageData → GPU integrada.

  // Canvas 1: Mapa de fase SLM (corrección perfecta) — escala de grises 1 byte/px
  drawRef.current = () => {
    fetch(`${API_BASE}/image/distorted-raw`)
      .then(res => res.arrayBuffer())
      .then(buffer => {
        const gray = new Uint8Array(buffer);          // 640×360 = 230 400 bytes
        const rgba = new Uint8ClampedArray(gray.length * 4);
        for (let i = 0; i < gray.length; i++) {
          rgba[i * 4] = gray[i];  // R
          rgba[i * 4 + 1] = gray[i];  // G
          rgba[i * 4 + 2] = gray[i];  // B
          rgba[i * 4 + 3] = 255;       // A
        }
        const ctx = canvasRef.current?.getContext('2d');
        if (ctx) ctx.putImageData(new ImageData(rgba, 640, 360), 0, 0);
        slmFramesCountRef.current += 1;
        if (methodRef.current === '2' || activeTabRef.current === 'camera') {
          setTimeout(() => {
            loopIdRef.current = requestAnimationFrame(drawRef.current);
          }, activeTabRef.current === 'camera' ? 66 : 40);
        }
      })
      .catch(() => {
        if (methodRef.current === '2' || activeTabRef.current === 'camera') {
          setTimeout(() => {
            loopIdRef.current = requestAnimationFrame(drawRef.current);
          }, activeTabRef.current === 'camera' ? 66 : 40);
        }
      });
  };

  // Canvas Cámara Preview: Muestra la vista de 320x240 con la cruz del centroide
  drawCameraRef.current = () => {
    if (activeTabRef.current !== 'camera') return;
    fetch(`${API_BASE}/image/camera-raw`)
      .then(res => {
        if (!res.ok) throw new Error("Cámara offline");
        return res.arrayBuffer();
      })
      .then(buffer => {
        const gray = new Uint8Array(buffer);
        const rgba = new Uint8ClampedArray(gray.length * 4);
        for (let i = 0; i < gray.length; i++) {
          rgba[i * 4] = gray[i];
          rgba[i * 4 + 1] = gray[i];
          rgba[i * 4 + 2] = gray[i];
          rgba[i * 4 + 3] = 255;
        }
        const ctx = cameraCanvasRef.current?.getContext('2d');
        if (ctx) {
          ctx.putImageData(new ImageData(rgba, 320, 240), 0, 0);
          
          // Dibujar el centroide detectado
          const cx = cameraCentroidRef.current[0] / 4;
          const cy = cameraCentroidRef.current[1] / 4;
          ctx.strokeStyle = '#ef4444';
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.moveTo(cx - 8, cy);
          ctx.lineTo(cx + 8, cy);
          ctx.moveTo(cx, cy - 8);
          ctx.lineTo(cx, cy + 8);
          ctx.stroke();
        }
        setTimeout(() => {
          if (activeTabRef.current === 'camera') {
            loopCameraIdRef.current = requestAnimationFrame(drawCameraRef.current);
          }
        }, 66);
      })
      .catch(() => {
        setTimeout(() => {
          if (activeTabRef.current === 'camera') {
            loopCameraIdRef.current = requestAnimationFrame(drawCameraRef.current);
          }
        }, 100);
      });
  };

  // Canvas 2: PSF con turbulencia (plano focal con aberración) — Comentado temporalmente
  drawPsfRef.current = () => {
    if (methodRef.current === '2' || activeTabRef.current === 'camera') {
      loopPsfIdRef.current = requestAnimationFrame(drawPsfRef.current);
    }
  };

  // Canvas 5: PSF corregida por CNN + Kalman/LQG — RGB 3 bytes/px
  drawKalmanPsfRef.current = () => {
    if (isModelSwitchingRef.current) {
      if (methodRef.current === '2' || activeTabRef.current === 'camera') {
        setTimeout(() => {
          loopKalmanPsfIdRef.current = requestAnimationFrame(drawKalmanPsfRef.current);
        }, activeTabRef.current === 'camera' ? 66 : 40);
      }
      return;
    }
    if (kalmanPsfInFlightRef.current) {
      if (methodRef.current === '2' || activeTabRef.current === 'camera') {
        setTimeout(() => {
          loopKalmanPsfIdRef.current = requestAnimationFrame(drawKalmanPsfRef.current);
        }, activeTabRef.current === 'camera' ? 66 : 40);
      }
      return;
    }
    kalmanPsfInFlightRef.current = true;
    fetch(`${API_BASE}/image/kalman-psf-raw`)
      .then(res => {
        if (!res.ok) {
          kalmanPsfInFlightRef.current = false;
          if (methodRef.current === '2' || activeTabRef.current === 'camera') {
            setTimeout(() => {
              loopKalmanPsfIdRef.current = requestAnimationFrame(drawKalmanPsfRef.current);
            }, activeTabRef.current === 'camera' ? 66 : 40);
          }
          throw new Error(`KALMAN_PSF HTTP ${res.status}`);
        }
        const acc = res.headers.get('X-CNN-Accuracy');
        if (acc) {
          if (methodRef.current === '2' || activeTabRef.current === 'camera') {
            kalmanAccuracyListRef.current.push(parseFloat(acc));
          } else {
            setAvgKalmanAccuracy(parseFloat(acc));
          }
        }
        return res.arrayBuffer();
      })
      .then(buffer => {
        const rgb = new Uint8Array(buffer);
        const numPixels = rgb.length / 3;
        const rgba = new Uint8ClampedArray(numPixels * 4);
        for (let i = 0; i < numPixels; i++) {
          rgba[i * 4] = rgb[i * 3];
          rgba[i * 4 + 1] = rgb[i * 3 + 1];
          rgba[i * 4 + 2] = rgb[i * 3 + 2];
          rgba[i * 4 + 3] = 255;
        }
        const ctx = kalmanPsfCanvasRef.current?.getContext('2d');
        if (ctx) ctx.putImageData(new ImageData(rgba, 640, 360), 0, 0);
        kalmanPsfFramesCountRef.current += 1;
        kalmanPsfInFlightRef.current = false;
        if (methodRef.current === '2' || activeTabRef.current === 'camera') {
          setTimeout(() => {
            loopKalmanPsfIdRef.current = requestAnimationFrame(drawKalmanPsfRef.current);
          }, activeTabRef.current === 'camera' ? 66 : 40);
        }
      })
      .catch(() => {
        kalmanPsfInFlightRef.current = false;
        if (methodRef.current === '2' || activeTabRef.current === 'camera') {
          setTimeout(() => {
            loopKalmanPsfIdRef.current = requestAnimationFrame(drawKalmanPsfRef.current);
          }, activeTabRef.current === 'camera' ? 66 : 40);
        }
      });
  };

  // Canvas 3: Mapa de fase CNN (corrección estimada) — escala de grises 1 byte/px
  // Throttling adaptativo + flag inFlight para no acumular peticiones pendientes.
  drawCnnPhaseRef.current = () => {
    if (isModelSwitchingRef.current) {
      if (methodRef.current === '2' || activeTabRef.current === 'camera') {
        setTimeout(() => {
          loopCnnPhaseIdRef.current = requestAnimationFrame(drawCnnPhaseRef.current);
        }, activeTabRef.current === 'camera' ? 66 : 40);
      }
      return;
    }
    if (cnnPhaseInFlightRef.current) {
      if (methodRef.current === '2' || activeTabRef.current === 'camera') {
        setTimeout(() => {
          loopCnnPhaseIdRef.current = requestAnimationFrame(drawCnnPhaseRef.current);
        }, activeTabRef.current === 'camera' ? 66 : 40);
      }
      return;
    }
    cnnPhaseInFlightRef.current = true;
    fetch(`${API_BASE}/image/cnn-phase-raw`)
      .then(res => {
        if (!res.ok) {
          cnnPhaseInFlightRef.current = false;
          if (methodRef.current === '2' || activeTabRef.current === 'camera') {
            setTimeout(() => {
              loopCnnPhaseIdRef.current = requestAnimationFrame(drawCnnPhaseRef.current);
            }, activeTabRef.current === 'camera' ? 66 : 40);
          }
          throw new Error(`CNN_PHASE HTTP ${res.status}`);
        }
        const acc = res.headers.get('X-CNN-Accuracy');
        if (acc) {
          if (methodRef.current === '2' || activeTabRef.current === 'camera') {
            accuracyListRef.current.push(parseFloat(acc));
          } else {
            setAvgAccuracy(parseFloat(acc));
          }
        }
        return res.arrayBuffer();
      })
      .then(buffer => {
        const gray = new Uint8Array(buffer);
        const rgba = new Uint8ClampedArray(gray.length * 4);
        for (let i = 0; i < gray.length; i++) {
          rgba[i * 4] = gray[i];
          rgba[i * 4 + 1] = gray[i];
          rgba[i * 4 + 2] = gray[i];
          rgba[i * 4 + 3] = 255;
        }
        const ctx = cnnPhaseCanvasRef.current?.getContext('2d');
        if (ctx) ctx.putImageData(new ImageData(rgba, 640, 360), 0, 0);
        cnnPhaseFramesCountRef.current += 1;
        cnnPhaseInFlightRef.current = false;
        if (methodRef.current === '2' || activeTabRef.current === 'camera') {
          setTimeout(() => {
            loopCnnPhaseIdRef.current = requestAnimationFrame(drawCnnPhaseRef.current);
          }, activeTabRef.current === 'camera' ? 66 : 40);
        }
      })
      .catch(() => {
        cnnPhaseInFlightRef.current = false;
        if (methodRef.current === '2' || activeTabRef.current === 'camera') {
          setTimeout(() => {
            loopCnnPhaseIdRef.current = requestAnimationFrame(drawCnnPhaseRef.current);
          }, activeTabRef.current === 'camera' ? 66 : 40);
        }
      });
  };

  // Canvas 4: PSF reconstruida (corregida por CNN) — RGB 3 bytes/px
  // Throttling adaptativo + flag inFlight para no acumular peticiones pendientes.
  drawReconPsfRef.current = () => {
    if (isModelSwitchingRef.current) {
      if (methodRef.current === '2' || activeTabRef.current === 'camera') {
        setTimeout(() => {
          loopReconPsfIdRef.current = requestAnimationFrame(drawReconPsfRef.current);
        }, activeTabRef.current === 'camera' ? 66 : 40);
      }
      return;
    }
    if (reconPsfInFlightRef.current) {
      if (methodRef.current === '2' || activeTabRef.current === 'camera') {
        setTimeout(() => {
          loopReconPsfIdRef.current = requestAnimationFrame(drawReconPsfRef.current);
        }, activeTabRef.current === 'camera' ? 66 : 40);
      }
      return;
    }
    reconPsfInFlightRef.current = true;
    fetch(`${API_BASE}/image/reconstructed-psf-raw`)
      .then(res => {
        if (!res.ok) {
          reconPsfInFlightRef.current = false;
          if (methodRef.current === '2' || activeTabRef.current === 'camera') {
            setTimeout(() => {
              loopReconPsfIdRef.current = requestAnimationFrame(drawReconPsfRef.current);
            }, activeTabRef.current === 'camera' ? 66 : 40);
          }
          throw new Error(`RECON_PSF HTTP ${res.status}`);
        }
        const acc = res.headers.get('X-CNN-Accuracy');
        if (acc) {
          if (methodRef.current === '2' || activeTabRef.current === 'camera') {
            accuracyListRef.current.push(parseFloat(acc));
          } else {
            setAvgAccuracy(parseFloat(acc));
          }
        }
        return res.arrayBuffer();
      })
      .then(buffer => {
        const rgb = new Uint8Array(buffer);
        const numPixels = rgb.length / 3;
        const rgba = new Uint8ClampedArray(numPixels * 4);
        for (let i = 0; i < numPixels; i++) {
          rgba[i * 4] = rgb[i * 3];
          rgba[i * 4 + 1] = rgb[i * 3 + 1];
          rgba[i * 4 + 2] = rgb[i * 3 + 2];
          rgba[i * 4 + 3] = 255;
        }
        const ctx = reconPsfCanvasRef.current?.getContext('2d');
        if (ctx) ctx.putImageData(new ImageData(rgba, 640, 360), 0, 0);
        reconPsfFramesCountRef.current += 1;
        reconPsfInFlightRef.current = false;
        if (methodRef.current === '2' || activeTabRef.current === 'camera') {
          setTimeout(() => {
            loopReconPsfIdRef.current = requestAnimationFrame(drawReconPsfRef.current);
          }, activeTabRef.current === 'camera' ? 66 : 40);
        }
      })
      .catch(() => {
        reconPsfInFlightRef.current = false;
        if (methodRef.current === '2' || activeTabRef.current === 'camera') {
          setTimeout(() => {
            loopReconPsfIdRef.current = requestAnimationFrame(drawReconPsfRef.current);
          }, activeTabRef.current === 'camera' ? 66 : 40);
        }
      });
  };

  // Verificar estado del simulador
  useEffect(() => {
    axios.get(`${API_BASE}/status`)
      .then((res) => {
        setSimOnline(true);
        if (res.data && res.data.state) {
          if (res.data.state.method) setMethod(res.data.state.method);
          if (res.data.state.active_model) setActiveModel(res.data.state.active_model);
          if (res.data.state.d_r0) setDR0(res.data.state.d_r0);
          if (res.data.state.wind_speed) setWindSpeed(res.data.state.wind_speed);
          if (res.data.state.zernikes) setZernikes(res.data.state.zernikes);
          if (res.data.state.camera_shutter !== undefined) setCameraShutter(res.data.state.camera_shutter);
          if (res.data.state.camera_gain !== undefined) setCameraGain(res.data.state.camera_gain);
          if (res.data.state.camera_roi_size !== undefined) setCameraRoiSize(res.data.state.camera_roi_size);
        }
        // Pintar el primer frame de cada canvas
        setTimeout(() => {
          if (drawRef.current) drawRef.current();
          if (drawPsfRef.current) drawPsfRef.current();
          if (drawCnnPhaseRef.current) drawCnnPhaseRef.current();
          if (drawReconPsfRef.current) drawReconPsfRef.current();
        }, 150);
      })
      .catch(() => setSimOnline(false));
  }, []);

  const handleModelChange = async (model) => {
    addLog('INFO', `MODEL_SWITCH → ${model} | método=${methodRef.current}`);
    setActiveModel(model);
    // Actualizar ref inmediatamente (no esperar al re-render del useEffect)
    isModelSwitchingRef.current = true;
    setIsModelSwitching(true);
    addLog('INFO', 'MODEL_SWITCH: isModelSwitchingRef=true | loops CNN pausados');
    try {
      await axios.post(`${API_BASE}/config`, { active_model: model });
      addLog('INFO', 'MODEL_SWITCH: POST /config OK — esperando 800ms estabilización');
      setTimeout(() => {
        addLog('INFO', `MODEL_SWITCH: 800ms fin | cnnInFlight=${cnnPhaseInFlightRef.current} reconInFlight=${reconPsfInFlightRef.current}`);
        // 1. Resetear flags inFlight ANTES de desbloquear
        cnnPhaseInFlightRef.current = false;
        reconPsfInFlightRef.current = false;
        kalmanPsfInFlightRef.current = false;
        // 2. Desbloquear loops CNN
        isModelSwitchingRef.current = false;
        setIsModelSwitching(false);
        addLog('INFO', `MODEL_SWITCH: desbloqueado | método=${methodRef.current}`);
        // 3. En método 2 los loops se auto-reanudan; en método 1 pedir frames manualmente
        if (methodRef.current !== '2') {
          addLog('DEBUG', 'MODEL_SWITCH: método 1 → pedir frames manuales');
          if (drawRef.current) drawRef.current();
          if (drawPsfRef.current) drawPsfRef.current();
          if (drawCnnPhaseRef.current) drawCnnPhaseRef.current();
          if (drawReconPsfRef.current) drawReconPsfRef.current();
          if (drawKalmanPsfRef.current) drawKalmanPsfRef.current();
        } else {
          addLog('DEBUG', 'MODEL_SWITCH: método 2 → loops se reanudan solos');
        }
      }, 800);
    } catch (err) {
      addLog('ERROR', `MODEL_SWITCH POST FAIL: ${err.message}`);
      cnnPhaseInFlightRef.current = false;
      reconPsfInFlightRef.current = false;
      kalmanPsfInFlightRef.current = false;
      isModelSwitchingRef.current = false;
      setIsModelSwitching(false);
    }
  };


  // Manejar loops automáticos para el Método 2 (Estocástico) y Modo Cámara Real
  useEffect(() => {
    let fpsIntervalId = null;
    let accuracyIntervalId = null;
    let zernikeIntervalId = null;

    if (method === '2' || activeTab === 'camera') {
      if (method === '2') {
        axios.post(`${API_BASE}/config`, { method: '2', d_r0, wind_speed: windSpeed })
          .catch(err => console.error("Error al iniciar modo estocástico:", err));
      }

      slmFramesCountRef.current = 0;
      turbPsfFramesCountRef.current = 0;
      cnnPhaseFramesCountRef.current = 0;
      reconPsfFramesCountRef.current = 0;
      kalmanPsfFramesCountRef.current = 0;
      accuracyListRef.current = [];
      kalmanAccuracyListRef.current = [];

      // Arrancar los cinco bucles GPU: fetch raw → canvas (auto-regulado por requestAnimationFrame)
      loopIdRef.current = requestAnimationFrame(drawRef.current);
      loopPsfIdRef.current = requestAnimationFrame(drawPsfRef.current);
      loopCnnPhaseIdRef.current = requestAnimationFrame(drawCnnPhaseRef.current);
      loopReconPsfIdRef.current = requestAnimationFrame(drawReconPsfRef.current);
      loopKalmanPsfIdRef.current = requestAnimationFrame(drawKalmanPsfRef.current);

      if (activeTab === 'camera') {
        if (loopCameraIdRef.current) cancelAnimationFrame(loopCameraIdRef.current);
        loopCameraIdRef.current = requestAnimationFrame(drawCameraRef.current);
      }

      // Contador de FPS visible en la UI para cada canal
      fpsIntervalId = setInterval(() => {
        setFpsSLM(slmFramesCountRef.current);
        setFpsTurbPsf(turbPsfFramesCountRef.current);
        setFpsCnnPhase(cnnPhaseFramesCountRef.current);
        setFpsReconPsf(reconPsfFramesCountRef.current);
        setFpsKalmanPsf(kalmanPsfFramesCountRef.current);
        setFps(slmFramesCountRef.current);

        slmFramesCountRef.current = 0;
        turbPsfFramesCountRef.current = 0;
        cnnPhaseFramesCountRef.current = 0;
        reconPsfFramesCountRef.current = 0;
        kalmanPsfFramesCountRef.current = 0;
      }, 1000);

      // Promediar precisión de la CNN y Kalman cada 2 segundos
      accuracyIntervalId = setInterval(() => {
        if (accuracyListRef.current.length > 0) {
          const sum = accuracyListRef.current.reduce((a, b) => a + b, 0);
          const avg = sum / accuracyListRef.current.length;
          setAvgAccuracy(avg);
          accuracyListRef.current = [];
        }
        if (kalmanAccuracyListRef.current.length > 0) {
          const sum = kalmanAccuracyListRef.current.reduce((a, b) => a + b, 0);
          const avg = sum / kalmanAccuracyListRef.current.length;
          setAvgKalmanAccuracy(avg);
          kalmanAccuracyListRef.current = [];
        }
      }, 2000);

      // Actualizar sliders y telemetría con datos dinámicos del backend
      zernikeIntervalId = setInterval(() => {
        axios.get(`${API_BASE}/status`)
          .then((res) => {
            if (res.data && res.data.state) {
              if (res.data.state.zernikes) {
                setZernikes(res.data.state.zernikes);
              }
              if (res.data.state.kalman_uncertainty !== undefined) {
                setKalmanUncertainty(res.data.state.kalman_uncertainty);
              }
              if (res.data.state.camera_active !== undefined) {
                setCameraActive(res.data.state.camera_active);
              }
              if (res.data.state.camera_fps !== undefined) {
                setCameraFps(res.data.state.camera_fps);
              }
              if (res.data.state.camera_centroid !== undefined) {
                setCameraCentroid(res.data.state.camera_centroid);
              }
            }
          })
          .catch(err => console.error("Error al actualizar telemetría:", err));
      }, 1000);
    } else {
      // Detener los cinco bucles GPU y volver a modo manual
      if (loopIdRef.current) cancelAnimationFrame(loopIdRef.current);
      if (loopPsfIdRef.current) cancelAnimationFrame(loopPsfIdRef.current);
      if (loopCnnPhaseIdRef.current) cancelAnimationFrame(loopCnnPhaseIdRef.current);
      if (loopReconPsfIdRef.current) cancelAnimationFrame(loopReconPsfIdRef.current);
      if (loopKalmanPsfIdRef.current) cancelAnimationFrame(loopKalmanPsfIdRef.current);
      if (loopCameraIdRef.current) cancelAnimationFrame(loopCameraIdRef.current);
      
      // Resetear flags inFlight
      cnnPhaseInFlightRef.current = false;
      reconPsfInFlightRef.current = false;
      kalmanPsfInFlightRef.current = false;

      setFpsSLM(0);
      setFpsTurbPsf(0);
      setFpsCnnPhase(0);
      setFpsReconPsf(0);
      setFpsKalmanPsf(0);

      axios.post(`${API_BASE}/config`, { method: '1' })
        .then(() => {
          if (drawRef.current) drawRef.current();
          if (drawPsfRef.current) drawPsfRef.current();
          if (drawCnnPhaseRef.current) drawCnnPhaseRef.current();
          if (drawReconPsfRef.current) drawReconPsfRef.current();
          if (drawKalmanPsfRef.current) drawKalmanPsfRef.current();
        })
        .catch(err => console.error("Error al volver a modo manual:", err));
    }

    return () => {
      if (loopIdRef.current) cancelAnimationFrame(loopIdRef.current);
      if (loopPsfIdRef.current) cancelAnimationFrame(loopPsfIdRef.current);
      if (loopCnnPhaseIdRef.current) cancelAnimationFrame(loopCnnPhaseIdRef.current);
      if (loopReconPsfIdRef.current) cancelAnimationFrame(loopReconPsfIdRef.current);
      if (loopKalmanPsfIdRef.current) cancelAnimationFrame(loopKalmanPsfIdRef.current);
      if (loopCameraIdRef.current) cancelAnimationFrame(loopCameraIdRef.current);
      if (fpsIntervalId) clearInterval(fpsIntervalId);
      if (accuracyIntervalId) clearInterval(accuracyIntervalId);
      if (zernikeIntervalId) clearInterval(zernikeIntervalId);
    };
  }, [method, activeTab]);

  const updateZernike = (id, value) => {
    const nextZernikes = { ...zernikes, [id]: parseFloat(value) };
    setZernikes(nextZernikes);

    clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(async () => {
      try {
        await axios.post(`${API_BASE}/config`, { zernikes: nextZernikes });
        // Modo manual: pide un frame a cada canvas
        drawRef.current();
        drawPsfRef.current();
        drawCnnPhaseRef.current();
        drawReconPsfRef.current();
        drawKalmanPsfRef.current();
      } catch (err) {
        console.error('Error al actualizar Zernikes:', err);
      }
    }, 80);
  };

  const handleDR0Change = (val) => {
    setDR0(val);
    if (method === '2') {
      axios.post(`${API_BASE}/config`, { d_r0: parseFloat(val) })
        .catch(err => console.error("Error al actualizar D/r0:", err));
    }
  };

  const handleWindSpeedChange = (val) => {
    setWindSpeed(val);
    if (method === '2') {
      axios.post(`${API_BASE}/config`, { wind_speed: parseFloat(val) })
        .catch(err => console.error("Error al actualizar wind_speed:", err));
    }
  };

  const resetAllZernikes = () => {
    const cleared = {
      Z1: 0.0, Z2: 0.0, Z3: 0.0, Z4: 0.0, Z5: 0.0, Z6: 0.0, Z7: 0.0, Z8: 0.0, Z9: 0.0, Z10: 0.0, Z11: 0.0
    };
    setZernikes(cleared);
    axios.post(`${API_BASE}/config`, { zernikes: cleared })
      .then(() => drawRef.current())
      .catch(err => console.error('Error al resetear Zernikes:', err));
  };

  const handleCalibrate = async () => {
    setLoading(true);
    try {
      await axios.post('http://localhost:5002/calibrate');
      alert('Calibración completada con éxito.');
    } catch {
      alert('Error de conexión con el controlador.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen p-6 flex flex-col gap-6 font-sans">

      {/* ── NAVBAR PREMIUM (Panel científico) ── */}
      <nav className="rounded-xl border border-[#223B53] bg-[#0F2433] px-6 py-4">
        <div className="flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-md border border-[#2A3E53] bg-[#0F2433]">
              <Activity className="text-[#4EA3FF]" size={18} />
            </div>
            <div className="space-y-0">
              <div className="flex items-center gap-2">
                <h1 className="text-sm font-semibold tracking-[0.02em] text-[#E8EEF7]">
                  Sistema de Control de Óptica Adaptativa
                </h1>
                <span className="rounded border border-[#223B53] bg-[#0F2433] px-2 py-0.5 text-[10px] uppercase tracking-[0.2em] text-[#4EA3FF]">
                  v2.0
                </span>
              </div>
              <p className="text-[10px] text-[#8FA0B3]">HOLOEYE PLUTO 2.1 · UBB Chile</p>
            </div>
          </div>

          {/* Centro: selector grande integrado */}
          <div className="flex items-center gap-3">
            <div className="rounded-lg border border-[#223B53] bg-[#101B2E] p-1.5 flex items-center">
              <button
                onClick={() => setActiveTab('simulation')}
                className={`px-5 py-2 rounded-lg text-sm font-semibold transition-colors duration-150 ${
                  activeTab === 'simulation' ? 'bg-[#0F2433] text-[#E8EEF7] border border-[#2A3E53]' : 'text-[#8FA0B3] hover:text-[#E8EEF7]'
                }`}
              >
                Modo Simulación
              </button>
              <div className="w-1" />
              {/* MÓDULO CÁMARA REAL — visible pero deshabilitado hasta completar desarrollo */}
              <div className="relative group">
                <button
                  disabled
                  className="flex items-center gap-2 px-5 py-2 rounded-lg text-sm font-semibold
                             text-[#8FA0B3]/50 cursor-not-allowed select-none"
                  title="Módulo de Cámara Real — En desarrollo"
                >
                  <div className="h-2 w-2 rounded-full bg-[#6B7280]/40" />
                  Cámara Real
                  <span className="ml-1 px-1.5 py-0.5 rounded text-[9px] font-bold uppercase tracking-wider
                                   bg-amber-500/15 text-amber-400 border border-amber-500/30">
                    WIP
                  </span>
                </button>
                {/* Tooltip al hacer hover */}
                <div className="pointer-events-none absolute left-1/2 -translate-x-1/2 top-full mt-2 z-50
                                w-44 rounded-lg border border-[#223B53] bg-[#0A1628] px-3 py-2 shadow-xl
                                opacity-0 group-hover:opacity-100 transition-opacity duration-150">
                  <p className="text-[10px] text-amber-400 font-semibold mb-0.5">En desarrollo</p>
                  <p className="text-[10px] text-[#8FA0B3] leading-snug">
                    El módulo de cámara infrarroja estará disponible próximamente.
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Derecha: estado compacto + logo (sin texto largo del simulador) */}
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-3 rounded-md border border-[#223B53] bg-[#101B2E] px-3 py-2">
              <StatusBadge online={simOnline} label={''} />
              <div className="h-5 w-px bg-[#223B53]" />
              <img src={logoUbb} alt="UBB" className="h-8 opacity-80" />
            </div>
          </div>
        </div>
      </nav>

      {/* ── CONTENIDO PRINCIPAL ── */}
      <main className="flex flex-col md:flex-row gap-6 items-start w-full">

        {/* Barra lateral de controles */}
        <aside className="w-full md:w-80 flex flex-col gap-4 shrink-0">
          {activeTab === 'simulation' ? (
            <div className="rounded-xl border border-[#223B53] bg-[#0F2433] p-5 flex flex-col">
              <h2 className="flex items-center gap-2 text-sm font-medium text-white mb-4">
                <Settings size={16} className="text-zinc-400" /> Configuración General
              </h2>

              {/* Método de Generación (botones) */}
              <div className="flex flex-col gap-2 mb-4">
                <label className="text-xs font-semibold text-[#8FA0B3] uppercase tracking-wider">
                  Método de Generación
                </label>
                <div className="flex gap-2">
                  <button
                    onClick={() => setMethod('1')}
                    className={`flex-1 px-3 py-2 rounded-lg text-sm font-semibold transition-colors ${method === '1' ? 'bg-[#0F2433] text-[#4EA3FF] border border-[#223B53]' : 'text-[#8FA0B3] hover:text-[#E8EEF7] bg-[#101B2E] border border-[#223B53]'}`}
                  >
                    1 · Zernike (Determinista)
                  </button>
                  <button
                    onClick={() => setMethod('2')}
                    className={`flex-1 px-3 py-2 rounded-lg text-sm font-semibold transition-colors ${method === '2' ? 'bg-[#0F2433] text-[#4EA3FF] border border-[#223B53]' : 'text-[#8FA0B3] hover:text-[#E8EEF7] bg-[#101B2E] border border-[#223B53]'}`}
                  >
                    2 · Kolmogorov (Estocástico)
                  </button>
                </div>
              </div>

              {/* Modelo de Red Neuronal (botones) */}
              <div className="flex flex-col gap-2 mb-4">
                <label className="text-xs font-semibold text-[#8FA0B3] uppercase tracking-wider">
                  Modelo de Red Neuronal (CNN)
                </label>
                <div className="flex gap-2">
                  <button
                    onClick={() => handleModelChange('phase_diversity')}
                    className={`flex-1 px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors ${activeModel === 'phase_diversity' ? 'bg-[#0F2433] text-[#4EA3FF] border border-[#223B53]' : 'text-[#8FA0B3] hover:text-[#E8EEF7] bg-[#101B2E] border border-[#223B53]'}`}
                  >
                    Modelo A
                  </button>
                  <button
                    onClick={() => handleModelChange('resnet10')}
                    className={`flex-1 px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors ${activeModel === 'resnet10' ? 'bg-[#0F2433] text-[#4EA3FF] border border-[#223B53]' : 'text-[#8FA0B3] hover:text-[#E8EEF7] bg-[#101B2E] border border-[#223B53]'}`}
                  >
                    ResNet-10
                  </button>
                  <button
                    onClick={() => handleModelChange('resnet18')}
                    className={`flex-1 px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors ${activeModel === 'resnet18' ? 'bg-[#0F2433] text-[#4EA3FF] border border-[#223B53]' : 'text-[#8FA0B3] hover:text-[#E8EEF7] bg-[#101B2E] border border-[#223B53]'}`}
                  >
                    ResNet-18
                  </button>
                </div>
              </div>

              {/* Sección Dinámica: 11 Polinomios de Zernike */}
              {method === '1' ? (
                <div className="border-t border-[#223B53] pt-4 flex flex-col gap-3 animate-in fade-in duration-200">
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-[10px] font-bold text-zinc-400 uppercase tracking-wider">
                      Modos Activos (Noll Zernike)
                    </span>
                    <button
                      onClick={resetAllZernikes}
                      className="text-[10px] text-blue-500 hover:text-blue-400 font-semibold transition-colors"
                    >
                      Resetear todo
                    </button>
                  </div>

                  {/* Contenedor scrollable con scrollbar personalizada */}
                  <div className="max-h-[320px] overflow-y-auto pr-1 space-y-3 zernike-scroll">
                    {ZERNIKE_MODES.map((mode) => (
                      <div key={mode.id} className="border-b border-[#223B53]/40 pb-3 last:border-0 last:pb-0">
                        {/* Nombre y valor actual */}
                        <div className="flex justify-between items-center mb-1.5">
                          <span className="text-[10px] font-mono text-[#8FA0B3]">
                            {mode.name}
                          </span>
                          <input
                            type="number"
                            value={parseFloat(zernikes[mode.id].toFixed(4))}
                            onChange={(e) => {
                              let val = parseFloat(e.target.value);
                              if (!isNaN(val)) {
                                val = Math.max(mode.min, Math.min(mode.max, val));
                                updateZernike(mode.id, val);
                              }
                            }}
                            step={0.01}
                            min={mode.min}
                            max={mode.max}
                            className="w-16 bg-[#101B2E] border border-[#223B53] text-[#4EA3FF] font-mono font-bold text-[10px] text-right rounded px-1 py-0.5 outline-none focus:border-[#2A3E53] [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
                          />
                        </div>
                        {/* Slider */}
                        <ZernikeSlider
                          value={zernikes[mode.id]}
                          onChange={(val) => updateZernike(mode.id, val)}
                          min={mode.min}
                          max={mode.max}
                          step={mode.step}
                        />
                        {/* Botones de ajuste fino */}
                        <div className="flex gap-1 mt-1.5">
                          <button
                            className="zernike-step-btn flex-1"
                            onClick={() => updateZernike(mode.id, Math.max(mode.min, parseFloat((zernikes[mode.id] - mode.step * 5).toFixed(4))))}
                            title={`−${(mode.step * 5).toFixed(2)}`}
                          >−</button>
                          <button
                            className="zernike-step-btn flex-1"
                            onClick={() => updateZernike(mode.id, 0)}
                            title="Resetear"
                            style={{ fontSize: '9px', color: '#71717a' }}
                          >RST</button>
                          <button
                            className="zernike-step-btn flex-1"
                            onClick={() => updateZernike(mode.id, Math.min(mode.max, parseFloat((zernikes[mode.id] + mode.step * 5).toFixed(4))))}
                            title={`+${(mode.step * 5).toFixed(2)}`}
                          >+</button>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : method === '2' ? (
                <div className="border-t border-[#223B53] pt-4 flex flex-col gap-4 animate-in fade-in duration-200">
                  <div className="text-[10px] font-bold text-zinc-400 uppercase tracking-wider mb-1">
                    Parámetros de Turbulencia (Método 2)
                  </div>

                  {/* Sliders for D/r0 */}
                  <div className="flex flex-col gap-1.5">
                    <div className="flex justify-between items-center text-[10px]">
                      <span className="text-zinc-300 font-mono">Fuerza Turbulencia (D/r₀)</span>
                      <input
                        type="number"
                        value={parseFloat(d_r0.toFixed(4))}
                        onChange={(e) => {
                          let val = parseFloat(e.target.value);
                          if (!isNaN(val)) {
                            val = Math.max(0.1, Math.min(6.0, val));
                            handleDR0Change(val);
                          }
                        }}
                        step={0.01}
                        min={0.1}
                        max={6.0}
                        className="w-16 bg-zinc-900 border border-zinc-800 text-blue-400 font-mono font-bold text-[10px] text-right rounded px-1 py-0.5 outline-none focus:border-blue-500/50 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
                      />
                    </div>
                    <ZernikeSlider
                      value={d_r0}
                      onChange={handleDR0Change}
                      min={0.1}
                      max={6.0}
                      step={0.1}
                    />
                  </div>

                  {/* Sliders for Wind Speed */}
                  <div className="flex flex-col gap-1.5">
                    <div className="flex justify-between items-center text-[10px]">
                      <span className="text-zinc-300 font-mono">Velocidad de Evolución (Viento)</span>
                      <input
                        type="number"
                        value={parseFloat(windSpeed.toFixed(4))}
                        onChange={(e) => {
                          let val = parseFloat(e.target.value);
                          if (!isNaN(val)) {
                            val = Math.max(0.0, Math.min(1.0, val));
                            handleWindSpeedChange(val);
                          }
                        }}
                        step={0.01}
                        min={0.0}
                        max={1.0}
                        className="w-16 bg-zinc-900 border border-zinc-800 text-blue-400 font-mono font-bold text-[10px] text-right rounded px-1 py-0.5 outline-none focus:border-blue-500/50 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
                      />
                    </div>
                    <ZernikeSlider
                      value={windSpeed}
                      onChange={handleWindSpeedChange}
                      min={0.0}
                      max={1.0}
                      step={0.05}
                    />
                  </div>
                </div>
              ) : (
                <div className="border-t border-[#223B53] pt-4 text-xs text-[#8FA0B3] font-mono italic animate-in fade-in duration-200">
                  Ajustes manuales deshabilitados para este método.
                </div>
              )}

              {/* Especificaciones del SLM */}
              <div className="mt-5 pt-4 border-t border-[#223B53] text-xs text-[#8FA0B3] space-y-1.5 font-mono">
                <div className="text-zinc-500 font-sans font-medium text-[11px] uppercase tracking-wider mb-1">
                  Especificaciones de Hardware
                </div>
                <p>Dispositivo: Holoeye Pluto 2.1</p>
                <p>Resolución: 1920 × 1080 px</p>
                <p>Paso de píxel: 8.0 µm</p>
                <p>Longitud de onda: 1550 nm (Telecom-C)</p>
                <p>Niveles de fase: 256 (8-bit)</p>
              </div>

              {/* Diagnóstico */}
              <div className="mt-4 pt-4 border-t border-[#223B53] text-xs text-[#8FA0B3] space-y-1 font-mono">
                <div className="text-zinc-500 font-sans font-medium text-[11px] uppercase tracking-wider mb-1">
                  Diagnóstico de Enlaces
                </div>
                <p>Simulador: <span className={simOnline ? 'text-emerald-400 font-semibold' : 'text-rose-400'}>{simOnline ? 'ONLINE' : 'OFFLINE'}</span></p>
                {method === '2' && (
                  <>
                    <p>FPS Simulación: <span className="text-blue-400 font-semibold">{fps} FPS</span></p>
                    <p>Precisión CNN (2s): <span className="text-violet-400 font-semibold">{avgAccuracy.toFixed(2)}%</span></p>
                  </>
                )}
                <p>Inferencia: READY</p>
              </div>
            </div>
          ) : (
            <div className="lab-panel p-5 flex flex-col animate-in fade-in duration-200">
              <h2 className="flex items-center gap-2 text-sm font-medium text-white mb-4">
                <Settings size={16} className="text-zinc-400" /> Configuración de Cámara
              </h2>

              {/* Selectbox para el Modelo de Red Neuronal (CNN) */}
              <div className="flex flex-col gap-2 mb-4">
                <label className="text-xs font-semibold text-[#8FA0B3] uppercase tracking-wider">
                  Modelo de Red Neuronal (CNN)
                </label>
                <select
                  value={activeModel}
                  onChange={(e) => handleModelChange(e.target.value)}
                  className="w-full bg-[#101B2E] border border-[#223B53] text-[#E8EEF7] text-xs rounded p-2.5 outline-none focus:border-[#2A3E53] font-sans"
                >
                  <option value="phase_diversity">Modelo A (Phase Diversity - 2 Ch)</option>
                  <option value="resnet10">Modelo ResNet-10 (Phase Diversity - 2 Ch)</option>
                  <option value="resnet18">Modelo ResNet-18 (Phase Diversity - 2 Ch)</option>
                </select>
              </div>

              {/* Parámetros de Control Físico de la Cámara */}
              <div className="border-t border-[#223B53] pt-4 flex flex-col gap-3">
                <div className="text-[10px] font-bold text-[#8FA0B3] uppercase tracking-wider mb-1">
                  Control Físico de la Cámara
                </div>
                <div className="flex flex-col gap-1.5">
                  <div className="flex justify-between text-[10px] text-[#8FA0B3] font-mono">
                    <span>Exposición (Shutter)</span>
                    <span className="text-blue-400">{cameraShutter.toFixed(1)} ms</span>
                  </div>
                  <ZernikeSlider
                    value={cameraShutter}
                    onChange={handleShutterChange}
                    min={1.0}
                    max={66.0}
                    step={0.5}
                  />
                </div>
                <div className="flex flex-col gap-1.5">
                  <div className="flex justify-between text-[10px] text-[#8FA0B3] font-mono">
                    <span>Ganancia (Gain)</span>
                    <span className="text-blue-400">{cameraGain.toFixed(1)} dB</span>
                  </div>
                  <ZernikeSlider
                    value={cameraGain}
                    onChange={handleGainChange}
                    min={0.0}
                    max={24.0}
                    step={0.1}
                  />
                </div>
                <div className="flex flex-col gap-1.5">
                  <div className="flex justify-between text-[10px] text-[#8FA0B3] font-mono">
                    <span>Zoom de Tracking (ROI)</span>
                    <span className="text-blue-400">{cameraRoiSize} px</span>
                  </div>
                  <ZernikeSlider
                    value={cameraRoiSize}
                    onChange={handleRoiSizeChange}
                    min={48}
                    max={200}
                    step={2}
                  />
                </div>
              </div>

              {/* Parámetros del Controlador Kalman / LQG */}
              <div className="border-t border-[#223B53] pt-4 flex flex-col gap-3">
                <div className="text-[10px] font-bold text-[#8FA0B3] uppercase tracking-wider mb-1">
                  Parámetros Kalman / LQG
                </div>
                <div className="flex flex-col gap-1.5">
                  <div className="flex justify-between text-[10px] text-[#8FA0B3] font-mono">
                    <span>Ruido Proceso (Q)</span>
                    <span className="text-blue-400">{kalmanQ}</span>
                  </div>
                  <ZernikeSlider
                    value={kalmanQ}
                    onChange={handleKalmanQChange}
                    min={0.0001}
                    max={0.1}
                    step={0.0001}
                  />
                </div>
                <div className="flex flex-col gap-1.5">
                  <div className="flex justify-between text-[10px] text-[#8FA0B3] font-mono">
                    <span>Ruido Medida (R)</span>
                    <span className="text-blue-400">{kalmanR}</span>
                  </div>
                  <ZernikeSlider
                    value={kalmanR}
                    onChange={handleKalmanRChange}
                    min={0.0001}
                    max={0.2}
                    step={0.0005}
                  />
                </div>
                <div className="flex flex-col gap-1.5">
                  <div className="flex justify-between text-[10px] text-[#8FA0B3] font-mono">
                    <span>Latencia (Delay)</span>
                    <span className="text-blue-400">{kalmanDelay} frames</span>
                  </div>
                  <ZernikeSlider
                    value={kalmanDelay}
                    onChange={handleKalmanDelayChange}
                    min={1}
                    max={3}
                    step={1}
                  />
                </div>
              </div>

              {/* Especificaciones de Hardware Real */}
              <div className="mt-5 pt-4 border-t border-[#223B53] text-xs text-[#8FA0B3] space-y-1.5 font-mono">
                <div className="text-[#8FA0B3] font-sans font-medium text-[11px] uppercase tracking-wider mb-1">
                  Especificaciones de Hardware
                </div>
                <p>Cámara: Point Grey Chameleon</p>
                <p>Sensor: CMLN-13S2M</p>
                <p>Resolución: 1280 × 960 px</p>
                <p>Formato Color: Mono8 (8-bit)</p>
                <p>Tasa Refresco: ~15 fps (Síncrona)</p>
                <p>Conexión: USB 2.0</p>
              </div>

              {/* Diagnóstico de Cámara */}
              <div className="mt-4 pt-4 border-t border-zinc-800 text-xs text-zinc-400 space-y-1 font-mono">
                <div className="text-zinc-500 font-sans font-medium text-[11px] uppercase tracking-wider mb-1">
                  Diagnóstico de Enlace
                </div>
                <p>Daemon Cámara: <span className={cameraActive ? 'text-emerald-400 font-semibold' : 'text-rose-400'}>{cameraActive ? 'ONLINE' : 'OFFLINE'}</span></p>
                <p>FPS Captura: <span className="text-blue-400 font-semibold">{cameraFps.toFixed(1)} FPS</span></p>
                <p>Centroide Spot: <span className="text-violet-400 font-semibold">{cameraCentroid[0]}, {cameraCentroid[1]}</span></p>
                <p>Incertidumbre Kalman: <span className="text-blue-400 font-semibold">{kalmanUncertainty.toFixed(4)}</span></p>
              </div>
            </div>
          )}

          {/* Botón de calibración eliminado por solicitud del usuario */}
        </aside>

        {/* ── PANELES VISUALES ── */}
        <section className="flex-1 flex flex-col gap-6 w-full">
          {activeTab === 'simulation' ? (
            method === '1' || method === '2' ? (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 w-full animate-in fade-in duration-300">

                {/* COLUMNA IZQUIERDA: SIMULADOR (CORRECCIÓN PERFECTA Y HÍBRIDA CNN+KALMAN) */}
                <div className="flex flex-col gap-6">
                  {/* 1. Mapa de Fase SLM (Corrección Perfecta) */}
                  <div className="lab-panel p-4 flex flex-col h-[380px]">
                    <div className="mb-3">
                      <span className="text-xs font-semibold text-white uppercase tracking-wider">MAPA DE FASE SLM (CORRECCIÓN PERFECTA)</span>
                      <p className="text-xs text-zinc-400 mt-0.5">
                        {method === '1' ? 'Fase correctora conjugada · bytes raw → Canvas GPU' : 'Fase correctora dinámica · bytes raw → Canvas GPU'}
                      </p>
                    </div>
                    <div className="flex-1 bg-black rounded border border-zinc-800 overflow-hidden flex items-center justify-center relative min-h-0">
                      <canvas
                        ref={canvasRef}
                        width={640}
                        height={360}
                        className="w-full h-full object-contain"
                        style={{ imageRendering: 'pixelated' }}
                      />
                      <div className="absolute top-2 left-2 font-mono text-[9px] text-zinc-400 bg-zinc-900/80 px-2 py-0.5 rounded border border-zinc-800">
                        SLM_CORRECTION_MAP (Prysm·GPU)
                      </div>
                      <div className="absolute top-2 right-2 flex gap-2">
                        {method === '2' && (
                          <span className="font-mono text-[9px] text-emerald-400 bg-zinc-900/80 px-2 py-0.5 rounded border border-zinc-800 animate-pulse">
                            {fpsSLM} FPS
                          </span>
                        )}
                        <span className="font-mono text-[9px] text-blue-400 bg-zinc-900/80 px-2 py-0.5 rounded border border-zinc-800">
                          GPU
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* 2. PSF Reconstruida (CNN + Kalman/LQG) */}
                  <div className="lab-panel p-4 flex flex-col h-[380px]">
                    <div className="mb-3 flex justify-between items-start">
                      <div>
                        <span className="text-xs font-semibold text-white uppercase tracking-wider">PSF RECONSTRUIDA (CNN + PREDICTOR LQG)</span>
                      </div>
                      <div className="flex gap-4 text-right shrink-0">
                        <div>
                          <span className="text-[9px] font-mono text-zinc-500 block uppercase">Precisión LQG t+1 (2s)</span>
                          <span className="text-xs font-mono font-bold text-emerald-400">{avgKalmanAccuracy.toFixed(2)}%</span>
                        </div>
                        <div>
                          <span className="text-[9px] font-mono text-zinc-500 block uppercase">vs CNN pura</span>
                          <span className={`text-xs font-mono font-bold ${(avgKalmanAccuracy - avgAccuracy) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                            {(avgKalmanAccuracy - avgAccuracy) >= 0 ? '+' : ''}{(avgKalmanAccuracy - avgAccuracy).toFixed(2)}%
                          </span>
                        </div>
                        <div>
                          <span className="text-[9px] font-mono text-zinc-500 block uppercase">Varianza P</span>
                          <span className="text-xs font-mono font-bold text-blue-400">{kalmanUncertainty.toFixed(4)}</span>
                        </div>
                      </div>
                    </div>
                    <div className="flex-1 bg-black rounded border border-zinc-800 overflow-hidden flex items-center justify-center relative min-h-0">
                      <canvas
                        ref={kalmanPsfCanvasRef}
                        width={640}
                        height={360}
                        className="w-full h-full object-contain"
                        style={{ imageRendering: 'pixelated' }}
                      />
                      <div className="absolute top-2 left-2 font-mono text-[9px] text-zinc-400 bg-zinc-900/80 px-2 py-0.5 rounded border border-zinc-800">
                        RECON_PSF_LQG_PREDICT (CNN+Kalman·GPU)
                      </div>
                      <div className="absolute top-2 right-2 flex gap-2">
                        {method === '2' && (
                          <span className="font-mono text-[9px] text-emerald-400 bg-zinc-900/80 px-2 py-0.5 rounded border border-zinc-800 animate-pulse">
                            {fpsKalmanPsf} FPS
                          </span>
                        )}
                        <span className="font-mono text-[9px] text-blue-400 bg-zinc-900/80 px-2 py-0.5 rounded border border-zinc-800">
                          GPU
                        </span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* COLUMNA DERECHA: RECONSTRUCCIÓN IA (CNN Y PSF RECONSTRUIDA PURA) */}
                <div className="flex flex-col gap-6">

                  {/* 3. Mapa de Fase SLM (Corrección CNN) */}
                  <div className="lab-panel p-4 flex flex-col h-[380px]">
                    <div className="mb-3 flex justify-between items-center">
                      <div>
                        <span className="text-xs font-semibold text-white uppercase tracking-wider">MAPA DE FASE SLM (CORRECCIÓN CNN)</span>
                        <p className="text-xs text-zinc-400 mt-0.5">
                          Fase correctora estimada por la Red Neuronal Convolucional
                        </p>
                      </div>
                      <div className="text-right">
                        <span className="text-[10px] font-mono text-zinc-400">PRECISIÓN CNN (2s)</span>
                        <p className="text-xs font-mono font-bold text-violet-400">{avgAccuracy.toFixed(2)}%</p>
                      </div>
                    </div>
                    <div className="flex-1 bg-black rounded border border-zinc-800 overflow-hidden flex items-center justify-center relative min-h-0">
                      <canvas
                        ref={cnnPhaseCanvasRef}
                        width={640}
                        height={360}
                        className={`w-full h-full object-contain ${isModelSwitching ? 'opacity-25' : ''}`}
                        style={{ imageRendering: 'pixelated' }}
                      />
                      {isModelSwitching && (
                        <div className="absolute inset-0 bg-black/60 backdrop-blur-xs flex flex-col items-center justify-center gap-3 animate-in fade-in duration-200">
                          <div className="w-8 h-8 border-2 border-violet-500/20 border-t-violet-400 rounded-full animate-spin" />
                          <span className="text-[10px] font-mono text-zinc-400 uppercase tracking-widest animate-pulse">Estabilizando CPU / Cargando pesos...</span>
                        </div>
                      )}
                      <div className="absolute top-2 left-2 font-mono text-[9px] text-zinc-400 bg-zinc-900/80 px-2 py-0.5 rounded border border-zinc-800">
                        SLM_CNN_PHASE_MAP (CNN·GPU)
                      </div>
                      <div className="absolute top-2 right-2 flex gap-2">
                        {method === '2' && (
                          <span className="font-mono text-[9px] text-emerald-400 bg-zinc-900/80 px-2 py-0.5 rounded border border-zinc-800 animate-pulse">
                            {fpsCnnPhase} FPS
                          </span>
                        )}
                        <span className="font-mono text-[9px] text-violet-400 bg-zinc-900/80 px-2 py-0.5 rounded border border-zinc-800">
                          GPU
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* 4. PSF RECONSTRUIDA PURA */}
                  <div className="lab-panel p-4 flex flex-col h-[380px]">
                    <div className="mb-3">
                      <span className="text-xs font-semibold text-white uppercase tracking-wider">PSF RECONSTRUIDA (CNN PURA)</span>
                      <p className="text-xs text-zinc-400 mt-0.5">
                        Imagen del plano focal corregido tras aplicar fase de la CNN sin control temporal
                      </p>
                    </div>
                    <div className="flex-1 bg-black rounded border border-zinc-800 overflow-hidden flex items-center justify-center relative min-h-0">
                      <canvas
                        ref={reconPsfCanvasRef}
                        width={640}
                        height={360}
                        className={`w-full h-full object-contain ${isModelSwitching ? 'opacity-25' : ''}`}
                        style={{ imageRendering: 'pixelated' }}
                      />
                      {isModelSwitching && (
                        <div className="absolute inset-0 bg-black/60 backdrop-blur-xs flex flex-col items-center justify-center gap-3 animate-in fade-in duration-200">
                          <div className="w-8 h-8 border-2 border-violet-500/20 border-t-violet-400 rounded-full animate-spin" />
                          <span className="text-[10px] font-mono text-zinc-400 uppercase tracking-widest animate-pulse">Amortiguando lazo cerrado...</span>
                        </div>
                      )}
                      <div className="absolute top-2 left-2 font-mono text-[9px] text-zinc-400 bg-zinc-900/80 px-2 py-0.5 rounded border border-zinc-800">
                        RECONSTRUCTED_PSF (Pure-CNN·GPU)
                      </div>
                      <div className="absolute top-2 right-2 flex gap-2">
                        {method === '2' && (
                          <span className="font-mono text-[9px] text-emerald-400 bg-zinc-900/80 px-2 py-0.5 rounded border border-zinc-800 animate-pulse">
                            {fpsReconPsf} FPS
                          </span>
                        )}
                        <span className="font-mono text-[9px] text-violet-400 bg-zinc-900/80 px-2 py-0.5 rounded border border-zinc-800">
                          GPU
                        </span>
                      </div>
                    </div>
                  </div>

                </div>

              </div>
            ) : (
              <div className="w-full flex items-center justify-center min-h-[380px] lab-panel p-12 animate-in fade-in duration-300">
                <div className="text-center max-w-md">
                  <div className="text-zinc-500 text-xs font-mono uppercase tracking-widest mb-3">
                    Aviso de Desarrollo
                  </div>
                  <h3 className="text-white text-base font-semibold tracking-wide mb-2">
                    Método en desarrollo para la siguiente etapa de la tesis
                  </h3>
                  <p className="text-zinc-400 text-xs leading-relaxed">
                    Las simulaciones estocásticas, de Kolmogorov y dinámicas evolutivas están contempladas en el plan de trabajo metodológico para fases posteriores.
                  </p>
                </div>
              </div>
            )
          ) : (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 w-full animate-in fade-in duration-300">

              {/* COLUMNA IZQUIERDA: CÁMARA RAW Y SPOT ROI */}
              <div className="flex flex-col gap-6">
                
                {/* 1. Feed completo de Cámara (1280x960, previsualizado en 320x240 en canvas) */}
                <div className="lab-panel p-4 flex flex-col h-[380px]">
                  <div className="mb-3">
                    <span className="text-xs font-semibold text-white uppercase tracking-wider">FEED COMPLETO DE CÁMARA (1280x960)</span>
                    <p className="text-xs text-zinc-400 mt-0.5">
                      Visualización del sensor completo con cruz de tracking de centroide (ROI 96x96)
                    </p>
                  </div>
                  <div className="flex-1 bg-black rounded border border-zinc-800 overflow-hidden flex items-center justify-center relative min-h-0">
                    <canvas
                      ref={cameraCanvasRef}
                      width={320}
                      height={240}
                      className="w-full h-full object-contain"
                      style={{ imageRendering: 'pixelated' }}
                    />
                    <div className="absolute top-2 left-2 font-mono text-[9px] text-zinc-400 bg-zinc-900/80 px-2 py-0.5 rounded border border-zinc-800">
                      POINT_GREY_CHAMELEON_STREAM
                    </div>
                    <div className="absolute top-2 right-2 flex gap-2">
                      <span className="font-mono text-[9px] text-emerald-400 bg-zinc-900/80 px-2 py-0.5 rounded border border-zinc-800 animate-pulse">
                        {cameraFps.toFixed(1)} FPS
                      </span>
                    </div>
                  </div>
                </div>

                {/* 2. Spot Diagram (ROI 96x96, extraído y colorizado) */}
                <div className="lab-panel p-4 flex flex-col h-[380px]">
                  <div className="mb-3">
                    <span className="text-xs font-semibold text-white uppercase tracking-wider">SPOT DIAGRAM DE ENTRADA (ROI 96x96)</span>
                    <p className="text-xs text-zinc-400 mt-0.5">
                      Región recortada del centroide, colorizada en espectro térmico para la CNN
                    </p>
                  </div>
                  <div className="flex-1 bg-black rounded border border-zinc-800 overflow-hidden flex items-center justify-center relative min-h-0">
                    <canvas
                      ref={canvasRef}
                      width={640}
                      height={360}
                      className="w-full h-full object-contain"
                      style={{ imageRendering: 'pixelated' }}
                    />
                    <div className="absolute top-2 left-2 font-mono text-[9px] text-zinc-400 bg-zinc-900/80 px-2 py-0.5 rounded border border-zinc-800">
                      PSF_INPUT_ROI (Mono8 Zoom)
                    </div>
                  </div>
                </div>
              </div>

              {/* COLUMNA DERECHA: CORRECCIONES DE LA IA Y FILTRADO */}
              <div className="flex flex-col gap-6">
                
                {/* 3. Mapa de Fase SLM (Corrección CNN) */}
                <div className="lab-panel p-4 flex flex-col h-[380px]">
                  <div className="mb-3 flex justify-between items-center">
                    <div>
                      <span className="text-xs font-semibold text-white uppercase tracking-wider">MAPA DE FASE SLM (CORRECCIÓN CNN)</span>
                      <p className="text-xs text-zinc-400 mt-0.5">
                        Fase correctora estimada por la Red Neuronal Convolucional
                      </p>
                    </div>
                    <div className="text-right">
                      <span className="text-[10px] font-mono text-zinc-400">PRECISIÓN CNN (2s)</span>
                      <p className="text-xs font-mono font-bold text-violet-400">{avgAccuracy.toFixed(2)}%</p>
                    </div>
                  </div>
                  <div className="flex-1 bg-black rounded border border-zinc-800 overflow-hidden flex items-center justify-center relative min-h-0">
                    <canvas
                      ref={cnnPhaseCanvasRef}
                      width={640}
                      height={360}
                      className={`w-full h-full object-contain ${isModelSwitching ? 'opacity-25' : ''}`}
                      style={{ imageRendering: 'pixelated' }}
                    />
                    {isModelSwitching && (
                      <div className="absolute inset-0 bg-black/60 backdrop-blur-xs flex flex-col items-center justify-center gap-3 animate-in fade-in duration-200">
                        <div className="w-8 h-8 border-2 border-violet-500/20 border-t-violet-400 rounded-full animate-spin" />
                        <span className="text-[10px] font-mono text-zinc-400 uppercase tracking-widest animate-pulse">Estabilizando CPU / Cargando pesos...</span>
                      </div>
                    )}
                    <div className="absolute top-2 left-2 font-mono text-[9px] text-zinc-400 bg-zinc-900/80 px-2 py-0.5 rounded border border-zinc-800">
                      SLM_CNN_PHASE_MAP
                    </div>
                  </div>
                </div>

                {/* 4. PSF Reconstruida (CNN + Predictor LQG) */}
                <div className="lab-panel p-4 flex flex-col h-[380px]">
                  <div className="mb-3 flex justify-between items-start">
                    <div>
                      <span className="text-xs font-semibold text-white uppercase tracking-wider">PSF CORREGIDA EN LAZO CERRADO (LQG)</span>
                    </div>
                    <div className="flex gap-4 text-right shrink-0">
                      <div>
                        <span className="text-[9px] font-mono text-zinc-500 block uppercase">Precisión LQG t+1 (2s)</span>
                        <span className="text-xs font-mono font-bold text-emerald-400">{avgKalmanAccuracy.toFixed(2)}%</span>
                      </div>
                      <div>
                        <span className="text-[9px] font-mono text-zinc-500 block uppercase">Varianza P</span>
                        <span className="text-xs font-mono font-bold text-blue-400">{kalmanUncertainty.toFixed(4)}</span>
                      </div>
                    </div>
                  </div>
                  <div className="flex-1 bg-black rounded border border-zinc-800 overflow-hidden flex items-center justify-center relative min-h-0">
                    <canvas
                      ref={kalmanPsfCanvasRef}
                      width={640}
                      height={360}
                      className="w-full h-full object-contain"
                      style={{ imageRendering: 'pixelated' }}
                    />
                    <div className="absolute top-2 left-2 font-mono text-[9px] text-zinc-400 bg-zinc-900/80 px-2 py-0.5 rounded border border-zinc-800">
                      RECON_PSF_LQG_PREDICT (Lazo Cerrado)
                    </div>
                  </div>
                </div>
              </div>

            </div>
          )}

          {/* Diagnóstico removido por solicitud */}
        </section>

      </main>
    </div>
  );
}

function ZernikeSlider({ value, onChange, min, max, step }) {
  return (
    <div className="relative h-6 flex items-center">
      <div
        className="absolute w-full h-1 rounded-full border border-zinc-800"
        style={{
          background: 'linear-gradient(to right, #000000, #3b82f6, #ffffff)',
        }}
      />
      <input
        type="range"
        min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="relative w-full h-1 appearance-none bg-transparent cursor-pointer zernike-slider"
      />
    </div>
  );
}

function StatusBadge({ online, label }) {
  return (
    <div className="flex items-center gap-2 text-xs font-mono">
      <div className={`w-2 h-2 rounded-full ${online ? 'bg-emerald-500 animate-pulse' : 'bg-rose-500'}`} />
      {label ? (
        <span className={online ? 'text-emerald-400' : 'text-rose-400 font-semibold'}>{label}</span>
      ) : null}
    </div>
  );
}

function VisualPanel({ title, subtitle, src, label, onLoad, fps }) {
  return (
    <div className="lab-panel p-4 flex flex-col h-[380px]">
      <div className="mb-3">
        <span className="text-xs font-semibold text-white uppercase tracking-wider">{title}</span>
        <p className="text-xs text-zinc-400 mt-0.5">{subtitle}</p>
      </div>
      <div className="flex-1 bg-black rounded border border-zinc-800 overflow-hidden flex items-center justify-center relative min-h-0">
        <img src={src} alt={title} className="w-full h-full object-contain" onLoad={onLoad} onError={onLoad} />
        <div className="absolute top-2 left-2 font-mono text-[9px] text-zinc-400 bg-zinc-900/80 px-2 py-0.5 rounded border border-zinc-800">
          {label}
        </div>
        {fps !== undefined && (
          <div className="absolute top-2 right-2 font-mono text-[9px] text-emerald-400 bg-zinc-900/80 px-2 py-0.5 rounded border border-zinc-800">
            {fps} FPS
          </div>
        )}
      </div>
    </div>
  );
}

export default App;

