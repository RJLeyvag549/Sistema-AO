import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Settings, Activity, Info } from 'lucide-react';
import logoUbb from './assets/v-escudo-color-gradiente.png';

const API_BASE = 'http://localhost:5000';

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
  const [avgAccuracy, setAvgAccuracy] = useState(100.0);
  const [isModelSwitching, setIsModelSwitching] = useState(false);

  // ── SISTEMA DE LOGS EN TIEMPO REAL ──────────────────────────────────────────
  const [debugLogs, setDebugLogs] = useState([]);
  const logBufferRef = useRef([]);       // buffer circular sin re-render por cada entrada
  const logFlushTimerRef = useRef(null); // timer para flush en lote cada 300ms

  const addLog = (level, msg) => {
    const ts = new Date().toISOString().slice(11, 23); // HH:MM:SS.mmm
    const entry = { ts, level, msg, id: Date.now() + Math.random() };
    logBufferRef.current = [entry, ...logBufferRef.current].slice(0, 80);
    // Flush en lote para no provocar 60 re-renders/seg
    if (!logFlushTimerRef.current) {
      logFlushTimerRef.current = setTimeout(() => {
        setDebugLogs([...logBufferRef.current]);
        logFlushTimerRef.current = null;
      }, 300);
    }
  };

  const debounceRef = useRef(null);
  const slmFramesCountRef = useRef(0);
  const turbPsfFramesCountRef = useRef(0);
  const cnnPhaseFramesCountRef = useRef(0);
  const reconPsfFramesCountRef = useRef(0);
  const accuracyListRef = useRef([]);

  const canvasRef = useRef(null);        // Canvas GPU: mapa de fase SLM (corrección perfecta)
  const psfCanvasRef = useRef(null);     // Canvas GPU: PSF con turbulencia
  const cnnPhaseCanvasRef = useRef(null); // Canvas GPU: mapa de fase estimado por CNN
  const reconPsfCanvasRef = useRef(null); // Canvas GPU: PSF reconstruida por CNN
  const drawRef = useRef(null);          // Función de dibujo del SLM (ref para recursividad)
  const drawPsfRef = useRef(null);       // Función de dibujo de la PSF con turbulencia
  const drawCnnPhaseRef = useRef(null);  // Función de dibujo del mapa de fase CNN
  const drawReconPsfRef = useRef(null);  // Función de dibujo de la PSF reconstruida
  const loopIdRef = useRef(null);        // ID de requestAnimationFrame del bucle SLM
  const loopPsfIdRef = useRef(null);     // ID de requestAnimationFrame del bucle PSF
  const loopCnnPhaseIdRef = useRef(null);
  const loopReconPsfIdRef = useRef(null);
  // Flags anti-acumulación: evitan lanzar un nuevo fetch si el anterior aún está en vuelo
  const cnnPhaseInFlightRef = useRef(false);
  const reconPsfInFlightRef = useRef(false);

  const methodRef = useRef(method);
  useEffect(() => {
    methodRef.current = method;
  }, [method]);

  const isModelSwitchingRef = useRef(isModelSwitching);
  useEffect(() => {
    isModelSwitchingRef.current = isModelSwitching;
  }, [isModelSwitching]);

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
          rgba[i * 4]     = gray[i];  // R
          rgba[i * 4 + 1] = gray[i];  // G
          rgba[i * 4 + 2] = gray[i];  // B
          rgba[i * 4 + 3] = 255;       // A
        }
        const ctx = canvasRef.current?.getContext('2d');
        if (ctx) ctx.putImageData(new ImageData(rgba, 640, 360), 0, 0);
        slmFramesCountRef.current += 1;
        if (methodRef.current === '2') setTimeout(() => {
          loopIdRef.current = requestAnimationFrame(drawRef.current);
        }, 40);
      })
      .catch(() => {
        if (methodRef.current === '2') setTimeout(() => {
          loopIdRef.current = requestAnimationFrame(drawRef.current);
        }, 40);
      });
  };

  // Canvas 2: PSF con turbulencia (plano focal con aberración) — RGB 3 bytes/px
  drawPsfRef.current = () => {
    fetch(`${API_BASE}/image/psf-raw`)
      .then(res => res.arrayBuffer())
      .then(buffer => {
        const rgb = new Uint8Array(buffer);           // 640×360×3 = 691 200 bytes
        const numPixels = rgb.length / 3;
        const rgba = new Uint8ClampedArray(numPixels * 4);
        for (let i = 0; i < numPixels; i++) {
          rgba[i * 4]     = rgb[i * 3];      // R
          rgba[i * 4 + 1] = rgb[i * 3 + 1]; // G
          rgba[i * 4 + 2] = rgb[i * 3 + 2]; // B
          rgba[i * 4 + 3] = 255;             // A
        }
        const ctx = psfCanvasRef.current?.getContext('2d');
        if (ctx) ctx.putImageData(new ImageData(rgba, 640, 360), 0, 0);
        turbPsfFramesCountRef.current += 1;
        if (methodRef.current === '2') setTimeout(() => {
          loopPsfIdRef.current = requestAnimationFrame(drawPsfRef.current);
        }, 40);
      })
      .catch(() => {
        if (methodRef.current === '2') setTimeout(() => {
          loopPsfIdRef.current = requestAnimationFrame(drawPsfRef.current);
        }, 40);
      });
  };

  // Canvas 3: Mapa de fase CNN (corrección estimada) — escala de grises 1 byte/px
  // Throttling adaptativo + flag inFlight para no acumular peticiones pendientes.
  drawCnnPhaseRef.current = () => {
    if (isModelSwitchingRef.current) {
      if (methodRef.current === '2') {
        loopCnnPhaseIdRef.current = requestAnimationFrame(drawCnnPhaseRef.current);
      }
      return;
    }
    if (cnnPhaseInFlightRef.current) {
      if (methodRef.current === '2') {
        loopCnnPhaseIdRef.current = requestAnimationFrame(drawCnnPhaseRef.current);
      }
      return;
    }
    cnnPhaseInFlightRef.current = true;
    fetch(`${API_BASE}/image/cnn-phase-raw`)
      .then(res => {
        if (!res.ok) {
          cnnPhaseInFlightRef.current = false;
          if (methodRef.current === '2') {
            loopCnnPhaseIdRef.current = requestAnimationFrame(drawCnnPhaseRef.current);
          }
          throw new Error(`CNN_PHASE HTTP ${res.status}`);
        }
        const acc = res.headers.get('X-CNN-Accuracy');
        if (acc) {
          if (methodRef.current === '2') {
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
          rgba[i * 4]     = gray[i];
          rgba[i * 4 + 1] = gray[i];
          rgba[i * 4 + 2] = gray[i];
          rgba[i * 4 + 3] = 255;
        }
        const ctx = cnnPhaseCanvasRef.current?.getContext('2d');
        if (ctx) ctx.putImageData(new ImageData(rgba, 640, 360), 0, 0);
        cnnPhaseFramesCountRef.current += 1;
        cnnPhaseInFlightRef.current = false;
        if (methodRef.current === '2') {
          loopCnnPhaseIdRef.current = requestAnimationFrame(drawCnnPhaseRef.current);
        }
      })
      .catch(() => {
        cnnPhaseInFlightRef.current = false;
        if (methodRef.current === '2') {
          loopCnnPhaseIdRef.current = requestAnimationFrame(drawCnnPhaseRef.current);
        }
      });
  };

  // Canvas 4: PSF reconstruida (corregida por CNN) — RGB 3 bytes/px
  // Throttling adaptativo + flag inFlight para no acumular peticiones pendientes.
  drawReconPsfRef.current = () => {
    if (isModelSwitchingRef.current) {
      if (methodRef.current === '2') {
        loopReconPsfIdRef.current = requestAnimationFrame(drawReconPsfRef.current);
      }
      return;
    }
    if (reconPsfInFlightRef.current) {
      if (methodRef.current === '2') {
        loopReconPsfIdRef.current = requestAnimationFrame(drawReconPsfRef.current);
      }
      return;
    }
    reconPsfInFlightRef.current = true;
    fetch(`${API_BASE}/image/reconstructed-psf-raw`)
      .then(res => {
        if (!res.ok) {
          reconPsfInFlightRef.current = false;
          if (methodRef.current === '2') {
            loopReconPsfIdRef.current = requestAnimationFrame(drawReconPsfRef.current);
          }
          throw new Error(`RECON_PSF HTTP ${res.status}`);
        }
        const acc = res.headers.get('X-CNN-Accuracy');
        if (acc) {
          if (methodRef.current === '2') {
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
          rgba[i * 4]     = rgb[i * 3];
          rgba[i * 4 + 1] = rgb[i * 3 + 1];
          rgba[i * 4 + 2] = rgb[i * 3 + 2];
          rgba[i * 4 + 3] = 255;
        }
        const ctx = reconPsfCanvasRef.current?.getContext('2d');
        if (ctx) ctx.putImageData(new ImageData(rgba, 640, 360), 0, 0);
        reconPsfFramesCountRef.current += 1;
        reconPsfInFlightRef.current = false;
        if (methodRef.current === '2') {
          loopReconPsfIdRef.current = requestAnimationFrame(drawReconPsfRef.current);
        }
      })
      .catch(() => {
        reconPsfInFlightRef.current = false;
        if (methodRef.current === '2') {
          loopReconPsfIdRef.current = requestAnimationFrame(drawReconPsfRef.current);
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
        } else {
          addLog('DEBUG', 'MODEL_SWITCH: método 2 → loops se reanudan solos');
        }
      }, 800);
    } catch (err) {
      addLog('ERROR', `MODEL_SWITCH POST FAIL: ${err.message}`);
      cnnPhaseInFlightRef.current = false;
      reconPsfInFlightRef.current = false;
      isModelSwitchingRef.current = false;
      setIsModelSwitching(false);
    }
  };


  // Manejar loops automáticos para el Método 2 (Estocástico)
  useEffect(() => {
    let fpsIntervalId = null;
    let accuracyIntervalId = null;
    let zernikeIntervalId = null;

    if (method === '2') {
      axios.post(`${API_BASE}/config`, { method: '2', d_r0, wind_speed: windSpeed })
        .catch(err => console.error("Error al iniciar modo estocástico:", err));

      slmFramesCountRef.current = 0;
      turbPsfFramesCountRef.current = 0;
      cnnPhaseFramesCountRef.current = 0;
      reconPsfFramesCountRef.current = 0;
      accuracyListRef.current = [];

      // Arrancar los cuatro bucles GPU: fetch raw → canvas (auto-regulado por requestAnimationFrame)
      loopIdRef.current = requestAnimationFrame(drawRef.current);
      loopPsfIdRef.current = requestAnimationFrame(drawPsfRef.current);
      loopCnnPhaseIdRef.current = requestAnimationFrame(drawCnnPhaseRef.current);
      loopReconPsfIdRef.current = requestAnimationFrame(drawReconPsfRef.current);

      // Contador de FPS visible en la UI para cada canal
      fpsIntervalId = setInterval(() => {
        setFpsSLM(slmFramesCountRef.current);
        setFpsTurbPsf(turbPsfFramesCountRef.current);
        setFpsCnnPhase(cnnPhaseFramesCountRef.current);
        setFpsReconPsf(reconPsfFramesCountRef.current);
        setFps(slmFramesCountRef.current);

        slmFramesCountRef.current = 0;
        turbPsfFramesCountRef.current = 0;
        cnnPhaseFramesCountRef.current = 0;
        reconPsfFramesCountRef.current = 0;
      }, 1000);

      // Promediar precisión de la CNN cada 2 segundos
      accuracyIntervalId = setInterval(() => {
        if (accuracyListRef.current.length > 0) {
          const sum = accuracyListRef.current.reduce((a, b) => a + b, 0);
          const avg = sum / accuracyListRef.current.length;
          setAvgAccuracy(avg);
          accuracyListRef.current = [];
        }
      }, 2000);

      // Actualizar sliders con coeficientes dinámicos del backend
      zernikeIntervalId = setInterval(() => {
        axios.get(`${API_BASE}/status`)
          .then((res) => {
            if (res.data && res.data.state && res.data.state.zernikes) {
              setZernikes(res.data.state.zernikes);
            }
          })
          .catch(err => console.error("Error al actualizar Zernikes:", err));
      }, 200);
    } else {
      // Detener los cuatro bucles GPU y volver a modo manual
      if (loopIdRef.current) cancelAnimationFrame(loopIdRef.current);
      if (loopPsfIdRef.current) cancelAnimationFrame(loopPsfIdRef.current);
      if (loopCnnPhaseIdRef.current) cancelAnimationFrame(loopCnnPhaseIdRef.current);
      if (loopReconPsfIdRef.current) cancelAnimationFrame(loopReconPsfIdRef.current);
      // Resetear flags inFlight para no quedar bloqueados al volver a modo 2
      cnnPhaseInFlightRef.current = false;
      reconPsfInFlightRef.current = false;
      
      setFpsSLM(0);
      setFpsTurbPsf(0);
      setFpsCnnPhase(0);
      setFpsReconPsf(0);

      axios.post(`${API_BASE}/config`, { method: '1' })
        .then(() => {
          if (drawRef.current) drawRef.current();
          if (drawPsfRef.current) drawPsfRef.current();
          if (drawCnnPhaseRef.current) drawCnnPhaseRef.current();
          if (drawReconPsfRef.current) drawReconPsfRef.current();
        })
        .catch(err => console.error("Error al volver a modo manual:", err));
    }

    return () => {
      if (loopIdRef.current) cancelAnimationFrame(loopIdRef.current);
      if (loopPsfIdRef.current) cancelAnimationFrame(loopPsfIdRef.current);
      if (loopCnnPhaseIdRef.current) cancelAnimationFrame(loopCnnPhaseIdRef.current);
      if (loopReconPsfIdRef.current) cancelAnimationFrame(loopReconPsfIdRef.current);
      if (fpsIntervalId) clearInterval(fpsIntervalId);
      if (accuracyIntervalId) clearInterval(accuracyIntervalId);
      if (zernikeIntervalId) clearInterval(zernikeIntervalId);
    };
  }, [method]);

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

      {/* ── HEADER ── */}
      <header className="flex justify-between items-center lab-panel px-6 py-4">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-full bg-blue-500/10 border border-blue-500/30 flex items-center justify-center">
            <Activity className="text-blue-500" size={16} />
          </div>
          <div>
            <h1 className="text-lg font-semibold tracking-tight text-white">
              Sistema de Control de Óptica Adaptativa
            </h1>
            <p className="text-xs text-zinc-400">
              HOLOEYE PLUTO 2.1 · 1550 nm · Universidad del Bío-Bío
            </p>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <StatusBadge online={simOnline} label="SIMULADOR FÍSICO PRYSM" />
          <img src={logoUbb} alt="UBB" className="h-9 opacity-90" />
        </div>
      </header>

      {/* ── CONTENIDO PRINCIPAL ── */}
      <main className="flex flex-col md:flex-row gap-6 items-start">

        {/* Barra lateral de controles */}
        <aside className="w-full md:w-80 flex flex-col gap-4 shrink-0">
          <div className="lab-panel p-5 flex flex-col">
            <h2 className="flex items-center gap-2 text-sm font-medium text-white mb-4">
              <Settings size={16} className="text-zinc-400" /> Configuración General
            </h2>

            {/* Selectbox para el Método de Generación */}
            <div className="flex flex-col gap-2 mb-4">
              <label className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">
                Método de Generación
              </label>
              <select
                value={method}
                onChange={(e) => setMethod(e.target.value)}
                className="w-full bg-zinc-900 border border-zinc-800 text-zinc-200 text-xs rounded p-2.5 outline-none focus:border-zinc-700 font-sans"
              >
                <option value="1">1. Modos Individuales Deterministas (Zernike)</option>
                <option value="2">2. Turbulencia Estocástica (Kolmogorov)</option>
              </select>
            </div>

            {/* Selectbox para el Modelo de Red Neuronal */}
            <div className="flex flex-col gap-2 mb-4">
              <label className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">
                Modelo de Red Neuronal (CNN)
              </label>
              <select
                value={activeModel}
                onChange={(e) => handleModelChange(e.target.value)}
                className="w-full bg-zinc-900 border border-zinc-800 text-zinc-200 text-xs rounded p-2.5 outline-none focus:border-zinc-700 font-sans"
              >
                <option value="phase_diversity">Modelo A (Phase Diversity - 2 Ch)</option>
                <option value="resnet10">Modelo ResNet-10 (Phase Diversity - 2 Ch)</option>
              </select>
            </div>



            {/* Sección Dinámica: 11 Polinomios de Zernike */}
            {method === '1' ? (
              <div className="border-t border-zinc-800 pt-4 flex flex-col gap-3 animate-in fade-in duration-200">
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
                    <div key={mode.id} className="border-b border-zinc-800/40 pb-3 last:border-0 last:pb-0">
                      {/* Nombre y valor actual */}
                      <div className="flex justify-between items-center mb-1.5">
                        <span className="text-[10px] font-mono text-zinc-300">
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
                          className="w-16 bg-zinc-900 border border-zinc-800 text-blue-400 font-mono font-bold text-[10px] text-right rounded px-1 py-0.5 outline-none focus:border-blue-500/50 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
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
              <div className="border-t border-zinc-800 pt-4 flex flex-col gap-4 animate-in fade-in duration-200">
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
              <div className="border-t border-zinc-800 pt-4 text-xs text-zinc-500 font-mono italic animate-in fade-in duration-200">
                Ajustes manuales deshabilitados para este método.
              </div>
            )}

            {/* Especificaciones del SLM */}
            <div className="mt-5 pt-4 border-t border-zinc-800 text-xs text-zinc-400 space-y-1.5 font-mono">
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
            <div className="mt-4 pt-4 border-t border-zinc-800 text-xs text-zinc-400 space-y-1 font-mono">
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

            {/* Panel de Logs en tiempo real */}
            <div className="mt-4 pt-4 border-t border-zinc-800">
              <div className="flex justify-between items-center mb-2">
                <span className="text-zinc-500 font-sans font-medium text-[11px] uppercase tracking-wider">
                  Logs CNN en Tiempo Real
                </span>
                <button
                  onClick={() => { logBufferRef.current = []; setDebugLogs([]); }}
                  className="text-[9px] text-zinc-600 hover:text-zinc-400 font-mono uppercase tracking-wider transition-colors"
                >
                  limpiar
                </button>
              </div>
              <div
                className="h-48 overflow-y-auto flex flex-col-reverse gap-0.5 bg-black/40 rounded border border-zinc-800 p-2"
                style={{ fontFamily: 'monospace' }}
              >
                {debugLogs.length === 0 ? (
                  <span className="text-zinc-700 text-[9px] italic">Sin eventos aún...</span>
                ) : (
                  debugLogs.map(entry => (
                    <div key={entry.id} className="flex gap-1.5 text-[9px] leading-relaxed">
                      <span className="text-zinc-600 shrink-0">{entry.ts}</span>
                      <span className={{
                        'INFO':  'text-emerald-400',
                        'WARN':  'text-amber-400',
                        'ERROR': 'text-rose-400',
                        'DEBUG': 'text-zinc-500',
                      }[entry.level] ?? 'text-zinc-400'}
                      >
                        [{entry.level}]
                      </span>
                      <span className="text-zinc-300 break-all">{entry.msg}</span>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>

          <button
            onClick={handleCalibrate}
            disabled={loading || method !== '1'}
            className={`w-full py-3.5 lab-button-primary uppercase tracking-wide text-xs ${(loading || method !== '1') ? 'opacity-50 cursor-wait' : ''}`}
          >
            {loading ? 'Calibrando Dispositivo...' : 'Calibrar Sistema'}
          </button>
        </aside>

        {/* ── PANELES VISUALES ── */}
        <section className="flex-1 flex flex-col gap-6 w-full">
          {method === '1' || method === '2' ? (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 w-full animate-in fade-in duration-300">

              {/* COLUMNA IZQUIERDA: SIMULADOR (CORRECCIÓN PERFECTA Y TURBULENCIA) */}
              <div className="flex flex-col gap-6">
                {/* 1. Mapa de Fase SLM (Corrección Perfecta) — Renderizado GPU via canvas (bytes raw de Prysm) */}
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

                {/* 2. PSF con Turbulencia (Plano Focal) — Renderizado GPU via canvas (bytes raw de Prysm) */}
                <div className="lab-panel p-4 flex flex-col h-[380px]">
                  <div className="mb-3">
                    <span className="text-xs font-semibold text-white uppercase tracking-wider">PSF CON TURBULENCIA</span>
                    <p className="text-xs text-zinc-400 mt-0.5">
                      Imagen del plano focal (Point Spread Function) con aberración
                    </p>
                  </div>
                  <div className="flex-1 bg-black rounded border border-zinc-800 overflow-hidden flex items-center justify-center relative min-h-0">
                    <canvas
                      ref={psfCanvasRef}
                      width={640}
                      height={360}
                      className="w-full h-full object-contain"
                      style={{ imageRendering: 'pixelated' }}
                    />
                    <div className="absolute top-2 left-2 font-mono text-[9px] text-zinc-400 bg-zinc-900/80 px-2 py-0.5 rounded border border-zinc-800">
                      FOCAL_PLANE_PSF (Prysm·GPU)
                    </div>
                    <div className="absolute top-2 right-2 flex gap-2">
                      {method === '2' && (
                        <span className="font-mono text-[9px] text-emerald-400 bg-zinc-900/80 px-2 py-0.5 rounded border border-zinc-800 animate-pulse">
                          {fpsTurbPsf} FPS
                        </span>
                      )}
                      <span className="font-mono text-[9px] text-blue-400 bg-zinc-900/80 px-2 py-0.5 rounded border border-zinc-800">
                        GPU
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              {/* COLUMNA DERECHA: RECONSTRUCCIÓN IA (CNN Y PSF RECONSTRUIDA) */}
              <div className="flex flex-col gap-6">

                {/* 3. Mapa de Fase SLM (Corrección CNN) — canvas raw idéntico al panel 1 */}
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

                {/* 4. PSF RECONSTRUIDA — canvas raw RGB idéntico al panel 1 pero a color */}
                <div className="lab-panel p-4 flex flex-col h-[380px]">
                  <div className="mb-3">
                    <span className="text-xs font-semibold text-white uppercase tracking-wider">PSF RECONSTRUIDA</span>
                    <p className="text-xs text-zinc-400 mt-0.5">
                      Imagen del plano focal corregido tras aplicar fase de la CNN
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
                      RECONSTRUCTED_PSF (Closed-Loop·GPU)
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
          )}

          {/* Barra de estado inferior */}
          <div className="lab-panel px-6 py-4 flex flex-col gap-4 text-xs font-mono text-zinc-400">
            <div className="text-[11px] text-zinc-500 uppercase font-sans font-semibold tracking-wider">
              Diagnóstico de Coeficientes Activos
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs">
              <div>
                <span className="text-zinc-500">Z₁ (Pistón):</span>
                <p className="text-zinc-300 font-semibold mt-0.5">{zernikes.Z1.toFixed(3)} rad</p>
              </div>
              <div>
                <span className="text-zinc-500">Z₂ (Tip X):</span>
                <p className="text-zinc-300 font-semibold mt-0.5">{zernikes.Z2.toFixed(3)} rad</p>
              </div>
              <div>
                <span className="text-zinc-500">Z₃ (Tilt Y):</span>
                <p className="text-zinc-300 font-semibold mt-0.5">{zernikes.Z3.toFixed(3)} rad</p>
              </div>
              <div>
                <span className="text-zinc-500">Z₄ (Defocus):</span>
                <p className="text-zinc-300 font-semibold mt-0.5">{zernikes.Z4.toFixed(3)} rad</p>
              </div>
            </div>
          </div>
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
      <span className={online ? 'text-emerald-400' : 'text-rose-400 font-semibold'}>{label}</span>
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

