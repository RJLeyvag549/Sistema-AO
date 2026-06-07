import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Settings, Activity, Info } from 'lucide-react';
import logoUbb from './assets/v-escudo-color-gradiente.png';

const API_BASE = 'http://localhost:5000';

const ZERNIKE_MODES = [
  { id: 'Z1', name: 'Z₁ — Pistón (Piston)', min: -Math.PI, max: Math.PI, step: 0.01 },
  { id: 'Z2', name: 'Z₂ — Tip (Tilt X)', min: -3.0, max: 3.0, step: 0.05 },
  { id: 'Z3', name: 'Z₃ — Tilt (Tilt Y)', min: -3.0, max: 3.0, step: 0.05 },
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
  const [psfImage] = useState(`${API_BASE}/image/psf?t=${Date.now()}`);
  const [loading, setLoading] = useState(false);
  const [simOnline, setSimOnline] = useState(false);
  const [fps, setFps] = useState(0);
  const debounceRef   = useRef(null);
  const loadedFramesRef = useRef(0);
  const canvasRef     = useRef(null);   // Canvas GPU para el mapa de fase SLM
  const drawRef       = useRef(null);   // Función de dibujo (ref para recursividad estable)
  const loopIdRef     = useRef(null);   // ID de requestAnimationFrame del bucle

  const methodRef = useRef(method);
  useEffect(() => {
    methodRef.current = method;
  }, [method]);

  // ── RENDER GPU: fetch bytes raw del simulador y pinta en canvas ──────────────
  // La física (Prysm) vive en el simulador. El browser solo recibe los píxeles.
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
        loadedFramesRef.current += 1;
        // Auto-loop solo en modo estocástico
        if (methodRef.current === '2') {
          loopIdRef.current = requestAnimationFrame(drawRef.current);
        }
      })
      .catch(() => {
        // Reintento suave en caso de error de red
        if (methodRef.current === '2') {
          loopIdRef.current = requestAnimationFrame(drawRef.current);
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
          if (res.data.state.d_r0) setDR0(res.data.state.d_r0);
          if (res.data.state.wind_speed) setWindSpeed(res.data.state.wind_speed);
          if (res.data.state.zernikes) setZernikes(res.data.state.zernikes);
        }
      })
      .catch(() => setSimOnline(false));
  }, []);

  // Manejar loops automáticos para el Método 2 (Estocástico)
  useEffect(() => {
    let fpsIntervalId    = null;
    let zernikeIntervalId = null;

    if (method === '2') {
      axios.post(`${API_BASE}/config`, { method: '2', d_r0, wind_speed: windSpeed })
        .catch(err => console.error("Error al iniciar modo estocástico:", err));

      loadedFramesRef.current = 0;

      // Arrancar el bucle GPU: fetch raw → canvas (auto-regulado por requestAnimationFrame)
      loopIdRef.current = requestAnimationFrame(drawRef.current);

      // Contador de FPS visible en la UI
      fpsIntervalId = setInterval(() => {
        setFps(loadedFramesRef.current);
        loadedFramesRef.current = 0;
      }, 1000);

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
      // Detener el bucle GPU y volver a modo manual
      if (loopIdRef.current) cancelAnimationFrame(loopIdRef.current);
      axios.post(`${API_BASE}/config`, { method: '1' })
        .catch(err => console.error("Error al volver a modo manual:", err));
    }

    return () => {
      if (loopIdRef.current) cancelAnimationFrame(loopIdRef.current);
      if (fpsIntervalId)    clearInterval(fpsIntervalId);
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
        // Modo manual: pide un frame al simulador y lo pinta en canvas
        drawRef.current();
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
                <p>FPS Simulación: <span className="text-blue-400 font-semibold">{fps} FPS</span></p>
              )}
              <p>Inferencia: READY</p>
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
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 w-full animate-in fade-in duration-300">

              {/* Mapa de Fase SLM — Renderizado GPU via canvas (bytes raw de Prysm) */}
              <div className="lab-panel p-4 flex flex-col h-[380px]">
                <div className="mb-3">
                  <span className="text-xs font-semibold text-white uppercase tracking-wider">MAPA DE FASE SLM</span>
                  <p className="text-xs text-zinc-400 mt-0.5">
                    {method === '1' ? 'Frente de onda · Prysm → bytes raw → Canvas GPU' : 'Turbulencia estocástica · Prysm → bytes raw → Canvas GPU'}
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
                    SLM_PHASE_MAP (Prysm·GPU)
                  </div>
                  <div className="absolute top-2 right-2 font-mono text-[9px] text-blue-400 bg-zinc-900/80 px-2 py-0.5 rounded border border-zinc-800">
                    ⚡ GPU
                  </div>
                </div>
              </div>

              {/* Frente de Onda Reconstruido (CNN) */}
              <VisualPanel
                title="FRENTE DE ONDA RECONSTRUIDO"
                subtitle="Estimación de fase predicha por la red neuronal (CNN)"
                src={psfImage}
                label="RECONSTRUCTED_WAVEFRONT (CNN)"
              />

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

function VisualPanel({ title, subtitle, src, label, onLoad }) {
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
      </div>
    </div>
  );
}

export default App;

