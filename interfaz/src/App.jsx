import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Settings, Activity, Info, ArrowLeft, RefreshCw, ExternalLink, TrendingUp, BarChart2 } from 'lucide-react';
import logoUbb from './assets/v-escudo-color-gradiente.png';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:5000';
const CONTROLADOR_BASE = import.meta.env.VITE_CONTROLADOR_BASE || 'http://localhost:5002';

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


function AnalyticsDashboard() {
  const [history, setHistory] = useState([]);
  const [summary, setSummary] = useState({ rmse_cnn: 0.0, rmse_control: 0.0, improvement: 0.0 });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [csvRecording, setCsvRecording] = useState(false);
  const [csvStatus, setCsvStatus] = useState({ file_path: '', frames_written: 0, elapsed_s: 0 });
  const [csvLoading, setCsvLoading] = useState(false);

  const fetchCsvStatus = () => {
    axios.get(`${CONTROLADOR_BASE}/telemetry/csv/status`)
      .then(res => {
        setCsvRecording(res.data.recording || false);
        setCsvStatus({
          file_path: res.data.file_path || '',
          frames_written: res.data.frames_written || 0,
          elapsed_s: res.data.elapsed_s || 0,
        });
      })
      .catch(err => {
        console.error('Error fetching CSV status:', err);
      });
  };

  const startCsvRecording = () => {
    setCsvLoading(true);
    axios.post(`${CONTROLADOR_BASE}/telemetry/csv/start`)
      .then(res => {
        if (res.data.status === 'success') {
          setCsvRecording(true);
          fetchCsvStatus();
        } else {
          console.error('CSV start failed:', res.data.message);
        }
      })
      .catch(err => {
        console.error('Error starting CSV recording:', err);
      })
      .finally(() => setCsvLoading(false));
  };

  const stopCsvRecording = () => {
    setCsvLoading(true);
    axios.post(`${CONTROLADOR_BASE}/telemetry/csv/stop`)
      .then(res => {
        if (res.data.status === 'success') {
          setCsvRecording(false);
          fetchCsvStatus();
        } else {
          console.error('CSV stop failed:', res.data.message);
        }
      })
      .catch(err => {
        console.error('Error stopping CSV recording:', err);
      })
      .finally(() => setCsvLoading(false));
  };

  const fetchStats = () => {
    axios.get(`${CONTROLADOR_BASE}/telemetry/stats`)
      .then(res => {
        if (res.data.status === 'success') {
          setHistory(res.data.history || []);
          setSummary(res.data.summary || { rmse_cnn: 0.0, rmse_control: 0.0, improvement: 0.0 });
          setError(null);
        }
      })
      .catch(err => {
        console.error('Error fetching telemetry:', err);
        setError('Error al conectar con la base de datos de telemetría.');
      })
      .finally(() => {
        setLoading(false);
      });
  };

  useEffect(() => {
    // Iniciar sesión de telemetría activa en el controlador
    axios.post(`${CONTROLADOR_BASE}/telemetry/session/start`)
      .then(() => {
        fetchStats();
        fetchCsvStatus();
      })
      .catch(err => console.error("Error al iniciar sesión de telemetría:", err));

    const interval = setInterval(() => {
      fetchStats();
      fetchCsvStatus();
    }, 1000);

    return () => {
      clearInterval(interval);
      // Apagar telemetría en segundo plano al salir
      axios.post(`${CONTROLADOR_BASE}/telemetry/session/stop`)
        .catch(err => console.error("Error al detener sesión de telemetría:", err));
    };
  }, []);

  // Obtener el último frame de telemetría para mostrar en vivo
  const lastFrame = history[history.length - 1] || {};

  // Preparar datos de comparación en vivo Real vs CNN vs Control para Z2-Z11
  const comparisonData = ZERNIKE_MODES.slice(1).map(mode => {
    const key = mode.id.toLowerCase();
    return {
      id: mode.id,
      name: mode.name.split(' — ')[1] || mode.id,
      real: lastFrame[key] || 0.0,
      cnn: lastFrame[`${key}_cnn`] || 0.0,
      control: lastFrame[`${key}_control`] || 0.0,
    };
  });

  const maxValModos = Math.max(
    ...comparisonData.flatMap(d => [Math.abs(d.real), Math.abs(d.cnn), Math.abs(d.control)]),
    0.1
  );

  // Funciones para graficar en SVG
  const renderRmseChart = () => {
    if (history.length < 2) return <text x="50" y="50" fill="#71717a" className="text-xs font-mono">Esperando suficientes datos...</text>;
    const w = 500;
    const h = 180;
    const padding = 30;
    const plotW = w - padding * 2;
    const plotH = h - padding * 2;

    // Obtener los RMSE por punto para graficar
    const points = history.map((row, idx) => {
      let cnn_se = 0;
      let ctrl_se = 0;
      for (let i = 2; i <= 11; i++) {
        const real = row[`z${i}`] || 0.0;
        const cnn = row[`z${i}_cnn`] || 0.0;
        const ctrl = row[`z${i}_control`] || 0.0;
        cnn_se += (cnn - real)**2;
        ctrl_se += (ctrl - real)**2;
      }
      return {
        x: idx,
        cnn_rmse: Math.sqrt(cnn_se / 10),
        ctrl_rmse: Math.sqrt(ctrl_se / 10)
      };
    });

    const maxVal = Math.max(...points.flatMap(p => [p.cnn_rmse, p.ctrl_rmse]), 0.1) * 1.15;

    const getX = (idx) => padding + (idx / (points.length - 1)) * plotW;
    const getY = (val) => h - padding - (val / maxVal) * plotH;

    const cnnPath = points.map((p, idx) => `${idx === 0 ? 'M' : 'L'} ${getX(idx)} ${getY(p.cnn_rmse)}`).join(' ');
    const ctrlPath = points.map((p, idx) => `${idx === 0 ? 'M' : 'L'} ${getX(idx)} ${getY(p.ctrl_rmse)}`).join(' ');

    return (
      <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-full">
        {/* Ejes */}
        <line x1={padding} y1={h - padding} x2={w - padding} y2={h - padding} stroke="#223B53" strokeWidth="1" />
        <line x1={padding} y1={padding} x2={padding} y2={h - padding} stroke="#223B53" strokeWidth="1" />
        
        {/* Gridlines y etiquetas Y */}
        {[0, 0.25, 0.5, 0.75, 1].map((ratio, i) => {
          const val = ratio * maxVal;
          const y = getY(val);
          return (
            <g key={i}>
              <line x1={padding} y1={y} x2={w - padding} y2={y} stroke="#182A3A" strokeWidth="0.5" strokeDasharray="3,3" />
              <text x={padding - 6} y={y + 3} fill="#8FA0B3" className="text-[8px] font-mono text-right" textAnchor="end">{val.toFixed(2)}</text>
            </g>
          );
        })}

        {/* Polilíneas */}
        <path d={cnnPath} fill="none" stroke="#f43f5e" strokeWidth="1.5" />
        <path d={ctrlPath} fill="none" stroke="#3b82f6" strokeWidth="1.5" />

        {/* Leyenda en gráfico */}
        <text x={w - 110} y={padding + 10} fill="#f43f5e" className="text-[9px] font-semibold">● CNN Directa</text>
        <text x={w - 110} y={padding + 22} fill="#3b82f6" className="text-[9px] font-semibold">● Kalman + LQG</text>
      </svg>
    );
  };

  return (
    <div className="min-h-screen p-6 flex flex-col gap-6 font-sans bg-[#0B1426] text-[#E8EEF7]">
      {/* Header */}
      <header className="rounded-xl border border-[#223B53] bg-[#0F2433] px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <button
            onClick={() => window.close()}
            className="flex h-9 w-9 items-center justify-center rounded-md border border-[#223B53] bg-[#101B2E] text-zinc-400 hover:text-white transition-colors"
            title="Cerrar pestaña"
          >
            <ArrowLeft size={16} />
          </button>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-sm font-semibold tracking-[0.02em]">
                Panel Científico de Analíticas AO
              </h1>
              <span className="rounded border border-blue-500/30 bg-blue-500/10 px-2 py-0.5 text-[9px] uppercase tracking-wider text-blue-400">
                Lazo Cerrado
              </span>
            </div>
            <p className="text-[10px] text-[#8FA0B3]">Monitoreo de rendimiento cuantitativo en tiempo real</p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={fetchStats}
            className="flex h-9 w-9 items-center justify-center rounded-md border border-[#223B53] bg-[#101B2E] text-zinc-400 hover:text-white transition-colors"
          >
            <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
          </button>
          <button
            onClick={startCsvRecording}
            disabled={csvRecording || csvLoading}
            className="rounded-md border border-emerald-500 bg-emerald-600/10 px-3 py-2 text-[11px] text-emerald-300 hover:bg-emerald-600/20 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Grabar en CSV
          </button>
          <button
            onClick={stopCsvRecording}
            disabled={!csvRecording || csvLoading}
            className="rounded-md border border-rose-500 bg-rose-600/10 px-3 py-2 text-[11px] text-rose-300 hover:bg-rose-600/20 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Detener
          </button>
          <img src={logoUbb} alt="UBB" className="h-8 opacity-80" />
        </div>
        <div className="mt-2 flex flex-wrap gap-3 text-[10px] text-zinc-400">
          <span>{csvRecording ? 'Grabando telemetría a CSV' : 'No se está grabando CSV'}</span>
          <span>{csvStatus.frames_written} puntos</span>
          {csvStatus.file_path ? <span className="truncate max-w-[24rem]">Archivo: {csvStatus.file_path}</span> : null}
        </div>
      </header>

      {error ? (
        <div className="lab-panel p-6 text-center text-rose-400 font-mono text-xs">
          {error}
        </div>
      ) : loading ? (
        <div className="lab-panel p-12 text-center text-zinc-500 font-mono text-xs animate-pulse">
          Cargando telemetría histórica desde InfluxDB...
        </div>
      ) : (
        <div className="flex flex-col gap-6">
          {/* KPI Dashboard */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="lab-panel p-4 flex flex-col">
              <span className="text-[10px] font-bold text-zinc-400 uppercase tracking-wider">RMSE CNN Directo</span>
              <span className="text-2xl font-mono font-bold text-rose-400 mt-1">{summary.rmse_cnn.toFixed(4)}</span>
              <span className="text-[9px] text-zinc-500 mt-1 font-mono">Último minuto (con lag de 1 frame)</span>
            </div>
            <div className="lab-panel p-4 flex flex-col">
              <span className="text-[10px] font-bold text-zinc-400 uppercase tracking-wider">RMSE Kalman + LQG</span>
              <span className="text-2xl font-mono font-bold text-blue-400 mt-1">{summary.rmse_control.toFixed(4)}</span>
              <span className="text-[9px] text-zinc-500 mt-1 font-mono">Corrección anticipada para compensar lag</span>
            </div>
            <div className="lab-panel p-4 flex flex-col">
              <span className="text-[10px] font-bold text-zinc-400 uppercase tracking-wider">Mejora Neta del Filtro</span>
              <span className="text-2xl font-mono font-bold text-emerald-400 mt-1">
                {summary.improvement >= 0 ? `+${summary.improvement}%` : `${summary.improvement}%`}
              </span>
              <span className="text-[9px] text-zinc-500 mt-1 font-mono">Reducción del RMSE acumulado</span>
            </div>
            <div className="lab-panel p-4 flex flex-col">
              <span className="text-[10px] font-bold text-zinc-400 uppercase tracking-wider">Incertidumbre Promedio</span>
              <span className="text-2xl font-mono font-bold text-zinc-300 mt-1">
                {(history[history.length - 1]?.kalman_uncertainty || 0.0).toFixed(6)}
              </span>
              <span className="text-[9px] text-zinc-500 mt-1 font-mono">Traza de la matriz P del filtro</span>
            </div>
          </div>

          {/* Gráficos Principales */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Comparativa RMSE */}
            <div className="lab-panel p-5 flex flex-col">
              <div className="mb-4">
                <span className="text-xs font-semibold uppercase tracking-wider text-white">Comparativa de RMSE de Lazo Cerrado</span>
                <p className="text-[10px] text-zinc-400 mt-0.5">Evolución temporal del error residual (menor es mejor)</p>
              </div>
              <div className="h-56 bg-[#071327] rounded border border-zinc-800 p-2 flex items-center justify-center">
                {renderRmseChart()}
              </div>
            </div>

            {/* Comparación Z2-Z11 Real vs CNN vs Kalman (Barras Agrupadas) */}
            <div className="lab-panel p-5 flex flex-col">
              <div className="mb-2 flex justify-between items-center">
                <div>
                  <span className="text-xs font-semibold uppercase tracking-wider text-white">Modos Zernike en Tiempo Real</span>
                  <p className="text-[10px] text-zinc-400 mt-0.5">Comparativa instantánea de amplitud absoluta por modo</p>
                </div>
                <div className="flex gap-3 text-[9px] font-mono">
                  <span className="flex items-center gap-1"><span className="w-2.5 h-1.5 bg-[#e4e4e7] rounded-sm inline-block" /> Real</span>
                  <span className="flex items-center gap-1"><span className="w-2.5 h-1.5 bg-[#f43f5e] rounded-sm inline-block" /> CNN</span>
                  <span className="flex items-center gap-1"><span className="w-2.5 h-1.5 bg-[#3b82f6] rounded-sm inline-block" /> Kalman</span>
                </div>
              </div>
              
              <div className="h-56 bg-[#071327] rounded border border-zinc-800 p-4 flex items-end justify-between gap-2">
                {comparisonData.map((item, i) => {
                  return (
                    <div key={i} className="flex-1 flex flex-col items-center h-full justify-end relative group">
                      {/* Tooltip con valores precisos */}
                      <div className="absolute -top-3 left-1/2 -translate-x-1/2 bg-[#0A1628] border border-[#223B53] text-[7px] font-mono p-1 rounded shadow-lg pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity z-50 whitespace-nowrap">
                        R: {item.real.toFixed(3)}<br />
                        C: {item.cnn.toFixed(3)}<br />
                        K: {item.control.toFixed(3)}
                      </div>
                      {/* Grupo de 3 barras alineadas en la base */}
                      <div className="w-full flex items-end gap-0.5 h-[80%] pb-1 border-b border-zinc-800">
                        <div className="flex-1 bg-zinc-400/35 border-t border-zinc-400/50 rounded-t-sm transition-all duration-300"
                          style={{ height: `${Math.max(Math.min((Math.abs(item.real) / maxValModos) * 100, 100), 2)}%` }} />
                        <div className="flex-1 bg-rose-500/40 border-t border-rose-500/60 rounded-t-sm transition-all duration-300"
                          style={{ height: `${Math.max(Math.min((Math.abs(item.cnn) / maxValModos) * 100, 100), 2)}%` }} />
                        <div className="flex-1 bg-blue-500/45 border-t border-blue-500/75 rounded-t-sm transition-all duration-300"
                          style={{ height: `${Math.max(Math.min((Math.abs(item.control) / maxValModos) * 100, 100), 2)}%` }} />
                      </div>
                      <div className="text-[9px] font-mono font-bold text-zinc-400 mt-1">{item.id}</div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Registros de Telemetría Recientes — ancho completo */}
          <div className="lab-panel p-4 flex flex-col overflow-hidden">
            <div className="mb-3">
              <span className="text-xs font-semibold uppercase tracking-wider text-white">Registros de Telemetría Recientes (Z₂ - Z₁₁)</span>
            </div>
            <div className="flex-1 overflow-x-auto max-h-[144px] zernike-scroll">
              <table className="w-full text-left font-mono text-[9px] border-collapse min-w-[700px]">
                <thead>
                  <tr className="border-b border-[#223B53] text-[#8FA0B3]">
                    <th className="py-1.5 px-2 bg-[#0F2433] sticky left-0 z-10">Timestamp</th>
                    <th className="py-1.5 px-2 text-right">D/r₀</th>
                    {ZERNIKE_MODES.slice(1).map(m => (
                      <th key={m.id} className="py-1.5 px-2 text-right">{m.id}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {history.slice(-6).reverse().map((row, i) => (
                    <tr key={i} className="border-b border-zinc-800/40 hover:bg-zinc-900/30">
                      <td className="py-1.5 px-2 text-zinc-400 bg-[#0B1426] sticky left-0 font-semibold">
                        {row.time ? new Date(row.time).toLocaleTimeString() : 'N/A'}
                      </td>
                      <td className="py-1.5 px-2 text-right text-emerald-400">{(row.d_r0 || 0).toFixed(2)}</td>
                      {ZERNIKE_MODES.slice(1).map(m => {
                        const val = row[m.id.toLowerCase()] || 0.0;
                        return (
                          <td key={m.id} className="py-1.5 px-2 text-right text-zinc-300">
                            {val.toFixed(3)}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

        </div>
      )}
    </div>
  );
}

function App() {
  const [method, setMethod] = useState('1'); // '1', '2', '3', or '4'
  const [activeModel, setActiveModel] = useState('phase_diversity');
  const [activeTab, setActiveTab] = useState('simulation'); // 'simulation' | 'camera'
  
  // Ruteador por parametros de URL simple para pestaña de analiticas
  const urlParams = new URLSearchParams(window.location.search);
  const view = urlParams.get('view');
  if (view === 'analytics') {
    return <AnalyticsDashboard />;
  }

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
              <button
                onClick={() => setActiveTab('camera')}
                className={`flex items-center gap-2 px-5 py-2 rounded-lg text-sm font-semibold transition-colors duration-150 ${
                  activeTab === 'camera' ? 'bg-[#0F2433] text-[#E8EEF7] border border-[#2A3E53]' : 'text-[#8FA0B3] hover:text-[#E8EEF7]'
                }`}
              >
                Cámara Real
              </button>
            </div>
          </div>

          {/* Derecha: estado compacto + boton estadisticas + logo */}
          <div className="flex items-center gap-3">
            <button
              onClick={() => window.open(window.location.origin + '?view=analytics', '_blank')}
              className="flex items-center gap-2 px-5 py-2 rounded-lg text-sm font-semibold
                         border border-[#2A3E53] bg-[#101B2E] text-[#4EA3FF]
                         hover:bg-[#0F2433] hover:text-[#E8EEF7] hover:border-[#4EA3FF]
                         transition-colors duration-150 select-none"
            >
              <TrendingUp size={14} />
              Ver Estadísticas
              <ExternalLink size={11} className="opacity-50" />
            </button>
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
              {/* 1. Feed completo de Cámara */}
              <div className="lab-panel p-4 flex flex-col h-[500px]">
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

                {/* 2. Mapa de Fase SLM (Corrección CNN) */}
                <div className="lab-panel p-4 flex flex-col h-[500px]">
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
    <div className="flex items-center gap-1.5 text-[10px] font-mono select-none">
      <span className={online ? 'text-[#8FA0B3]' : 'text-rose-500 font-semibold'}>
        {online ? 'SIMULATOR ACTIVE' : 'SIMULATOR OFFLINE'}
      </span>
      {label && <span className="text-zinc-500 ml-1">· {label}</span>}
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

