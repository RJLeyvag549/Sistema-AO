import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Settings, Wind, Droplets, Activity, Zap, Info, Lock, User, LogIn } from 'lucide-react';
import logoUbb from './assets/v-escudo-color-gradiente.png';

const API_BASE = 'http://localhost:5000';

function App() {
  const [params, setParams] = useState({
    turbulencia: 0.5,
    viento: 10,
    humedad: 50
  });
  const [images, setImages] = useState({
    distorted: `${API_BASE}/image/distorted?t=${Date.now()}`,
    reconstructed: `${API_BASE}/image/reconstructed?t=${Date.now()}`
  });
  const [loading, setLoading] = useState(false);

  const updateParams = async (newParams) => {
    setParams(newParams);
    try {
      await axios.post(`${API_BASE}/config`, newParams);
      setImages({
        distorted: `${API_BASE}/image/distorted?t=${Date.now()}`,
        reconstructed: `${API_BASE}/image/reconstructed?t=${Date.now()}`
      });
    } catch (error) {
      console.error("Error updating config:", error);
    }
  };

  const handleSliderChange = (e) => {
    const { name, value } = e.target;
    updateParams({ ...params, [name]: parseFloat(value) });
  };

  const handleCalibrate = async () => {
    setLoading(true);
    try {
      const response = await axios.post('http://localhost:5002/calibrate');
      console.log("Calibracion exitosa:", response.data);
      alert("Calibracion completada con exito. Revisa los logs de Docker.");
    } catch (error) {
      console.error("Error en calibracion:", error);
      alert("Error al conectar con el controlador.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen p-6 flex flex-col gap-6 animate-in fade-in duration-1000">
      {/* HEADER */}
      <header className="flex justify-between items-center glass-panel p-4 rounded-lg neon-border">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-ao-accent rounded-full flex items-center justify-center animate-pulse">
            <Activity className="text-ao-bg" size={24} />
          </div>
          <div>
            <h1 className="text-2xl font-orbitron font-bold text-ao-accent tracking-widest uppercase">Sistema AO</h1>
            <p className="text-[10px] font-mono text-ao-accent/60">ADAPTIVE OPTICS CONTROL UNIT // UBB CHILE</p>
          </div>
        </div>
        <img src={logoUbb} alt="UBB" className="h-12 opacity-80" />
      </header>

      <main className="flex flex-1 gap-6">
        <aside className="w-80 flex flex-col gap-4">
          <div className="glass-panel p-6 rounded-lg flex-1">
            <h2 className="flex items-center gap-2 font-orbitron text-lg mb-6 text-ao-accent">
              <Settings size={20} /> PARAMETROS
            </h2>
            <div className="space-y-8">
              <ControlSlider label="Turbulencia" icon={<Activity size={18}/>} name="turbulencia" value={params.turbulencia} min={0} max={1} step={0.01} onChange={handleSliderChange} />
              <ControlSlider label="Viento (m/s)" icon={<Wind size={18}/>} name="viento" value={params.viento} min={0} max={50} step={1} onChange={handleSliderChange} />
              <ControlSlider label="Humedad (%)" icon={<Droplets size={18}/>} name="humedad" value={params.humedad} min={0} max={100} step={1} onChange={handleSliderChange} />
            </div>
            <div className="mt-12 p-4 bg-ao-bg/50 border border-ao-accent/20 rounded font-mono text-xs text-ao-accent/80">
              <p className="flex items-center gap-2 mb-2"><Info size={14}/> DIAGNOSTICO</p>
              <ul className="space-y-1">
                <li>&gt; Simulador: ONLINE</li>
                <li>&gt; Inferencia: READY</li>
              </ul>
            </div>
          </div>
          <button 
            onClick={handleCalibrate}
            disabled={loading}
            className={`bg-ao-plasma text-white py-4 font-orbitron font-bold rounded tracking-widest cyber-button neon-border ${loading ? 'opacity-50 cursor-wait' : ''}`}>
            {loading ? 'CALIBRANDO...' : 'CALIBRAR SISTEMA'}
          </button>
        </aside>

        <section className="flex-1 flex flex-col gap-6">
          <div className="grid grid-cols-2 gap-6 flex-1">
            <VisualPanel title="FRENTE DISTORSIONADO" src={images.distorted} label="RAW_STREAM" color="ao-accent" />
            <VisualPanel title="IMAGEN RECONSTRUIDA" src={images.reconstructed} label="AO_CORRECTED" color="ao-plasma" glow />
          </div>
          <div className="glass-panel h-32 p-4 rounded-lg flex gap-8 items-center font-mono text-xs">
            <div className="flex-1">
              <p className="text-ao-accent mb-2 tracking-tighter">RED NEURONAL CONVOLUCIONAL</p>
              <div className="w-full bg-ao-bg h-2 rounded-full overflow-hidden">
                <div className="bg-ao-accent h-full w-[85%] animate-pulse"></div>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-x-4">
              <span className="text-gray-500">Z0 (Piston):</span> <span className="text-ao-accent">0.0042</span>
              <span className="text-gray-500">Z3 (Defocus):</span> <span className="text-ao-accent">0.3421</span>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

function ControlSlider({ label, icon, name, value, min, max, step, onChange }) {
  return (
    <div className="flex flex-col gap-3">
      <div className="flex justify-between items-center text-xs font-mono">
        <span className="flex items-center gap-2 text-gray-400">{icon} {label.toUpperCase()}</span>
        <span className="text-ao-accent">{value}</span>
      </div>
      <input type="range" name={name} min={min} max={max} step={step} value={value} onChange={onChange} className="w-full h-1 bg-ao-panel rounded-lg appearance-none cursor-pointer accent-ao-accent" />
    </div>
  );
}

function VisualPanel({ title, src, label, color, glow }) {
  return (
    <div className={`glass-panel rounded-lg p-4 flex flex-col ${glow ? 'border-ao-accent/30' : ''}`}>
      <div className="flex justify-between items-center mb-4">
        <span className={`font-orbitron text-[10px] text-${color}`}>{title}</span>
        <div className={`w-1.5 h-1.5 bg-${glow ? 'ao-accent' : 'red-500'} rounded-full animate-pulse`}></div>
      </div>
      <div className={`flex-1 bg-black rounded border border-white/5 overflow-hidden flex items-center justify-center relative`}>
        <img src={src} alt={title} className="max-w-full max-h-full object-contain" />
        <div className="absolute top-2 left-2 font-mono text-[9px] text-white/30 bg-black/50 p-1 tracking-widest">{label}</div>
      </div>
    </div>
  );
}

export default App;
