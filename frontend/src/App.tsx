import React, { useRef, useState, useEffect } from "react";
import "./App.css";

//Graphs for analyze section
import graph1 from "./assets/images/resnet18_loss.png";
import graph4 from "./assets/images/resnet18_roc.png";
import graph2 from "./assets/images/mobilenetv3_loss.png";
import graph5 from "./assets/images/mobilenetv3_roc.png";
import graph3 from "./assets/images/custom_cnn_loss.png";
import graph6 from "./assets/images/custom_cnn_roc.png";

const App: React.FC = () => {
  const getStartedRef = useRef<HTMLDivElement>(null);
  const analysisRef = useRef<HTMLDivElement>(null);
  const infoRef = useRef<HTMLDivElement>(null);

  const [status, setStatus] = useState("Idle");

  const scrollToRef = (ref: React.RefObject<HTMLDivElement | null>) => {
    ref.current?.scrollIntoView({ behavior: "smooth" });
  };

  // Polling backend every 2 seconds
  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch("http://127.0.0.1:8000/status");
        const data = await res.json();
        setStatus(data.status === "running" ? "Running" : "Stopped");
      } catch {
        setStatus("Error");
      }
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  const handleStartCam = async () => {
    const select = document.getElementById("modelSelect") as HTMLSelectElement;
    const modelName = select.value;

    setStatus("Starting...");

    try {
      const response = await fetch("http://127.0.0.1:8000/start-cam", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_path: modelName,
          device_preference:"auto",
          camera_index:0
         }),
      })
      
      const data = await response.json();
      if (response.ok) setStatus("Running");
      else setStatus(`Error: ${data.detail}`);
    } catch {
      setStatus("Error starting camera");
    }
  };

  const handleStopCam = async () => {
  setStatus("Stopping...");
  try {
    const response = await fetch("http://127.0.0.1:8000/stop-cam", {
      method: "POST",
    });
    const data = await response.json();
    if (response.ok) setStatus("Stopped");
    else setStatus(`Error: ${data.detail}`);
  } catch {
    setStatus("Error stopping camera");
  }
};

  return (
    <div>
      <nav className="menu-bar">
        <div className="logo">NeuroSigny AI</div>
        <button onClick={() => scrollToRef(getStartedRef)}>Get Started</button>
        <button onClick={() => scrollToRef(analysisRef)}>Analysis</button>
        <button onClick={() => scrollToRef(infoRef)}>Information</button>
      </nav>

      <section ref={getStartedRef} className="section get-started-section">
        <div className="overlay">
          <h2>Get Started</h2>
          <div className="model-container">
            <div className="model-selection">
              <select id="modelSelect" className="model-dropdown">
                <option value="Resnet">Resnet Model</option>
                <option value="MobileNet">MobileNetV3 Model</option>
                <option value="CUSTOMCNN">Custom CNN Model</option>
              </select>
              <button className="set-model-button" onClick={handleStartCam}>
                Set and Start
              </button>
                <button className="stop-model-button" onClick={handleStopCam}>
                 Stop Camera
              </button>

            </div>
            <div className="status-area">
              <p>Status: {status}</p>
            </div>
          </div>
        </div>
      </section>

      <section ref={analysisRef} className="section analysis-section">
  <h2>Analysis</h2>
  <div className="analysis-images">
    <div className="image-card">
      <img src={graph1} alt="Graph 1" />

    </div>
    <div className="image-card">
      <img src={graph2} alt="Graph 2" />

    </div>
    <div className="image-card">
      <img src={graph3} alt="Graph 3" />

    </div>
    <div className="image-card">
      <img src={graph4} alt="Graph 4" />

    </div>
    <div className="image-card">
      <img src={graph5} alt="Graph 5" />

    </div>
    <div className="image-card">
      <img src={graph6} alt="Graph 6" />

    </div>
  </div>
</section>


      <section ref={infoRef} className="section info-section">
        <h2>Information</h2>
        <p>Short description text here.</p>
      </section>
    </div>
  );
};

export default App;
