import "maplibre-gl/dist/maplibre-gl.css";
import "./styles.css";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App.js";
import Risk from "./risk/Risk.js";
import Triage from "./triage/Triage.js";

// the triage dashboard by default; ?app=fuel opens the fuel map, ?app=risk the fire risk map
const app = new URLSearchParams(location.search).get("app");
const Root = app === "fuel" ? App : app === "risk" ? Risk : Triage;

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <Root />
  </StrictMode>,
);
